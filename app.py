# config.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TimeRange:
    start_time: str
    end_time: Optional[str]
    id: int

@dataclass
class VideoConfig:
    url: str
    time_ranges: List[TimeRange]

@dataclass
class AppConfig:
    audio_output_dir: str = "extracted_audio"
    transcription_output_dir: str = "transcriptions"
    whisper_model: str = "base"
    audio_format: str = "wav"

# audio_processor.py
import os
import subprocess
from typing import Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the pyannote.audio pipeline for diarization
from pyannote.audio import Pipeline

def seconds_to_hms(seconds: float) -> str:
    """Convert seconds (float) to HH:MM:SS.sss format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600 + minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

class SpeakerDiarizer:
    def __init__(self):
        # This loads a pre-trained diarization pipeline.
        # Note: You may need to set the HUGGINGFACE_TOKEN environment variable if required.
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    
    def diarize(self, audio_file: Path):
        """
        Runs speaker diarization on the given audio file.
        Returns a list of segments, each with start time, end time, and speaker label.
        """
        diarization = self.pipeline(str(audio_file))
        segments = []
        # The pipeline returns segments as (start, end) with a speaker label.
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments


class AudioProcessor:
    def __init__(self, output_path: str, audio_format: str = "wav"):
        self.output_path = Path(output_path)
        self.audio_format = audio_format
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _run_command(self, command: List[str], error_message: str) -> None:
        try:
            subprocess.run(command, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"{error_message}: {e.stderr}")
            raise

    def download_audio(self, url: str, file_id: Optional[int]) -> Path:
        output_file = self.output_path / f"{file_id}_raw.{self.audio_format}"
        command = [
            "yt-dlp",
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", self.audio_format,
            "-o", str(output_file),
            url,
        ]
        
        logger.info(f"Downloading audio from {url}")
        self._run_command(command, "Error downloading audio")
        return output_file

    def trim_audio(
        self, 
        input_file: Path, 
        start_time: str, 
        end_time: Optional[str], 
        file_id: Optional[int]
    ) -> Path:
        output_file = self.output_path / f"{file_id}.{self.audio_format}"
        
        command = [
            "ffmpeg",
            "-i", str(input_file),
            "-ss", start_time,
        ]
        
        if end_time:
            command.extend(["-to", end_time])
            
        command.extend([
            "-c", "copy",
            str(output_file)
        ])
        
        logger.info(f"Trimming audio from {start_time} to {end_time}")
        self._run_command(command, "Error trimming audio")
        return output_file

    def process_audio(
        self, 
        url: str, 
        start_time: str, 
        end_time: Optional[str], 
        file_id: Optional[int]
    ) -> Path:
        raw_file = self.download_audio(url, file_id)
        trimmed_file = self.trim_audio(raw_file, start_time, end_time, file_id)
        raw_file.unlink(missing_ok=True)
        return trimmed_file

# transcriber.py
import whisper
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)
        
    def transcribe_file(self, audio_file: Path) -> str:
        try:
            logger.info(f"Transcribing {audio_file}")
            result = self.model.transcribe(str(audio_file))
            transcript = result["text"].strip()
            logger.info(f"Transcription complete for {audio_file}")
            return transcript
        except Exception as e:
            logger.error(f"Error transcribing {audio_file}: {e}")
            raise

class TranscriptionWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_transcriptions(
        self, 
        transcriptions: List[Dict[str, str]], 
        output_file: str = "list.txt"
    ) -> None:
        output_path = self.output_dir / output_file
        with output_path.open("w") as f:
            for entry in transcriptions:
                f.write(f"{entry['file']}|{entry['transcript']}\n")
        logger.info(f"Transcriptions written to {output_path}")

# main.py
from typing import List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def process_videos(config: AppConfig, videos: List[VideoConfig]) -> None:
    audio_processor = AudioProcessor(config.audio_output_dir, config.audio_format)
    transcriber = Transcriber(config.whisper_model)
    writer = TranscriptionWriter(Path(config.transcription_output_dir))
    diarizer = SpeakerDiarizer()  # Initialize diarization pipeline
    
    transcriptions = []
    
    for video in videos:
        for time_range in video.time_ranges:
            try:
                # Process audio
                audio_file = audio_processor.process_audio(
                    video.url,
                    time_range.start_time,
                    time_range.end_time,
                    time_range.id
                )

                logger.info(f"Running diarization on {audio_file}")
                # Run speaker diarization on the trimmed audio file.
                diarization_segments = diarizer.diarize(audio_file)

                speaker_transcriptions = []
                # For each diarized segment, extract that audio and transcribe it.
                for i, segment in enumerate(diarization_segments):
                    seg_start = seconds_to_hms(segment["start"])
                    seg_end = seconds_to_hms(segment["end"])
                    # Use a compound id (e.g. "1_0", "1_1", …) for the segment file name.
                    segment_file = audio_processor.trim_audio(
                        audio_file,
                        seg_start,
                        seg_end,
                        file_id=f"{time_range.id}_{i}"
                    )
                    transcript_segment = transcriber.transcribe_file(segment_file)
                    # Optionally, remove the segment file after transcription.
                    segment_file.unlink(missing_ok=True)
                    speaker_transcriptions.append(f"{segment['speaker']}: {transcript_segment}")
                
                # Combine the diarized segment transcripts into one full transcript.
                combined_transcript = "\n".join(speaker_transcriptions)
                
                # Transcribe audio
                # transcript = transcriber.transcribe_file(audio_file)
                
                # Store result
                transcriptions.append({
                    "file": os.path.relpath(audio_file, start=config.transcription_output_dir),
                    "transcript": combined_transcript
                })
                # transcriptions.append({
                #     "file": audio_file.relative_to(config.transcription_output_dir),
                #     "transcript": transcript
                # })
                  # Optionally, remove the original trimmed audio file if not needed.
                audio_file.unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"Error processing video {video.url} at {time_range}: {e}")
                continue
    
    # Write all transcriptions
    writer.write_transcriptions(transcriptions)

def main():
    # Example configuration
    config = AppConfig()
    
    videos = [
        VideoConfig(
            url="https://www.youtube.com/watch?v=7ARBJQn6QkM",
            time_ranges=[
                TimeRange(start_time="00:00:00", end_time="00:10:00", id=1),
  
            ]
        ),
        
    ]
    
    process_videos(config, videos)

if __name__ == "__main__":
    main()