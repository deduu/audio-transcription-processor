import os
import subprocess
from typing import List, Dict, Any
import whisper  # Import Whisper for transcription

def download_and_trim_audio(
    url: str, output_path: str, start_time: str = "00:00:00", end_time: str = None, file_id: int = None
) -> str:
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Use file_id to name the initial downloaded file and the trimmed file
    raw_audio_file = os.path.join(output_path, f"{file_id}_raw.wav" if file_id else "raw_audio.wav")
    trimmed_audio_file = os.path.join(output_path, f"{file_id}.wav" if file_id else "trimmed_audio.wav")

    # Download the full audio
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "-o", raw_audio_file,
        url,
    ]
    
    print(f"Downloading audio from {url}")
    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading audio: {e}")
        raise

    # Trim the audio using ffmpeg
    ffmpeg_command = [
        "ffmpeg",
        "-i", raw_audio_file,
        "-ss", start_time,
        "-to", end_time,
        "-c", "copy",
        trimmed_audio_file,
    ]
    
    print(f"Trimming audio from {start_time} to {end_time}")
    try:
        subprocess.run(ffmpeg_command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error trimming audio: {e}")
        raise

    # Remove the raw audio file to save space
    if os.path.exists(raw_audio_file):
        os.remove(raw_audio_file)

    if os.path.exists(trimmed_audio_file):
        print(f"Trimmed audio saved to: {trimmed_audio_file}")
        return trimmed_audio_file
    else:
        raise FileNotFoundError("Trimmed audio file not created.")



def transcribe_audio_files(audio_files: List[str], output_dir: str) -> List[str]:
    file_and_transcripts = []
    
    # Load the Whisper model
    model = whisper.load_model("base")  # You can choose 'tiny', 'base', 'small', 'medium', 'large'
    
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            try:
                # Transcribe the audio file using Whisper
                print(f"Transcribing {audio_file}...")
                result = model.transcribe(audio_file)
                transcript = result["text"].strip()
                print(f"Transcribed {audio_file}: {transcript}")
            except Exception as e:
                print(f"Error transcribing {audio_file}: {e}")
                continue

            # Build the output line
            relative_audio_path = os.path.relpath(audio_file, output_dir)
            file_and_transcripts.append(f"{relative_audio_path}|{transcript}")
        else:
            print(f"File not found: {audio_file}")
    return file_and_transcripts

def create_transcription_file(file_and_transcripts: List[str], output_file: str):
    with open(output_file, "w") as f:
        for line in file_and_transcripts:
            f.write(f"{line}\n")
    print(f"File '{output_file}' created successfully.")

def main():
    # User specifies the list of URLs and corresponding time ranges
    url_info_list: List[Dict[str, Any]] = [
        {
            'url': "https://www.youtube.com/watch?v=7ARBJQn6QkM",
            'time_ranges': [
                {'start_time': '00:00:10', 'end_time': '00:00:20', 'id': 1},
                {'start_time': '00:01:00', 'end_time': '00:01:10', 'id': 2},
                # Add more time ranges as needed
            ]
        },
        # {
        #     'url': 'https://www.youtube.com/watch?v=VIDEO_ID_2',
        #     'time_ranges': [
        #         {'start_time': '00:02:00', 'end_time': '00:02:10', 'id': 3},
        #         {'start_time': '00:03:00', 'end_time': '00:03:10', 'id': 4},
        #         # Add more time ranges as needed
        #     ]
        # },
        # Add more URLs and time ranges as needed
    ]

    audio_output_dir = "extracted_audio"
    transcription_output_dir = "transcriptions"
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(transcription_output_dir, exist_ok=True)

    audio_files = []
    for info in url_info_list:
        url = info['url']
        time_ranges = info.get('time_ranges', [])
        for time_range in time_ranges:
            start_time = time_range.get('start_time', '00:00:00')
            print(f"start_time: {start_time}")
            end_time = time_range.get('end_time', None)
            print(f"end_time: {end_time}")
            file_id = time_range.get('id', None)
            # Download the audio from the YouTube video in the specified time range
            try:
                audio_file = download_and_trim_audio(url, audio_output_dir, start_time, end_time, file_id)
                audio_files.append(audio_file)
            except Exception as e:
                print(f"Error downloading audio from {url}: {e}")

    # Transcribe the audio files
    file_and_transcripts = transcribe_audio_files(audio_files, transcription_output_dir)

    # Create the transcription file
    output_file = os.path.join(transcription_output_dir, "list.txt")
    create_transcription_file(file_and_transcripts, output_file)

if __name__ == "__main__":
    main()
