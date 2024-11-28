import os
import subprocess
from typing import List, Dict, Any
import whisper  # Import Whisper for transcription

def download_youtube_audio(url: str, output_path: str, start_time: str = "00:00:00", end_time: str = None, file_id: int = None) -> str:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Use file_id to name the output file uniquely
    output_file = os.path.join(output_path, f"{file_id}.%(ext)s" if file_id else '%(title)s.%(ext)s')
    
    # Build the yt-dlp command
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "-o", output_file
    ]
    
    # If start_time and end_time are provided, use --download-sections
    if start_time and end_time:
        # Using --download-sections "*start-end"
        command.extend(["--download-sections", f"*{start_time}-{end_time}"])
    elif start_time:
        # Only start_time is provided
        command.extend(["--postprocessor-args", f"-ss {start_time}"])
    elif end_time:
        # Only end_time is provided
        command.extend(["--postprocessor-args", f"-to {end_time}"])
    
    # Add the URL
    command.append(url)
    
    # Running the yt-dlp command to download the audio
    print(f"Downloading audio from {url} between {start_time} and {end_time}")
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Expected output file
    expected_filename = f"{file_id}.wav" if file_id else None
    if expected_filename and os.path.exists(os.path.join(output_path, expected_filename)):
        audio_file = os.path.join(output_path, expected_filename)
        print(f"Downloaded audio: {audio_file}")
        return audio_file
    else:
        # If file_id is not specified, try to find any .wav file in the output directory
        downloaded_files = [f for f in os.listdir(output_path) if f.endswith(".wav")]
        if downloaded_files:
            audio_file = os.path.join(output_path, downloaded_files[0])
            print(f"Downloaded audio: {audio_file}")
            return audio_file
        else:
            raise FileNotFoundError("Audio file not found in the specified directory.")

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
            'url': 'https://www.youtube.com/watch?v=dAI12OGD04A',
            'time_ranges': [
                {'start_time': '00:00:00', 'end_time': '00:00:10', 'id': 1},
                {'start_time': '00:00:15', 'end_time': '00:00:20', 'id': 2},
                # Add more time ranges as needed
            ]
        }
        # {
        #     'url': 'https://www.youtube.com/watch?v=pC1QH-0BhmE',
        #     'time_ranges': [
        #         {'start_time': '00:00:44', 'end_time': '00:00:54', 'id': 3},
        #         {'start_time': '00:01:00', 'end_time': '00:01:10', 'id': 4},
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
            end_time = time_range.get('end_time', None)
            file_id = time_range.get('id', None)
            # Download the audio from the YouTube video in the specified time range
            try:
                audio_file = download_youtube_audio(url, audio_output_dir, start_time, end_time, file_id)
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
