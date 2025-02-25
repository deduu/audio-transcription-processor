# audio_processor.py
import os
import subprocess
import tempfile
from typing import List, Optional
import streamlit as st

def run_command(command: List[str]) -> subprocess.CompletedProcess:
    """
    Run a shell command and return its result.
    Raises CalledProcessError if it fails.
    """
    return subprocess.run(command, check=True, text=True, capture_output=True)

def create_temp_file(uploaded_file):
    """
    Create a temporary file from a Streamlit uploaded file.
    Returns the path to the temp file.
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def trim_audio(input_file: str, start_time: str, end_time: Optional[str], output_file: str) -> Optional[str]:
    """
    Trim audio using ffmpeg and re-encode to 16 kHz mono WAV.
    Returns the path to the trimmed file or None on error.
    """
    command = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-ss", start_time
    ]
    if end_time:
        command.extend(["-to", end_time])
    command.extend([
        "-vn",                   # no video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",          # 16 kHz sample rate
        "-ac", "1",              # mono
        output_file
    ])
    try:
        run_command(command)
        return output_file
    except subprocess.CalledProcessError as e:
        st.error(f"Error trimming audio: {e.stderr}")
        return None

def download_youtube_audio(url: str, output_file: str) -> Optional[str]:
    """
    Download audio from YouTube using yt-dlp in WAV format.
    Returns the path to the downloaded file or None on error.
    """
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "-o", output_file,
        url,
    ]
    try:
        run_command(command)
        return output_file
    except subprocess.CalledProcessError as e:
        st.error(f"Error downloading audio: {e.stderr}")
        return None
