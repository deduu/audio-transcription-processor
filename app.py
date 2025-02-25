import sys
import asyncio

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import os
import tempfile
from pathlib import Path
import time
import subprocess
import json
import re
from typing import List, Optional
import whisper
from pyannote.audio import Pipeline

# Set page configuration
st.set_page_config(
    page_title="Audio Transcription & Diarization",
    page_icon="🎙️",
    layout="wide"
)

# Apply custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
        max-width: 1200px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .output-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px;
        background-color: #f9f9f9;
        margin-top: 20px;
    }
    .speaker-label {
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Define a simple TimeRange data class (not used for processing but available for future config)
class TimeRange:
    def __init__(self, start_time: str, end_time: Optional[str], id: int):
        self.start_time = start_time
        self.end_time = end_time
        self.id = id

# Cache the diarization pipeline so it loads only once per session.
@st.cache_resource
def get_diarization_pipeline():
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token is None:
        st.error("HUGGINGFACE_TOKEN environment variable is not set.")
        return None
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        return pipeline
    except Exception as e:
        st.error(f"Error loading diarization pipeline: {e}")
        return None

# Audio processing functions
def seconds_to_hms(seconds: float) -> str:
    """Convert seconds (float) to HH:MM:SS.sss format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600 + minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def hms_to_seconds(hms: str) -> float:
    """Convert HH:MM:SS or HH:MM:SS.sss format to seconds."""
    if '.' in hms:
        time_part, ms_part = hms.split('.')
        ms = float(f"0.{ms_part}")
    else:
        time_part = hms
        ms = 0.0
    parts = time_part.split(':')
    if len(parts) == 3:
        h, m, s = parts
        seconds = int(h) * 3600 + int(m) * 60 + int(s) + ms
    elif len(parts) == 2:
        m, s = parts
        seconds = int(m) * 60 + int(s) + ms
    else:
        seconds = int(parts[0]) + ms
    return seconds

def run_command(command: List[str]) -> subprocess.CompletedProcess:
    """Run a shell command and return its result."""
    return subprocess.run(command, check=True, text=True, capture_output=True)

def create_temp_file(uploaded_file):
    """Create a temporary file from an uploaded file."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def trim_audio(input_file, start_time, end_time, output_file):
    """Trim audio using ffmpeg and re-encode to 16 kHz mono WAV."""
    command = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-ss", start_time
    ]
    if end_time:
        command.extend(["-to", end_time])
    command.extend([
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_file
    ])
    try:
        run_command(command)
        return output_file
    except subprocess.CalledProcessError as e:
        st.error(f"Error trimming audio: {e.stderr}")
        return None

def download_youtube_audio(url, output_file):
    """Download audio from YouTube using yt-dlp."""
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

def run_diarization(audio_file):
    """
    Run speaker diarization on the given audio file using pyannote.audio.
    """
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        return []
    try:
        diarization = pipeline(audio_file)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments
    except Exception as e:
        st.error(f"Error during diarization: {str(e)}")
        return []

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)
    
    def transcribe(self, audio_file):
        """Transcribe an audio file using Whisper."""
        result = self.model.transcribe(audio_file)
        return result["text"].strip()

def display_header():
    """Display the header with title and description."""
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f399.png", width=80)
    with col2:
        st.title("Audio Transcription & Speaker Diarization")
        st.markdown("Process audio files or YouTube videos to get transcriptions with speaker identification.")

def get_whisper_models():
    """Return available Whisper models."""
    return ["tiny", "base", "small", "medium", "large"]

def display_transcription(transcriptions):
    """Display transcription results with speaker labels."""
    if not transcriptions:
        st.warning("No transcription results available.")
        return
    st.subheader("Transcription Results")
    with st.container():
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        speakers = list(set(t["speaker"] for t in transcriptions))
        colors = ["#FF9AA2", "#FFDAC1","#FFB7B2", "#E2F0CB", "#B5EAD7", "#C7CEEA"]
        speaker_colors = {speakers[i]: colors[i % len(colors)] for i in range(len(speakers))}
        for segment in transcriptions:
            speaker = segment["speaker"]
            text = segment["text"]
            start_time = segment["start"]
            end_time = segment["end"]
            timestamp = f"{start_time:.2f}s - {end_time:.2f}s"
            speaker_html = f'<span class="speaker-label" style="background-color: {speaker_colors[speaker]};">{speaker}</span>'
            st.markdown(f'{speaker_html} <small>({timestamp})</small> {text}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def process_audio(file_path, start_time, end_time, whisper_model="base"):
    """Process an audio file: trim, diarize, segment, and transcribe."""
    temp_dir = tempfile.mkdtemp()
    trimmed_path = os.path.join(temp_dir, "trimmed_audio.wav")
    trimmed_file = trim_audio(file_path, start_time, end_time, trimmed_path)
    if not trimmed_file:
        return []
    # Run real speaker diarization via pyannote.audio
    diarization_segments = run_diarization(trimmed_file)
    transcriber = WhisperTranscriber(whisper_model)
    transcriptions = []
    for i, segment in enumerate(diarization_segments):
        seg_start = seconds_to_hms(segment["start"])
        seg_end = seconds_to_hms(segment["end"])
        segment_path = os.path.join(temp_dir, f"segment_{i}.wav")
        segment_file = trim_audio(trimmed_file, seg_start, seg_end, segment_path)
        if segment_file:
            transcript = transcriber.transcribe(segment_file)
            transcriptions.append({
                "speaker": segment["speaker"],
                "text": transcript,
                "start": segment["start"],
                "end": segment["end"]
            })
            os.remove(segment_file)
    os.remove(trimmed_file)
    return transcriptions

def validate_time_format(time_str):
    """Validate time format (HH:MM:SS or MM:SS)."""
    pattern = r'^([0-9]{1,2}:)?[0-5]?[0-9]:[0-5][0-9](\.[0-9]{1,3})?$'
    return re.match(pattern, time_str) is not None

def main():
    display_header()
    # Create tabs for audio file upload and YouTube URL
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎥 YouTube URL"])
    st.sidebar.header("Configuration")
    whisper_model = st.sidebar.selectbox("Whisper Model", get_whisper_models(), index=1)
    
    # Upload Audio tab
    with tab1:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.audio(uploaded_file)
            with col2:
                st.subheader("Time Range")
                start_time = st.text_input("Start Time (HH:MM:SS)", "00:00:00")
                end_time_enabled = st.checkbox("Set End Time", value=True)
                end_time = st.text_input("End Time (HH:MM:SS)", "00:00:30") if end_time_enabled else None
            if st.button("Process Audio", key="process_upload"):
                if not validate_time_format(start_time):
                    st.error("Invalid start time format. Use HH:MM:SS or MM:SS.")
                    return
                if end_time and not validate_time_format(end_time):
                    st.error("Invalid end time format. Use HH:MM:SS or MM:SS.")
                    return
                with st.spinner("Processing audio..."):
                    temp_path = create_temp_file(uploaded_file)
                    transcriptions = process_audio(temp_path, start_time, end_time, whisper_model)
                    display_transcription(transcriptions)
                    if transcriptions:
                        download_text = "\n\n".join([
                            f"{t['speaker']} ({t['start']:.2f}s - {t['end']:.2f}s): {t['text']}"
                            for t in transcriptions
                        ])
                        st.download_button(
                            label="Download Transcription",
                            data=download_text,
                            file_name=f"transcription_{int(time.time())}.txt",
                            mime="text/plain"
                        )
    
    # YouTube URL tab
    with tab2:
        st.header("YouTube Video")
        youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        if youtube_url:
            st.subheader("Time Range")
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.text_input("Start Time (HH:MM:SS)", "00:00:00", key="yt_start")
            with col2:
                end_time_enabled = st.checkbox("Set End Time", value=True, key="yt_end_enable")
                end_time = st.text_input("End Time (HH:MM:SS)", "00:05:00", key="yt_end") if end_time_enabled else None
            if youtube_url.startswith("https://"):
                st.info("Note: The application will download the audio from this YouTube video, which may take some time.")
            if st.button("Process Video", key="process_yt"):
                if not validate_time_format(start_time):
                    st.error("Invalid start time format. Use HH:MM:SS or MM:SS.")
                    return
                if end_time and not validate_time_format(end_time):
                    st.error("Invalid end time format. Use HH:MM:SS or MM:SS.")
                    return
                if not youtube_url.startswith("https://"):
                    st.error("Please enter a valid YouTube URL.")
                    return
                with st.spinner("Downloading and processing video..."):
                    temp_dir = tempfile.mkdtemp()
                    output_file = os.path.join(temp_dir, "youtube_audio.wav")
                    audio_file = download_youtube_audio(youtube_url, output_file)
                    if audio_file:
                        transcriptions = process_audio(audio_file, start_time, end_time, whisper_model)
                        display_transcription(transcriptions)
                        if transcriptions:
                            download_text = "\n\n".join([
                                f"{t['speaker']} ({t['start']:.2f}s - {t['end']:.2f}s): {t['text']}"
                                for t in transcriptions
                            ])
                            st.download_button(
                                label="Download Transcription",
                                data=download_text,
                                file_name=f"youtube_transcription_{int(time.time())}.txt",
                                mime="text/plain"
                            )
                    if os.path.exists(audio_file):
                        os.remove(audio_file)

if __name__ == "__main__":
    main()
