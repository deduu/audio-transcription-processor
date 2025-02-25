# app/ui.py
import streamlit as st
import os
import tempfile
import time
from pathlib import Path
import re

from app.audio_processor import trim_audio, download_youtube_audio, run_command
from app.transcriber import Transcriber
from app.diarizer import SpeakerDiarizer

def set_page_config():
    st.set_page_config(
        page_title="Audio Transcription & Diarization",
        page_icon="🎙️",
        layout="wide"
    )

def apply_custom_styles():
    st.markdown("""
    <style>
        /* Your custom CSS styles */
    </style>
    """, unsafe_allow_html=True)

def display_header():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f399.png", width=80)
    with col2:
        st.title("Audio Transcription & Speaker Diarization")
        st.markdown("Process audio files or YouTube videos to get transcriptions with speaker identification.")

def validate_time_format(time_str: str) -> bool:
    pattern = r'^([0-9]{1,2}:)?[0-5]?[0-9]:[0-5][0-9](\.[0-9]{1,3})?$'
    return re.match(pattern, time_str) is not None

def display_transcription(transcriptions):
    if not transcriptions:
        st.warning("No transcription results available.")
        return
    st.subheader("Transcription Results")
    with st.container():
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        speakers = list(set(t["speaker"] for t in transcriptions))
        colors = ["#FF9AA2", "#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA"]
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
