# diarizer.py
import os
import streamlit as st
from pyannote.audio import Pipeline

@st.cache_resource
def get_diarization_pipeline():
    """
    Cache the pyannote pipeline so it's loaded only once.
    Requires HUGGINGFACE_TOKEN environment variable.
    """
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token is None:
        st.error("HUGGINGFACE_TOKEN environment variable is not set.")
        return None
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        return pipeline
    except Exception as e:
        st.error(f"Error loading diarization pipeline: {e}")
        return None

def run_diarization(audio_file: str):
    """
    Run speaker diarization on the given audio file.
    Returns a list of dicts with {start, end, speaker}.
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
