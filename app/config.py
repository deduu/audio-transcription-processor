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
class LocalAudioConfig:
    path: str  # local path to the audio file (e.g. "path/to/file.wav")
    time_ranges: List[TimeRange]


@dataclass
class AppConfig:
    audio_output_dir: str = "extracted_audio"
    transcription_output_dir: str = "transcriptions"
    whisper_model: str = "base"
    audio_format: str = "wav"