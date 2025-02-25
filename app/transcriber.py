# transcriber.py
import whisper

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        """
        Load the specified Whisper model.
        Model can be 'tiny', 'base', 'small', 'medium', or 'large'.
        """
        self.model = whisper.load_model(model_name)
    
    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe an audio file using Whisper.
        Returns the text transcript.
        """
        result = self.model.transcribe(audio_file)
        return result["text"].strip()
