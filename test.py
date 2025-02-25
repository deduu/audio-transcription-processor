from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline (with optional progress hook)
with ProgressHook() as hook:
    diarization = pipeline("./extracted_audio/3.wav", hook=hook)

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...