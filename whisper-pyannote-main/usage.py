import whisper
import os
print(os.getcwd())
import json

from pyannote.audio import Pipeline
from utils import words_per_segment
# Chemin vers le fichier config.json
CONFIG_FILE = "secrets.json"

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token="HF_AUTH_TOKEN"
)
audio_path = "data/raw/Audio-SCA-1.wav"

model = whisper.load_model("small")
diarization_result = pipeline(audio_path)
transcription_result = model.transcribe(audio_path, word_timestamps=True)

final_result = words_per_segment(transcription_result, diarization_result)

for _, segment in final_result.items():
    print(
        f'{segment["start"]:.3f}\t{segment["end"]:.3f}\t {segment["speaker"]}\t{segment["text"]}'
    )
