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
# Save each speaker's transcription into separate files
output_dir = "data/transcriptions audio"
os.makedirs(output_dir, exist_ok=True)

speaker_files = {}

# Write transcriptions into respective files
for _, segment in final_result.items():
    speaker = segment["speaker"]
    if speaker not in speaker_files:
        speaker_files[speaker] = open(os.path.join(output_dir, f"{speaker}.txt"), "w", encoding="utf-8")
    speaker_files[speaker].write(f'{segment["text"]}\n')

# Close all file handles
for file in speaker_files.values():
    file.close()

print(f"Transcriptions saved to: {output_dir}")

for _, segment in final_result.items():
    print(
        f'{segment["start"]:.3f}\t{segment["end"]:.3f}\t {segment["speaker"]}\t{segment["text"]}'
    )
