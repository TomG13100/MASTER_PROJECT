import os
import json
import numpy as np
import subprocess
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
from transcribe_diarizetest import transcribe_audio
from scripts.pre_process_key_word import all_words, severity_map

# Paths
AUDIO_TO_PREDICT = "data/raw/Audio-SCA-1.wav"
TRANSCRIPTION_DIR = "data/transcriptions_audio"
MODEL_PATH = "models/sca_model.keras"
OUTPUT_JSON = "data/results/prediction_result.json"

# Ensure transcription directory exists
os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)

# Step 1: Transcribe and diarize the new audio
print("Executing transcribe_diarize.py for prediction...")
transcriptions = transcribe_audio(AUDIO_TO_PREDICT)
transcription_text = " ".join(transcriptions.values())  # Merge all speakers
transcription_file = os.path.join(TRANSCRIPTION_DIR, "Audio-SCA-1.txt")

# Save transcription
with open(transcription_file, "w", encoding="utf-8") as f:
    f.write(transcription_text)

# Step 2: Detect symptoms
detected_symptoms = {word: severity_map[word] for word in all_words if word in transcription_text.lower()}
avg_severity = np.mean(list(detected_symptoms.values())) if detected_symptoms else 0

# Tokenize and prepare input for model
tokenizer = Tokenizer()
tokenizer.fit_on_texts([transcription_text])
sequence = tokenizer.texts_to_sequences([transcription_text])
X_test = pad_sequences(sequence, maxlen=5)

# Load model and predict
with custom_object_scope({'DTypePolicy': tf.keras.mixed_precision.Policy}):
    model = load_model(MODEL_PATH, compile=False)

prediction = model.predict(X_test)
probability_sca = prediction[0][1]

# Save prediction result
result = {
    "probabilité_SCA": float(round(probability_sca, 2)),
    "symptômes_detectés": detected_symptoms,
    "score_de_gravité": float(round(avg_severity, 2)),
    "conclusion": "SCA détecté" if probability_sca > 0.5 else "Pas de SCA"
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"Prediction result saved to {OUTPUT_JSON}")
