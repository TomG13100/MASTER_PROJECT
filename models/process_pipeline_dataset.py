import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from transcribe_diarizetest import transcribe_audio
from scripts.pre_process_key_word import all_words, severity_map
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
RAW_AUDIO_DIR = "data/raw2"
OUTPUT_CSV = "data/dataset.csv"
TRANSCRIPTION_DIR = "data/transcriptions_audio"

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)

# Initialize dataset
dataset = []

for condition in os.listdir(RAW_AUDIO_DIR):
    condition_path = os.path.join(RAW_AUDIO_DIR, condition)
    if os.path.isdir(condition_path):
        for filename in os.listdir(condition_path):
            if filename.endswith(".mp3"):
                patient_id = filename.split("-")[0]  # Extract patient ID
                audio_path = os.path.join(condition_path, filename)
                
                # Transcribe the audio using transcribe_diarize.py
                transcription_file = os.path.join(TRANSCRIPTION_DIR, f"{patient_id}.txt")
                if not os.path.exists(transcription_file):
                    print(f"Transcribing {audio_path}...")
                    transcriptions = transcribe_audio(audio_path)
                    transcription = transcribe_audio(audio_path)  # Merge all speakers
                    with open(transcription_file, "w", encoding="utf-8") as f:
                        f.write(transcription)
                else:
                    with open(transcription_file, "r", encoding="utf-8") as f:
                        transcription = f.read()
                
                # Detect keywords
                detected_keywords = {word: severity_map[word] for word in all_words if word in transcription.lower()}
                
                # Add to dataset
                dataset.append({
                    "patient_id": patient_id,
                    "transcription": transcription,
                    "keywords_detected": json.dumps(detected_keywords),
                    "classification": condition  # Classification from folder name
                })

# Convert to DataFrame
df = pd.DataFrame(dataset)

# Tokenization
print("Tokenizing text data...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["transcription"].tolist())
sequences = tokenizer.texts_to_sequences(df["transcription"].tolist())
X = pad_sequences(sequences)
y = np.array([1 if c == "SCA ST+" else 0 for c in df["classification"]])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save dataset as CSV
train_df = pd.DataFrame({"X": list(X_train), "y": y_train})
test_df = pd.DataFrame({"X": list(X_test), "y": y_test})
train_df.to_csv("data/train_dataset.csv", index=False, encoding="utf-8")
test_df.to_csv("data/test_dataset.csv", index=False, encoding="utf-8")

print(f"Training dataset saved to data/train_dataset.csv")
print(f"Testing dataset saved to data/test_dataset.csv")
