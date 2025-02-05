import os
import json
import numpy as np
import subprocess
from tensorflow.keras.models import load_model
from scripts.pre_process_key_word import all_words, severity_map
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

transcription_dir = "data/transcriptions_audio"
# Run transcribe_diarize.py
# Vérifier si le dossier existe et contient au moins un fichier
if os.path.exists(transcription_dir) and any(os.listdir(transcription_dir)):
    print("Transcriptions déjà présentes. Skipping transcribe_diarize.py.")
else:
    print(f"Executing transcribe_diarize.py to generate transcriptions...")
    subprocess.run(["python", "models/transcribe_diarize.py"], check=True)



import h5py

# Charger le modèle .h5 et supprimer batch_shape
with h5py.File("models/sca_model.h5", "r+") as f:
    if "model_config" in f.attrs:
        model_config = f.attrs["model_config"]
        if isinstance(model_config, bytes):  # Si c'est un byte, on le décode
            model_config = model_config.decode("utf-8")

        model_config = model_config.replace('"batch_shape"', '"input_shape"')
        f.attrs["model_config"] = model_config.encode("utf-8")

from tensorflow.keras.mixed_precision import Policy

# Charger le modèle en précisant les objets personnalisés
with custom_object_scope({'DTypePolicy': Policy}):
    model = load_model("models/sca_model.h5", compile=False)

# Sauvegarde le modèle corrigé au format Keras
model.save("models/sca_model.keras")

# Charger le modèle corrigé
with custom_object_scope({'DTypePolicy': Policy}):
    model = load_model("models/sca_model.keras", compile=False)




def predict_from_transcriptions(transcription_dir, model_path):
    """
    Analyse les transcriptions des speakers, détecte les symptômes et prédit le risque SCA.
    """
    print("Début de l'analyse des transcriptions...")

    files = [f for f in os.listdir(transcription_dir) if f.endswith(".txt")]
    tokenizer = Tokenizer()

    results = {}

    for file in files:
        with open(os.path.join(transcription_dir, file), "r", encoding="utf-8") as f:
            text = f.read()

        # Vérifier si un symptôme clé est détecté
        detected_symptoms = {word: severity_map[word] for word in all_words if word in text.lower()}

        # Calculer un score de gravité moyen
        avg_severity = np.mean(list(detected_symptoms.values())) if detected_symptoms else 0

        # Prédiction IA basée sur le texte
        tokenizer.fit_on_texts([text])
        sequence = tokenizer.texts_to_sequences([text])
        X_test = pad_sequences(sequence, maxlen=5)

        prediction = model.predict(X_test)
        probability_sca = prediction[0][1]

        # Résumé des résultats
        results[file] = {
            "probabilité_SCA": float(round(probability_sca, 2)),  # Convertir float32 en float standard
            "symptômes_detectés": detected_symptoms,
            "score_de_gravité": float(round(avg_severity, 2)),  # Convertir en float standard
            "conclusion": "SCA détecté" if probability_sca > 0.5 else "Pas de SCA"
        }

    #  Enregistrer les résultats dans un fichier JSON
    with open("data/transcriptions_audio/diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(" Analyse terminée ! Résultats enregistrés dans `data/transcriptions_audio/diagnostic_results.json`")

# Exécuter la prédiction
predict_from_transcriptions("data/transcriptions_audio/", "models/sca_model.keras")
