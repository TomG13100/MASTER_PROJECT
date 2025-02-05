import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from mlp_model import build_mlp_model

# Charger les bases de données de mots-clés
def load_keywords(file_path):
    """
    Charge la base de données de mots-clés et extrait les termes sous forme de liste de mots.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    severity_map = {}  # Associe chaque mot-clé à sa sévérité

    for item in data["keywords"]:
        word = item["word"]
        synonyms = item["synonyms"]
        severity = item["severity"]

        # Vérifier si "word" est une chaîne et non un dictionnaire
        if isinstance(word, str):
            words.append(word)
            severity_map[word] = severity

        # Vérifier si "synonyms" est une liste de chaînes et non un dict
        if isinstance(synonyms, list):
            for synonym in synonyms:
                if isinstance(synonym, str) and synonym.strip():  # Exclure les chaînes vides
                    words.append(synonym)
                    severity_map[synonym] = severity

    return words, severity_map

# Charger les mots-clés SCA et non-SCA
sca_words, severity_sca = load_keywords("data/sca_words.json")
non_sca_words, severity_non_sca = load_keywords("data/non_sca_words.json")

# Fusionner les bases de données
all_words = sca_words + non_sca_words
severity_map = {**severity_sca, **severity_non_sca}

# Vérification après extraction
print(f"Nombre total de mots-clés : {len(all_words)}")
print(f"Exemple de mots-clés : {all_words[:5]}")
print(f"Exemple de sévérité : {list(severity_map.items())[:5]}")

# Vérification finale
if any(isinstance(item, dict) for item in all_words):
    raise TypeError(" ERREUR : `all_words` contient encore des dictionnaires.")

# Création des labels
labels = [1] * len(sca_words) + [0] * len(non_sca_words)

# Tokenisation des mots-clés
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_words)  # Maintenant `all_words` est une liste de chaînes

sequences = tokenizer.texts_to_sequences(all_words)
X = pad_sequences(sequences)

# Séparer en données d'entraînement et validation
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

# Construire le modèle
input_shape = (X_train.shape[1],)
print("input_shape", input_shape)
model = build_mlp_model(input_shape, num_classes=2)

# Entraîner le modèle
model.fit(X_train, np.array(y_train), validation_data=(X_val, np.array(y_val)), epochs=10, batch_size=8)

# Sauvegarder le modèle entraîné
model_save_path = "models/sca_model.keras"
model.save(model_save_path)

# Vérifier si le fichier a bien été créé
import os
if os.path.exists(model_save_path):
    print(f"Modèle entraîné et sauvegardé avec succès à {model_save_path}")
else:
    print("Erreur : Le modèle n'a pas été sauvegardé correctement.")
