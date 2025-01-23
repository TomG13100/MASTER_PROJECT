from models.mlp_model import build_cnn_model
from scripts.preprocess_audio import preprocess_audio
from sklearn.model_selection import train_test_split
import tensorflow as tf

def train_model(audio_dir, labels_csv, model_save_path):
    # Prétraiter les données
    spectrograms, targets = preprocess_audio(audio_dir, 'data/processed', labels_csv)
    spectrograms = spectrograms / 255.0  # Normalisation

    # Diviser les données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(spectrograms, targets, test_size=0.2, random_state=42)

    # Construire le modèle
    input_shape = X_train[0].shape
    num_classes = len(set(y_train))
    model = build_cnn_model(input_shape, num_classes)

    # Entraîner le modèle
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

    # Sauvegarder le modèle
    model.save(model_save_path)
    print(f"Modèle entraîné et sauvegardé à {model_save_path}")
