import tensorflow as tf
import pickle

def predict_keywords(texts, model_path):
    # Charger le modèle et le vectorizer
    model = tf.keras.models.load_model(model_path)
    with open(model_path.replace('.h5', '_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    # Vectoriser les nouveaux textes
    X = vectorizer.transform(texts).toarray()

    # Faire les prédictions
    predictions = model.predict(X)
    return predictions
