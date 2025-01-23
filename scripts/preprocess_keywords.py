import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def preprocess_keywords(csv_path):
    # Charger le fichier CSV
    data = pd.read_csv(csv_path)
    texts = data['text']
    labels = data['class'].map({'sca': 1, 'non_sca': 0})  # Convertir les classes en entiers

    # Vectorisation avec CountVectorizer
    vectorizer = CountVectorizer(max_features=1000)  # Limite à 1000 mots les plus fréquents
    X = vectorizer.fit_transform(texts).toarray()
    y = labels.values

    # Diviser les données en ensemble d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, vectorizer
