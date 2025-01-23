import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Fonction pour charger les mots depuis un fichier JSON
def load_words_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [word.strip().lower() for word in data.get("words", [])]

# Chargement des bases de données de mots
sca_words = load_words_from_json("/Users/straudothea/Documents/GitHub/MASTER_PROJECT/base_de_donnees/sca_words.json")
non_sca_words = load_words_from_json("/Users/straudothea/Documents/GitHub/MASTER_PROJECT/base_de_donnees/non_sca_words.json")

# Vérification des chargements
print("SCA Words:", sca_words)
print("Non-SCA Words:", non_sca_words)

# Combinaison des mots pour vectorisation
all_words = sca_words + non_sca_words

# Vectorisation avec CountVectorizer
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(all_words)
print("Bag-of-Words:\n", X_bow.toarray())

# Vectorisation avec TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(all_words)
print("TF-IDF:\n", X_tfidf.toarray())
