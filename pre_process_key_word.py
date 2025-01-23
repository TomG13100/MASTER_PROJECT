import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Fonction pour charger les mots depuis un fichier JSON
def load_words_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = []
    
    # Extraire les mots et synonymes de chaque entrée
    for item in data.get("keywords", []):
        # Ajouter le mot principal
        words.append(item["word"].strip().lower())  
        for synonym in item["synonyms"]:
            if synonym.strip():  # Ignorer les synonymes vides
                words.append(synonym.strip().lower())
    
    print(f"Words loaded from {file_path}: {words}")  # Vérification
    return words

# Chargement des bases de données de mots
sca_words = load_words_from_json("/Users/straudothea/Documents/GitHub/MASTER_PROJECT/base_de_donnees/sca_words.json")
non_sca_words = load_words_from_json("/Users/straudothea/Documents/GitHub/MASTER_PROJECT/base_de_donnees/non_sca_words.json")

# Vérification que les mots sont bien chargés
if not sca_words or not non_sca_words:
    print("Erreur : les listes de mots sont vides. Vérifiez vos fichiers JSON.")
    exit()  # Arrête le script si les listes sont vides

# Fusionner les deux listes pour la vectorisation
all_words = sca_words + non_sca_words

# Vérification de la liste combinée
print(f"All words: {all_words}")

# Vectorisation avec CountVectorizer (comptage simple des mots)
vectorizer = CountVectorizer()

# Appliquer la vectorisation sur les mots
X_bow = vectorizer.fit_transform(all_words)

# Afficher la matrice de mots (bag-of-words)
print(f"Matrice de Bag of Words (CountVectorizer):\n{X_bow.toarray()}")

# Afficher les termes extraits
print(f"Termes extraits par CountVectorizer: {vectorizer.get_feature_names_out()}")

# Vous pouvez aussi utiliser TfidfVectorizer si vous voulez un poids pour chaque mot
tfidf_vectorizer = TfidfVectorizer()

# Appliquer la vectorisation TF-IDF sur les mots
X_tfidf = tfidf_vectorizer.fit_transform(all_words)

# Afficher la matrice TF-IDF
print(f"Matrice TF-IDF:\n{X_tfidf.toarray()}")

# Afficher les termes extraits par TF-IDF
print(f"Termes extraits par TfidfVectorizer: {tfidf_vectorizer.get_feature_names_out()}")
