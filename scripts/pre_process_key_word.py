import json


import scripts.pre_process_key_word

def load_keywords(file_path):
    """
    Charge la base de données de mots-clés et extrait tous les termes importants.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    severity_map = {}  # Associe chaque mot-clé à sa sévérité

    for item in data["keywords"]:
        word = item["word"]
        synonyms = item["synonyms"]
        severity = item["severity"]

        # Ajouter le mot principal et ses synonymes
        words.append(word)
        words.extend(synonyms)

        # Associer chaque mot à sa sévérité
        severity_map[word] = severity
        for synonym in synonyms:
            severity_map[synonym] = severity

    return words, severity_map

# Charger les mots-clés et leur sévérité
sca_words, severity_sca = load_keywords("data/sca_words.json")
non_sca_words, severity_non_sca = load_keywords("data/non_sca_words.json")

# Fusionner toutes les données
all_words = sca_words + non_sca_words
severity_map = {**severity_sca, **severity_non_sca}

# Vérification après chargement
if not all_words:
    raise ValueError(" Erreur : `all_words` est vide ! Vérifiez `sca_words.json` et `non_sca_words.json`.")

if not severity_map:
    raise ValueError(" Erreur : `severity_map` est vide !")

# Affichage pour debug
print(f" Nombre total de mots-clés : {len(all_words)}")
print(f" Exemple de mots-clés : {all_words[:5]}")
print(f" Exemple de sévérité : {list(severity_map.items())[:5]}")
