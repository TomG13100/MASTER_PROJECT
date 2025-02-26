import re
import json

# Charger la base de données de keywords
with open("base_de_donnees/sca_words.json", "r", encoding="utf-8") as f:
    sca_data = json.load(f)

with open("base_de_donnees/non_sca_words.json", "r", encoding="utf-8") as f:
    non_sca_data = json.load(f)

# Extraire les symptômes et leurs scores
symptomes_severite = {}
for item in sca_data["keywords"]:
    symptomes_severite[item["word"]] = item["severity"]
    for synonym in item["synonyms"]:
        symptomes_severite[synonym] = item["severity"]

# Extraire les symptômes non-SCA et leurs scores
non_sca_severite = {}
for item in non_sca_data["keywords"]:
    non_sca_severite[item["word"]] = item["severity"]
    for synonym in item["synonyms"]:
        non_sca_severite[synonym] = item["severity"]

# Lire les symptômes détectés dans le fichier texte
with open("symptomes.txt", "r", encoding="utf-8") as f:
    texte = f.read().lower()

# Afficher le texte
print("Transcription :", texte)

# Extraction de l'âge du patient
match_age = re.search(r"(\d{2})\s*ans", texte)
age = int(match_age.group(1)) if match_age else None

# Détection du genre
homme = bool(re.search(r"\b(il|homme|monsieur)\b", texte))
femme = bool(re.search(r"\b(elle|femme|madame)\b", texte))

# Vérifier l'âge critique
age_critique = (homme and age and age >= 50) or (femme and age and age >= 55)

# Détection des symptômes
symptomes_detectes = [mot for mot in symptomes_severite.keys() if mot in texte]
symptomes_non_sca = [mot for mot in non_sca_severite.keys() if mot in texte]

print("Symptômes reconnus :", symptomes_detectes)
print("Symptômes non-SCA reconnus :", symptomes_non_sca)

# Calcul du score de sévérité
score_total = sum(symptomes_severite[s] for s in symptomes_detectes) - sum(non_sca_severite[s] for s in symptomes_non_sca)

# Ajouter un bonus de score si l'âge est critique
if age_critique:
    score_total += 5

# Empêcher un score négatif
score_total = max(score_total, 0)

# Affichage des résultats
print("Âge détecté :", age)
print("Genre détecté :", "Homme" if homme else "Femme" if femme else "Inconnu")
print("Score total de sévérité ajusté :", score_total)
