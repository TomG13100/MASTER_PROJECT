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
    symptomes_detectes = [ligne.strip().lower() for ligne in f.readlines()]

# Afficher les symptômes détectés
print("Symptômes détectés :", symptomes_detectes)

# Vérifier quels symptômes sont reconnus
symptomes_reconnus = [s for s in symptomes_detectes if s in symptomes_severite]
symptomes_non_sca = [s for s in symptomes_detectes if s in non_sca_severite]

print("Symptômes reconnus :", symptomes_reconnus)
print("Symptômes non-SCA reconnus :", symptomes_non_sca)

# Calcul du score
score_total = sum(symptomes_severite.get(s, 0) for s in symptomes_reconnus) - sum(non_sca_severite.get(s, 0) for s in symptomes_non_sca)

# Empêcher un score négatif
score_total = max(score_total, 0)

print(f"Score total de sévérité ajusté : {score_total}")

