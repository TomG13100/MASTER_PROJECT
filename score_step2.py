import re
import json

# === Charger la base de données de keywords ===
with open("base_de_donnees/sca_words.json", "r", encoding="utf-8") as f:
    sca_data = json.load(f)

with open("base_de_donnees/non_sca_words.json", "r", encoding="utf-8") as f:
    non_sca_data = json.load(f)

# === Extraire les scores de sévérité ===
symptomes_severite = {}
for item in sca_data["keywords"]:
    symptomes_severite[item["word"]] = item["severity"]
    for synonym in item["synonyms"]:
        symptomes_severite[synonym] = item["severity"]

non_sca_severite = {}
for item in non_sca_data["keywords"]:
    non_sca_severite[item["word"]] = item["severity"]
    for synonym in item["synonyms"]:
        non_sca_severite[synonym] = item["severity"]

# === Lire la transcription ===
with open("symptomes.txt", "r", encoding="utf-8") as f:
    texte = f.read().lower()

print("Transcription :", texte)

# === Extraction âge ===
match_age = re.search(r"(\d{2})\s*ans", texte)
age = int(match_age.group(1)) if match_age else None

# === Genre ===
homme = bool(re.search(r"\b(il|homme|monsieur)\b", texte))
femme = bool(re.search(r"\b(elle|femme|madame)\b", texte))

# === Âge critique ===
age_critique = (homme and age and age >= 50) or (femme and age and age >= 55)

# === Symptômes détectés ===
symptomes_detectes = [mot for mot in symptomes_severite if mot in texte]
symptomes_non_sca = [mot for mot in non_sca_severite if mot in texte]

print("Symptômes SCA détectés :", symptomes_detectes)
print("Symptômes non-SCA détectés :", symptomes_non_sca)

# === Score brut ===
score_total = sum(symptomes_severite[s] for s in symptomes_detectes) - sum(non_sca_severite[s] for s in symptomes_non_sca)

# === Bonus âge critique ===
if age_critique:
    score_total += 5

# Pas de score négatif
score_total = max(score_total, 0)

# === Calcul du pourcentage de risque ===
score_max_sca = sum(item["severity"] for item in sca_data["keywords"])
score_max_possible = score_max_sca + 5  # bonus âge inclus
pourcentage_risque = (score_total / score_max_possible) * 100
pourcentage_risque = min(round(pourcentage_risque, 2), 100.0)

# === Catégorisation du risque ===
if pourcentage_risque >= 75:
    interpretation = "🚨 Risque ÉLEVÉ de SCA ST+"
elif pourcentage_risque >= 40:
    interpretation = "⚠️ Risque MODÉRÉ de SCA ST+"
elif pourcentage_risque >= 10:
    interpretation = "🟡 Risque FAIBLE mais présent"
else:
    interpretation = "🟢 Risque très faible"

# === Affichage final ===
print("Âge détecté :", age if age else "Non détecté")
print("Genre détecté :", "Homme" if homme else "Femme" if femme else "Inconnu")
print("Score total de sévérité ajusté :", score_total)
print("Pourcentage estimé de risque de SCA ST+ :", f"{pourcentage_risque} %")
print("Interprétation :", interpretation)
