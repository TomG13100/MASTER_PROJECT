from transformers import pipeline

# Charger le pipeline NER (Reconnaissance d'Entités Nommées)
nlp_ner = pipeline("ner", model="Jean-Baptiste/camembert-ner", tokenizer="Jean-Baptiste/camembert-ner")

# Texte médical d'exemple
texte = "Le patient souffre d'une douleur thoracique intense et d’un essoufflement."

# Appliquer l'analyse NLP
resultats = nlp_ner(texte)

# Afficher les entités reconnues avec toutes les informations
for entite in resultats:
    print(f"Texte: {entite['word']}, Type: {entite.get('entity_group', entite.get('entity', 'N/A'))}, Score: {entite['score']:.2f}, Autres informations: {entite}")
