from transformers import CamembertTokenizer, CamembertModel

# Charger le tokenizer et le modèle CamemBERT
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base")

# Exemple de texte
texte = "Le patient souffre d'une douleur thoracique et d'un essoufflement."

# Tokeniser le texte
tokens = tokenizer(texte, return_tensors="pt", padding=True, truncation=True)

# Passer le texte à CamemBERT
output = model(**tokens)

print("Tokenisation réussie, modèle chargé !")
