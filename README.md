# MASTER_PROJECT

commande à connaitre pour utiliser github :

git add .
git commit -m "Initial setup complete"
git pull origin master
git push origin master  # Assurez-vous d'être sur la bonne branche

# Installer tous les outils nécessaires :

pip install -r requirements.txt


sca15_project/
├── data/
│   ├── raw/                   # Données brutes audio
│   ├── transcriptions/        # Transcriptions audio (fichiers texte)
│   ├── keywords_labels.csv    # Fichier CSV associant mots-clés et classes
├── models/
│   ├── mlp_model.py           # Modèle MLP (Multi-Layer Perceptron) pour les mots-clés
│   ├── train_model.py         # Entraînement du modèle avec les mots-clés
│   ├── inference.py           # Prédictions sur de nouvelles données
├── scripts/
│   ├── preprocess_keywords.py # Prétraitement et vectorisation des mots-clés
│   ├── transcribe_audio.py    # Script pour transcrire l'audio en texte (via Whisper)
├── requirements.txt           # Dépendances Python nécessaires
├── main.py                    # Pipeline principal
├── README.md                  # Documentation du projet
