import os
import whisper

def transcribe_audio(audio_dir, output_dir):
    """
    Transcrit tous les fichiers audio du répertoire donné à l'aide de Whisper
    et enregistre les transcriptions dans un fichier texte individuel.

    :param audio_dir: Répertoire contenant les fichiers audio (.wav, .mp3, etc.)
    :param output_dir: Répertoire où les transcriptions seront enregistrées
    """
    # Charger le modèle Whisper
    model = whisper.load_model("base")

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir tous les fichiers dans le répertoire audio
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.wav', '.mp3', '.m4a')):
            audio_path = os.path.join(audio_dir, filename)

            print(f"Transcription en cours pour {filename}...")
            # Transcrire l'audio
            result = model.transcribe(audio_path)

            # Enregistrer la transcription dans un fichier texte
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])

            print(f"Transcription terminée et sauvegardée : {output_file}")

    print("Toutes les transcriptions ont été générées.")

if __name__ == "__main__":
    # Définir les chemins des répertoires
    AUDIO_DIR = "data/transcriptions audio"  # Répertoire des fichiers audio
    OUTPUT_DIR = "data/transcriptions audio"  # Répertoire pour les transcriptions

    # Exécuter la transcription
    transcribe_audio(AUDIO_DIR, OUTPUT_DIR)
