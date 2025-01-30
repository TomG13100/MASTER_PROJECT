import os
import whisper
from pyannote.audio import Pipeline

def transcribe_and_separate(audio_path, output_dir, hf_token):
    """
    Transcrit un fichier audio et sépare les transcriptions par locuteur.

    Args:
        audio_path (str): Chemin du fichier audio.
        output_dir (str): Répertoire où enregistrer les fichiers de transcription.
        hf_token (str): Jeton Hugging Face pour Pyannote.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Transcrire avec Whisper
    print("Transcription avec Whisper...")
    model = whisper.load_model("small")
    transcription = model.transcribe(audio_path)

    # Diarisation avec Pyannote
    print("Diarisation avec Pyannote...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    diarization = pipeline(audio_path)

    # Associer les segments à des locuteurs
    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        text = " ".join(
            word["text"] for word in transcription["segments"]
            if turn.start <= word["start"] <= turn.end
        )
        speakers.setdefault(speaker, []).append(text)

    # Sauvegarder chaque locuteur dans un fichier séparé
    for speaker, texts in speakers.items():
        with open(os.path.join(output_dir, f"{speaker}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(texts))

    print(f"Fichiers de transcription sauvegardés dans {output_dir}")
