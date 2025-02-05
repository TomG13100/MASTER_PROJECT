import whisper
import os
import torch
from pyannote.audio import Pipeline
from whisperpyannotemain.utils import words_per_segment

def transcribe_audio(audio_path):
    """
    Transcribes and diarizes an audio file, returning structured text per speaker.
    """
    print(f"Processing audio: {audio_path}")

    # Vérification du périphérique (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Chargement des modèles
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="HF_AUTH_TOKEN").to(device)
        model = whisper.load_model("medium").to(device)

        # Diarisation et transcription
        diarization_result = pipeline(audio_path)
        transcription_result = model.transcribe(audio_path, word_timestamps=True)
        final_result = words_per_segment(transcription_result, diarization_result)

        # Organiser les transcriptions par speaker avec numérotation
        speaker_mapping = {}  # Associer ID interne à Speaker 1, Speaker 2, etc.
        speaker_index = 1
        transcriptions = {}

        for _, segment in final_result.items():
            speaker_id = segment["speaker"]

            if speaker_id not in speaker_mapping:
                speaker_mapping[speaker_id] = f"Speaker {speaker_index}"
                speaker_index += 1

            speaker_label = speaker_mapping[speaker_id]

            if speaker_label not in transcriptions:
                transcriptions[speaker_label] = []

            transcriptions[speaker_label].append(segment["text"])

        # Générer un format clair pour l'affichage
        formatted_transcriptions = "\n\n".join(
            f"{speaker}:\n" + " ".join(texts) for speaker, texts in transcriptions.items()
        )

        return formatted_transcriptions

    except Exception as e:
        print(f"Erreur pendant la transcription : {str(e)}")
        return None

if __name__ == "__main__":
    test_audio = "data/raw/Audio-SCA-1.wav"
    transcriptions = transcribe_audio(test_audio)

    if transcriptions:
        print("\n**Transcription terminée !**\n")
        print(transcriptions)
