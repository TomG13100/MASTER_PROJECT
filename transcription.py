import whisper

model = whisper.load_model("turbo")
result = model.transcribe("data/audio/test.mp3")
print(result["text"])
