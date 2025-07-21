import os
import whisper
import torch
from transformers import pipeline

# 👇 Change this to your actual audio file name
AUDIO_FILE = os.path.join(os.path.dirname(__file__), "test_audio.mp3")



print("📁 Current Directory:", os.getcwd())
print("📄 Files in current folder:", os.listdir())



def transcribe_audio(audio_path):
    print("🔁 Loading Whisper model...")
    model = whisper.load_model("base")
    print("✅ Whisper model loaded.")

    if not os.path.exists(audio_path):
        print(f"❌ Audio file '{audio_path}' not found in folder.")
        return None

    try:
        result = model.transcribe(audio_path)
        text = result["text"]
        print(f"📝 Transcription: {text}")
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None

def classify_mood(text):
    print("🧠 Loading sentiment analysis model (PyTorch)...")
    try:
        classifier = pipeline("sentiment-analysis", framework="pt")  # use PyTorch
        result = classifier(text)[0]
        mood = result["label"]
        score = result["score"]
        print(f"💡 Mood: {mood} (Confidence: {score:.2f})")
        return mood
    except Exception as e:
        print(f"❌ Mood classification error: {e}")
        return None

if __name__ == "__main__":
    print("🚀 NeuroFit: Voice-to-Mood Pipeline Starting...")

    text = transcribe_audio(AUDIO_FILE)
    if text:
        classify_mood(text)
