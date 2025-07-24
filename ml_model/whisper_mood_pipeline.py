import os
import whisper
import torch
from transformers import pipeline

# 👇 Change this to your actual audio file name
AUDIO_FILE = os.path.join(os.path.dirname(__file__), "test_audio.mp3")

# Descriptive mood to Spotify playlist mapping
MOOD_PLAYLIST_MAP = {
    "happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",   # upbeat/happy
    "sad": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1",     # emotional/sad
    "angry": "https://open.spotify.com/playlist/37i9dQZF1DX1tyCD9QhIWF",   # rage/metal/hype
    "neutral": "https://open.spotify.com/playlist/37i9dQZF1DWYBO1MoTDhZI", # lo-fi/study
    "calm": "https://open.spotify.com/playlist/37i9dQZF1DWZqd5JICZI0u"     # meditation/calm
}

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

def map_sentiment_to_mood(sentiment_label, text=""):
    """ Maps general sentiment to more descriptive mood """
    sentiment_label = sentiment_label.lower()

    if sentiment_label == "positive":
        # Optional: check for calm keywords in text
        if any(word in text.lower() for word in ["relaxed", "peaceful", "calm", "quiet", "okay"]):
            return "calm"
        return "happy"
    elif sentiment_label == "negative":
        # Optional: check if user sounds angry vs sad
        if any(word in text.lower() for word in ["angry", "annoyed", "frustrated", "mad"]):
            return "angry"
        return "sad"
    else:
        return "neutral"

def classify_mood(text):
    print("🧠 Loading sentiment analysis model (PyTorch)...")
    try:
        classifier = pipeline("sentiment-analysis", framework="pt")
        result = classifier(text)[0]
        raw_sentiment = result["label"]
        score = result["score"]

        mood = map_sentiment_to_mood(raw_sentiment, text)
        playlist_url = MOOD_PLAYLIST_MAP.get(mood, MOOD_PLAYLIST_MAP["neutral"])

        print(f"💡 Mood Detected: {mood} (Confidence: {score:.2f})")
        return mood, playlist_url
    except Exception as e:
        print(f"❌ Mood classification error: {e}")
        return None, None

if __name__ == "__main__":
    print("🚀 NeuroFit: Voice-to-Mood + Music Suggestion Starting...")

    text = transcribe_audio(AUDIO_FILE)
    if text:
        mood, playlist = classify_mood(text)
        if playlist:
            print(f"🎵 Recommended Spotify Playlist for '{mood}' mood:")
            print(f"🔗 {playlist}")

