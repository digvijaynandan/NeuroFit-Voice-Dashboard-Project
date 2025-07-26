import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from textblob import TextBlob


# Optional post-processing (uncomment if you install textblob)
# from textblob import TextBlob

# ======= Settings =======
AUDIO_MP3 = os.path.join(os.path.dirname(__file__), "test_audio2.mp3")
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
WAV_FILE = "converted_audio.wav"

# ======= MP3 to WAV Conversion =======
def convert_mp3_to_wav(mp3_path, wav_path=WAV_FILE):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # mono, 16kHz
    audio.export(wav_path, format="wav")
    return wav_path

# ======= Vosk Transcription =======
def transcribe_audio(wav_path):
    if not os.path.exists(VOSK_MODEL_PATH):
        print("❌ Vosk model not found at", VOSK_MODEL_PATH)
        return None

    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)

    results = []
    with wave.open(wav_path, "rb") as wf:
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                partial_result = json.loads(rec.Result())
                results.append(partial_result.get("text", ""))
        final_result = json.loads(rec.FinalResult())
        results.append(final_result.get("text", ""))

    full_text = " ".join(results).strip()

    # Optional: Correct common small typos using TextBlob
    corrected = str(TextBlob(full_text).correct())

    return corrected  

# ======= Mood Classification =======
def predict_mood(text):
    text = text.lower()

    mood_keywords = {
        "happy": ["happy", "joy", "excited", "good", "cheerful", "delighted", "glad"],
        "sad": ["sad", "down", "depressed", "unhappy", "cry", "upset", "blue"],
        "angry": ["angry", "mad", "furious", "irritated", "annoyed"],
        "calm": ["calm", "peaceful", "relaxed", "okay", "normal"],
    }

    for mood, keywords in mood_keywords.items():
        for word in keywords:
            if word in text:
                return mood

    return "neutral"

# ======= Main =======
if __name__ == "__main__":
    print("🚀 NeuroFit Voice-to-Mood Pipeline (Vosk Version) Starting...\n")

    print("🎧 Converting MP3 to WAV...")
    wav_path = convert_mp3_to_wav(AUDIO_MP3)

    print("📝 Transcribing...")
    text = transcribe_audio(wav_path)
    print("🗣️ Transcript:", text)

    if text:
        mood = predict_mood(text)
        print(f"💡 Detected Mood: {mood}")
    else:
        print("⚠️ Could not detect speech.")
