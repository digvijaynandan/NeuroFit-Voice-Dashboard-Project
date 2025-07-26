# ml_model/vosk_mood_pipeline.py
import os
import wave
import json
from vosk import Model, KaldiRecognizer
from transformers import pipeline

# Load Vosk model
model_path = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
vosk_model = Model(model_path)

# Load Hugging Face sentiment classifier
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

def transcribe_audio(audio_path):
    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        raise ValueError("Audio must be WAV mono PCM 16bit at 16kHz.")

    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))

    transcript = " ".join([res.get("text", "") for res in results])
    return transcript.strip()
def classify_mood(text):
    emotions = classifier(text)[0]
    top_emotion = max(emotions, key=lambda x: x['score'])['label'].lower()

    mood_map = {
        "joy": "Happy",
        "happiness": "Happy",
        "sadness": "Sad",
        "anger": "Angry",
        "fear": "Sad",
        "neutral": "Neutral",
        "love": "Happy",
        "surprise": "Neutral"
    }
    return mood_map.get(top_emotion, "Neutral")
