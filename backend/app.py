from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load models once
whisper_model = whisper.load_model("medium")  # use 'medium' for better accuracy
sentiment_model = pipeline("sentiment-analysis")

@app.route("/")
def home():
    return "✅ NeuroFit API is running."

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file found"}), 400

    audio_file = request.files["audio"]
    audio_path = "temp.wav"
    audio_file.save(audio_path)

    try:
        result = whisper_model.transcribe(audio_path, fp16=False, language="en")
        text = result.get("text", "").strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return jsonify({"transcription": text})

@app.route("/sentiment", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = sentiment_model(text)[0]
        sentiment = result["label"]
        return jsonify({"sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
