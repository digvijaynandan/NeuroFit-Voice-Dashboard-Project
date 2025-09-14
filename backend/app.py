import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment  # ✅ convert formats
import tempfile

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_model.whisper_mood_pipeline import process_audio_file

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save uploaded file temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    file.save(temp_input.name)

    # ✅ Convert to WAV (16kHz mono) for Whisper
    audio = AudioSegment.from_file(temp_input.name)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio.export(wav_path, format="wav")

    # Run through Whisper + sentiment
    result = process_audio_file(wav_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
