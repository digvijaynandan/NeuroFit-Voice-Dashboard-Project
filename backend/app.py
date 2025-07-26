# backend/app.py
import os
import sys
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Fix import path to reach ml_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_model.vosk_mood_pipeline import transcribe_audio, classify_mood

app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    filename = secure_filename(audio_file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        audio_path = temp_file.name
        audio_file.save(audio_path)

    try:
        transcription = transcribe_audio(audio_path)
        mood = classify_mood(transcription)

        # Mood → Spotify playlist mapping
        playlists = {
            "Happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
            "Sad": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1",
            "Angry": "https://open.spotify.com/playlist/37i9dQZF1DX59NCqCqJtoH",
            "Neutral": "https://open.spotify.com/playlist/37i9dQZF1DX4E3UdUs7fUx"
        }

        return jsonify({
            "transcription": transcription,
            "mood": mood,
            "spotify_url": playlists.get(mood, "")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
