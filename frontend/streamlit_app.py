# frontend/streamlit_app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import tempfile
import requests
from pydub import AudioSegment
from pydub.utils import which

# Setup ffmpeg for pydub
AudioSegment.converter = which("ffmpeg")

st.set_page_config(page_title="NeuroFit Voice Dashboard", page_icon="🎙️")
st.title("🎙️ NeuroFit: Real-Time Voice Mood Detection")
st.markdown("Speak into the mic and get your mood with a personalized Spotify playlist!")

# Audio Processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

    def get_audio(self):
        return np.concatenate(self.frames) if self.frames else None

# Start WebRTC Stream
ctx = webrtc_streamer(
    key="neurofit-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"video": False, "audio": True},
)

if ctx.audio_processor and st.button("🎤 Analyze Mood"):
    audio_data = ctx.audio_processor.get_audio()

    if audio_data is not None:
        # Convert to 16-bit PCM AudioSegment
        raw_audio_bytes = audio_data.astype(np.int16).tobytes()
        audio_segment = AudioSegment(
            data=raw_audio_bytes,
            sample_width=2,
            frame_rate=48000,
            channels=1
        )

        # Downsample for Vosk
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

        # Export as WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")
            temp_wav_path = temp_wav.name

        # Send to backend
        with open(temp_wav_path, "rb") as audio_file:
            files = {"audio": audio_file}
            response = requests.post("http://127.0.0.1:5000/process", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"📝 Transcription: {result['transcription']}")
            st.info(f"🧠 Mood: {result['mood']}")
            if result.get("spotify_url"):
                st.markdown(f"▶️ [Open Spotify Playlist]({result['spotify_url']})", unsafe_allow_html=True)
        else:
            st.error(f"⚠️ Failed to process audio: {response.json().get('error')}")
