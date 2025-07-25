import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import tempfile
import requests
import wave
from pydub import AudioSegment
from pydub.utils import which

# Set ffmpeg path for pydub
AudioSegment.converter = which("ffmpeg")

st.set_page_config(page_title="NeuroFit Voice Dashboard", page_icon="🎙️")
st.title("🎙️ NeuroFit Voice Dashboard")
st.write("Speak into your mic, transcribe your voice, and get sentiment analysis in real-time!")

# Audio processor to store raw audio frames
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

    def get_audio_data(self):
        return np.concatenate(self.frames) if self.frames else None

# ✅ Use proper WebRtcMode enum
ctx = webrtc_streamer(
    key="realtime-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"video": False, "audio": True},
)

# When the button is clicked
if ctx.audio_processor and st.button("🔊 Transcribe Now"):
    raw_audio = ctx.audio_processor.get_audio_data()
    
    if raw_audio is not None:
        # Save audio to a temporary .wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            with wave.open(temp_wav.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(raw_audio.tobytes())

        # Send audio to backend
        with open(temp_wav.name, "rb") as f:
            files = {"audio": f}
            response = requests.post("http://127.0.0.1:5000/transcribe", files=files)

        if response.status_code == 200:
            transcription = response.json().get("transcription")
            st.success("✅ Transcription:")
            st.write(transcription)

            # Sentiment analysis
            sentiment_response = requests.post(
                "http://127.0.0.1:5000/sentiment", json={"text": transcription}
            )
            if sentiment_response.status_code == 200:
                sentiment = sentiment_response.json().get("sentiment")
                st.info(f"🧠 Sentiment: {sentiment}")
            else:
                st.warning("⚠️ Sentiment analysis failed.")
        else:
            st.error("❌ Transcription failed.")
