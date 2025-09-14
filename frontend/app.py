import streamlit as st
import requests
import tempfile
import os
from streamlit_mic_recorder import mic_recorder

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(
    page_title="NeuroFit Voice Dashboard",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 NeuroFit Voice Dashboard")
st.markdown("Real-time voice transcription, mood detection, and music suggestions.")

# ------------------------------
# Record Audio
# ------------------------------
audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="🛑 Stop & Analyze",
    just_once=True,
    use_container_width=True,
)

if audio and "bytes" in audio:
    st.info("Audio recorded. Sending to backend for processing...")

    # Save audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio["bytes"])
        tmp_path = tmp.name

    try:
        # Send file to Flask backend
        with open(tmp_path, "rb") as f:
            response = requests.post(
                "http://localhost:5000/analyze",
                files={"file": f}
            )

        if response.status_code == 200:
            result = response.json()

            # Display transcription + results
            st.subheader("📝 Transcription")
            st.write(result["transcription"])

            st.subheader("📊 Sentiment Analysis")
            st.write(f"**Sentiment:** {result['sentiment']} (Confidence: {result['confidence']})")

            st.subheader("🎶 Suggested Mood & Playlist")
            st.write(f"Mood: {result['mood']}")
            st.markdown(f"[Open Playlist 🎵]({result['spotify_playlist']})")

        else:
            st.error(f"Backend error: {response.text}")

    except Exception as e:
        st.error(f"⚠️ Could not reach backend: {e}")

    finally:
        os.remove(tmp_path)  # cleanup
