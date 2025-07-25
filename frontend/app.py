import streamlit as st
import requests

st.set_page_config(page_title="NeuroFit Voice Dashboard", page_icon="🎙️")
st.title("🎙️ NeuroFit Voice Dashboard")
st.write("Upload your voice and get the emotion behind it!")

# Upload .wav file
audio_file = st.file_uploader("Upload your voice file (WAV format)", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    if st.button("Analyze"):
        # Send to backend Flask server
        files = {"file": audio_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/predict", files={"file": audio_file})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"🎯 Sentiment: {result['sentiment']}")
            st.info(f"📝 Transcribed Text: {result['text']}")
        else:
            st.error("❌ Something went wrong. Please try again.")
