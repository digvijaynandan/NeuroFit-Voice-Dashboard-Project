print("Hello from NeuroFit!")
import whisper

print("🔁 Loading Whisper model...")
model = whisper.load_model("base")
print("✅ Whisper model loaded.")

AUDIO_FILE = "test_audio.mp3"  # Replace this with your file name if different

try:
    print("🎧 Transcribing...")
    result = model.transcribe(AUDIO_FILE)
    print("📝 Transcription result:")
    print(result["text"])
except Exception as e:
    print("❌ Error:", e)
