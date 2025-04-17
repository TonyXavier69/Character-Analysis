import streamlit as st
import whisper
from transformers import pipeline
from moviepy.editor import VideoFileClip
import tempfile
import os



@st.cache_resource()
def load_models():
    stt_model = whisper.load_model("base", device="cpu")

    classifier = pipeline(
    "text-classification",
    model="./personality_classifier",  
    device=-1,
    top_k=2 
)
    return stt_model, classifier

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")
    video.audio.write_audiofile(audio_path)
    return audio_path

# Load models
try:
    stt_model, classifier = load_models()
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# App UI
st.title("Character Analysis AI")
st.markdown("⚠️ **Note**: This is a prototype. Human character cannot be fully judged by AI.")

# Input choice
input_type = st.radio("Choose input type:", ("Upload Video", "Enter Text"))

text = ""
if input_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video (max 5 mins)", type=["mp4", "mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.getvalue())
            video_path = tmp.name

        with st.spinner("Extracting audio..."):
            try:
                audio_path = extract_audio_from_video(video_path)
            except Exception as e:
                st.error(f"Audio extraction failed: {str(e)}")
                st.stop()

        with st.spinner("Transcribing audio..."):
            try:
                transcription = stt_model.transcribe(audio_path)["text"]
                text = st.text_area("Transcribed Text:", transcription, height=150)
            except Exception as e:
                st.error(f"Transcription failed: {str(e)}")
                st.stop()
else:
    text = st.text_area("Enter your text:", height=150, placeholder="Describe about yourself")

# Analyze
if st.button("Analyze") and text:
    with st.spinner("Analyzing..."):
        try:
            results = classifier(text)[0]
            scores = {res['label']: res['score'] for res in results}

            good_score = scores.get('LABEL_1', scores.get('POSITIVE', 0))
            bad_score = scores.get('LABEL_0', scores.get('NEGATIVE', 0))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Good Person Score", f"{good_score * 100:.1f}%")
            with col2:
                st.metric("Bad Person Score", f"{bad_score * 100:.1f}%")

            dominant_label = "good" if good_score > 0.5 else "bad"
            st.subheader(f"Prediction: **{dominant_label.upper()}** person")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("This AI model analyzes text for behavioral traits. It is not a substitute for human judgment.")
