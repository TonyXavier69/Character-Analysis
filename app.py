import streamlit as st
from transformers import pipeline


@st.cache_resource()
def load_models():

    classifier = pipeline(
        "text-classification",
        model="./personality_classifier",
        device=-1,
        top_k=2
    )
    return  classifier


# Load models
try:
    classifier = load_models()
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# App UI
st.title("Character Analysis AI")
st.markdown("⚠️ **Note**: This is a prototype. Human character cannot be fully judged by AI.")

# Text Input
text = st.text_area("Enter your text:", height=150, placeholder="Describe about yourself")

# Analyze
if st.button("Analyze") and text:
    with st.spinner("Analyzing..."):
        try:
            results = classifier(text)[0]
            scores = {res['label']: res['score'] for res in results}

            good_score = scores.get('LABEL_1')
            bad_score = scores.get('LABEL_0')

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
