"""
app.py — Streamlit UI for Sign Language Recognition
----------------------------------------------------
Upload a sign language video → Predict sentence → Play audio
"""

import os
import tempfile
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

from src.data_preprocessing import extract_frames, SEQUENCE_LENGTH
from src.feature_extraction import extract_video_keypoints
from src.predict import text_to_speech

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .header-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .header-container h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-container p {
        color: #6b7280;
        font-size: 1.1rem;
    }

    .result-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .result-card .predicted-text {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .result-card .confidence-text {
        font-size: 1.1rem;
        color: #6b7280;
    }

    .upload-section {
        background: #f9fafb;
        border: 2px dashed #d1d5db;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .prob-bar {
        background: #e5e7eb;
        border-radius: 8px;
        height: 24px;
        margin: 4px 0;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.6s ease;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load model (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the trained model, label encoder, and normalization stats."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    model = load_model(os.path.join(models_dir, "sign_language_model.h5"))
    classes = np.load(os.path.join(models_dir, "label_encoder.npy"),
                      allow_pickle=True)
    norm = np.load(os.path.join(models_dir, "norm_stats.npz"))
    return model, classes, norm['mean'], norm['std']


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1>🤟 Sign Language Recognition</h1>
    <p>Upload a sign language video to get text translation and audio output</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Video Upload
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a sign language video",
    type=["mp4", "avi", "mov"],
    help="Upload a video of a person performing sign language"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Show the uploaded video
    st.video(uploaded_file)
    st.markdown("---")

    # Predict button
    if st.button("🔍 Recognize Sign Language", use_container_width=True):
        with st.spinner("⏳ Processing video... Extracting keypoints..."):
            try:
                # Load model
                model, classes, norm_mean, norm_std = load_artifacts()

                # Extract frames
                frames = extract_frames(tmp_path, SEQUENCE_LENGTH)
                if frames is None:
                    st.error("❌ Could not read the video. Please upload a valid file.")
                    st.stop()

                # Extract keypoints
                keypoints = extract_video_keypoints(frames)

                # Normalize
                keypoints = (keypoints - norm_mean) / (norm_std + 1e-8)

                # Predict
                X = keypoints[np.newaxis, ...]
                probs = model.predict(X, verbose=0)[0]
                predicted_idx = np.argmax(probs)
                predicted_label = classes[predicted_idx]
                confidence = float(probs[predicted_idx])

            except FileNotFoundError:
                st.error("❌ Model not found! Please run `python src/train.py` first.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

        # ── Show Results ──
        st.markdown(f"""
        <div class="result-card">
            <div class="predicted-text">"{predicted_label}"</div>
            <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


        # ── Text to Speech ──
        st.markdown("#### 🔊 Audio Output")
        with st.spinner("Generating audio..."):
            audio_path = os.path.join(tempfile.gettempdir(), "sign_output.mp3")
            text_to_speech(predicted_label, audio_path)

        st.audio(audio_path, format="audio/mp3")
        st.success(f"✅ Predicted: **{predicted_label}** | Confidence: **{confidence*100:.1f}%**")

    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except:
        pass

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
    "Sign Language Recognition System • Final Year Project • "
    "Built with MediaPipe, LSTM & Streamlit"
    "</p>",
    unsafe_allow_html=True
)
