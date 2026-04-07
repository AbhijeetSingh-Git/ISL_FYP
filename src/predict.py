"""
predict.py
----------
Load trained model and predict sign language from a video file.
Optionally convert the predicted text to speech.
"""

import os
import sys
import argparse
import numpy as np
from tensorflow.keras.models import load_model

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_preprocessing import extract_frames, SEQUENCE_LENGTH
from src.feature_extraction import extract_video_keypoints, FEATURE_DIM

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sign_language_model.h5")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.npy")
NORM_PATH = os.path.join(MODELS_DIR, "norm_stats.npz")


def load_artifacts():
    """Load trained model, label encoder, and normalization stats."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train.py first.")

    model = load_model(MODEL_PATH)
    classes = np.load(ENCODER_PATH, allow_pickle=True)
    norm = np.load(NORM_PATH)
    return model, classes, norm['mean'], norm['std']


def predict_video(video_path: str, model=None, classes=None,
                  norm_mean=None, norm_std=None):
    """
    Predict the sign language sentence from a video file.

    Parameters
    ----------
    video_path : str – path to the .MP4 file
    model, classes, norm_mean, norm_std – pre-loaded artifacts (optional)

    Returns
    -------
    predicted_label : str
    confidence : float (0-1)
    all_probs : dict mapping class → probability
    """
    # Load artifacts if not provided
    if model is None:
        model, classes, norm_mean, norm_std = load_artifacts()

    # Step 1: Extract frames
    frames = extract_frames(video_path, SEQUENCE_LENGTH)
    if frames is None:
        return None, 0.0, {}

    # Step 2: Extract keypoints
    keypoints = extract_video_keypoints(frames)

    # Step 3: Normalize using training stats
    keypoints = (keypoints - norm_mean) / (norm_std + 1e-8)

    # Step 4: Predict
    X = keypoints[np.newaxis, ...]   # shape: (1, 30, 258)
    probs = model.predict(X, verbose=0)[0]

    predicted_idx = np.argmax(probs)
    predicted_label = classes[predicted_idx]
    confidence = float(probs[predicted_idx])

    all_probs = {classes[i]: float(probs[i]) for i in range(len(classes))}

    return predicted_label, confidence, all_probs


def text_to_speech(text: str, output_path: str = "output_audio.mp3") -> str:
    """
    Convert text to speech using gTTS and save as .mp3.

    Returns
    -------
    output_path : str – path to the saved audio file
    """
    from gtts import gTTS

    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_path)
    print(f"[INFO] Audio saved → {output_path}")
    return output_path


# ──────────────────────────────────────────────
# CLI interface
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sign language from video")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file")
    parser.add_argument("--speak", action="store_true",
                        help="Also generate speech audio")
    args = parser.parse_args()

    label, conf, probs = predict_video(args.video)

    if label:
        print(f"\n{'='*50}")
        print(f"  Predicted: {label}")
        print(f"  Confidence: {conf*100:.1f}%")
        print(f"{'='*50}")
        print("\n  All probabilities:")
        for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(p * 30)
            print(f"    {cls:35s} {p*100:5.1f}% {bar}")

        if args.speak:
            audio_path = text_to_speech(label)
            print(f"\n  🔊 Audio file: {audio_path}")
    else:
        print("[ERROR] Could not process the video.")
