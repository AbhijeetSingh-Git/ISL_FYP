"""
train.py
--------
End-to-end training pipeline:
  1. Load videos from dataset
  2. Extract MediaPipe keypoints
  3. Augment data
  4. Train LSTM model
  5. Save model + label encoder + normalization stats
  6. Plot training history
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_preprocessing import load_dataset, SEQUENCE_LENGTH
from src.feature_extraction import process_dataset, FEATURE_DIM
from src.model import build_model


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sign_language_model.h5")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.npy")
NORM_PATH = os.path.join(MODELS_DIR, "norm_stats.npz")
HISTORY_PLOT_PATH = os.path.join(MODELS_DIR, "training_history.png")


def plot_history(history, save_path: str):
    """Save accuracy & loss curves to disk."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training history plot saved → {save_path}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Step 1: Load videos ──
    print("=" * 60)
    print("STEP 1 / 5 — Loading videos...")
    print("=" * 60)
    videos, labels, class_names = load_dataset()

    # ── Step 2: Extract keypoints + augment ──
    print("\n" + "=" * 60)
    print("STEP 2 / 5 — Extracting keypoints & augmenting data...")
    print("=" * 60)
    NUM_AUGMENTS = 8      # 8 augments per video → 29 × 9 = 261 total samples
    X, y_labels, norm_mean, norm_std = process_dataset(
        videos, labels, num_augments=NUM_AUGMENTS
    )

    # Save normalization statistics (needed at inference time)
    np.savez(NORM_PATH, mean=norm_mean, std=norm_std)
    print(f"[INFO] Normalization stats saved → {NORM_PATH}")

    # ── Step 3: Encode labels ──
    print("\n" + "=" * 60)
    print("STEP 3 / 5 — Encoding labels...")
    print("=" * 60)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    y_cat = to_categorical(y_encoded)
    num_classes = len(le.classes_)
    print(f"  Classes ({num_classes}): {list(le.classes_)}")

    # Save label encoder
    np.save(ENCODER_PATH, le.classes_)
    print(f"[INFO] Label encoder saved → {ENCODER_PATH}")

    # ── Step 4: Train / Val split ──
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")

    # ── Step 5: Build & train model ──
    print("\n" + "=" * 60)
    print("STEP 4 / 5 — Building & training LSTM model...")
    print("=" * 60)
    model = build_model(
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=FEATURE_DIM,
        num_classes=num_classes,
        lstm_units=(128, 64),
        dropout_rate=0.4
    )

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # ── Step 6: Results ──
    print("\n" + "=" * 60)
    print("STEP 5 / 5 — Results")
    print("=" * 60)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Final Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  Final Validation Loss:     {val_loss:.4f}")
    print(f"  Model saved → {MODEL_PATH}")

    # Plot training history
    plot_history(history, HISTORY_PLOT_PATH)

    print("\n✅ Training complete!")
    return val_acc


if __name__ == "__main__":
    accuracy = main()
