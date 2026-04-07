"""
model.py
--------
LSTM-based model for sign language sequence classification.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input
)


def build_model(
    sequence_length: int = 30,
    feature_dim: int = 258,
    num_classes: int = 6,
    lstm_units: tuple = (128, 64),
    dropout_rate: float = 0.4
) -> Sequential:
    """
    Build a stacked-LSTM classifier.

    Architecture
    ------------
    Input → LSTM(128) → BatchNorm → Dropout →
            LSTM(64)  → BatchNorm → Dropout →
            Dense(64, relu) → Dropout →
            Dense(num_classes, softmax)

    Parameters
    ----------
    sequence_length : int   – timesteps (frames)
    feature_dim : int       – features per frame (258 for pose+hands)
    num_classes : int       – number of output classes
    lstm_units : tuple      – units for each LSTM layer
    dropout_rate : float    – dropout probability

    Returns
    -------
    Compiled Keras Sequential model
    """
    model = Sequential([
        # Input layer
        Input(shape=(sequence_length, feature_dim)),

        # First LSTM layer – return sequences for stacking
        LSTM(lstm_units[0], return_sequences=True),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Second LSTM layer
        LSTM(lstm_units[1], return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Dense classifier head
        Dense(64, activation='relu'),
        Dropout(dropout_rate * 0.75),   # slightly less dropout here

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────
if __name__ == "__main__":
    m = build_model()
    print(f"\nTotal params: {m.count_params():,}")
