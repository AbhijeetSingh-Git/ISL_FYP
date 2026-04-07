"""
feature_extraction.py
---------------------
Extract hand + pose keypoints from video frames using MediaPipe Holistic.
Includes data augmentation to expand the small dataset.
"""

import numpy as np
import mediapipe as mp

# ──────────────────────────────────────────────
# MediaPipe setup
# ──────────────────────────────────────────────
mp_holistic = mp.solutions.holistic

# Keypoint dimensions:
#   Pose:       33 landmarks × 4 values (x, y, z, visibility) = 132
#   Left hand:  21 landmarks × 3 values (x, y, z) = 63
#   Right hand: 21 landmarks × 3 values (x, y, z) = 63
#   Total per frame = 258
NUM_POSE = 33 * 4      # 132
NUM_HAND = 21 * 3      # 63    (each hand)
FEATURE_DIM = NUM_POSE + 2 * NUM_HAND   # 258


def extract_keypoints(frame: np.ndarray, holistic) -> np.ndarray:
    """
    Extract pose + hand keypoints from a single RGB frame.

    Returns
    -------
    np.ndarray of shape (258,)
    """
    results = holistic.process(frame)

    # Pose landmarks (33 × 4)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(NUM_POSE)

    # Left hand landmarks (21 × 3)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(NUM_HAND)

    # Right hand landmarks (21 × 3)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(NUM_HAND)

    return np.concatenate([pose, lh, rh])


def extract_video_keypoints(frames: np.ndarray) -> np.ndarray:
    """
    Process all frames of a single video and return the keypoint sequence.

    Parameters
    ----------
    frames : np.ndarray of shape (T, H, W, 3) – RGB frames

    Returns
    -------
    np.ndarray of shape (T, 258) – keypoints per frame
    """
    keypoints_seq = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for frame in frames:
            kp = extract_keypoints(frame, holistic)
            keypoints_seq.append(kp)
    return np.array(keypoints_seq)


# ──────────────────────────────────────────────
# Data Augmentation (critical for small datasets)
# ──────────────────────────────────────────────
def augment_keypoints(sequence: np.ndarray, num_augments: int = 5) -> list:
    """
    Generate augmented versions of a keypoint sequence.

    Augmentation strategies:
    1. Gaussian noise injection
    2. Time-shift (roll the sequence)
    3. Scale jitter
    4. Mirror left/right hands (swap hand columns)
    5. Random frame dropout + interpolation

    Parameters
    ----------
    sequence : np.ndarray of shape (T, 258)
    num_augments : int – how many augmented copies to create

    Returns
    -------
    list of np.ndarray, each of shape (T, 258)
    """
    augmented = []

    for i in range(num_augments):
        aug = sequence.copy()

        # 1) Gaussian noise (always applied, varying intensity)
        noise_scale = np.random.uniform(0.002, 0.01)
        aug = aug + np.random.normal(0, noise_scale, aug.shape)

        # 2) Time shift – roll sequence by a few frames (50% chance)
        if np.random.random() < 0.5:
            shift = np.random.randint(-3, 4)
            aug = np.roll(aug, shift, axis=0)

        # 3) Scale jitter – slightly scale keypoints (50% chance)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            aug = aug * scale

        # 4) Mirror hands – swap left and right hand columns (30% chance)
        if np.random.random() < 0.3:
            pose_end = NUM_POSE
            lh_end = NUM_POSE + NUM_HAND
            rh_end = NUM_POSE + 2 * NUM_HAND
            mirrored = aug.copy()
            mirrored[:, pose_end:lh_end] = aug[:, lh_end:rh_end]
            mirrored[:, lh_end:rh_end] = aug[:, pose_end:lh_end]
            # Flip x-coordinates for mirrored hands (every 3rd element starting at 0)
            for start in range(pose_end, rh_end, 3):
                mirrored[:, start] = 1.0 - mirrored[:, start]
            aug = mirrored

        # 5) Random frame dropout – zero out 1-2 frames (30% chance)
        if np.random.random() < 0.3:
            n_drop = np.random.randint(1, 3)
            drop_indices = np.random.choice(len(aug), n_drop, replace=False)
            aug[drop_indices] = 0.0

        augmented.append(aug.astype(np.float32))

    return augmented


def process_dataset(videos: list, labels: list, num_augments: int = 5):
    """
    Extract keypoints from all videos and augment.

    Returns
    -------
    X : np.ndarray of shape (N, T, 258)
    y : list[str] of length N
    """
    X_all = []
    y_all = []

    total = len(videos)
    for i, (video_frames, label) in enumerate(zip(videos, labels)):
        print(f"  Processing video {i+1}/{total} — class: {label}")

        # Extract keypoints
        kp_seq = extract_video_keypoints(video_frames)

        # Original sample
        X_all.append(kp_seq.astype(np.float32))
        y_all.append(label)

        # Augmented samples
        aug_seqs = augment_keypoints(kp_seq, num_augments=num_augments)
        for aug in aug_seqs:
            X_all.append(aug)
            y_all.append(label)

    X = np.array(X_all)

    # Normalize: per-feature zero-mean unit-variance
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std

    print(f"[INFO] Dataset shape after augmentation: {X.shape}")
    return X, y_all, mean.squeeze(), std.squeeze()
