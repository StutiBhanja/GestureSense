"""
generate_synthetic_dataset.py
------------------------------
DEMO-ONLY HELPER (not part of the official 20-step pipeline)

This script generates a synthetic-but-realistic hand landmark dataset that
mimics MediaPipe's 21-point hand output for 8 gestures. It exists purely so
the rest of the pipeline (EDA -> preprocessing -> model training -> tuning
-> evaluation -> Streamlit app) can be demonstrated end-to-end with working
code and real results, WITHOUT needing live webcam access.

>>> Replace dataset/gesture_data.csv with your own data collected using
>>> collect_data.py on your local machine before your final submission. <<<

Landmark index reference (MediaPipe Hands):
 0: wrist
 1-4:  thumb  (CMC, MCP, IP, TIP)
 5-8:  index  (MCP, PIP, DIP, TIP)
 9-12: middle (MCP, PIP, DIP, TIP)
 13-16: ring   (MCP, PIP, DIP, TIP)
 17-20: pinky  (MCP, PIP, DIP, TIP)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

GESTURES = [
    "thumbs_up", "thumbs_down", "open_palm", "fist",
    "peace", "ok_sign", "pointing", "call_me",
]

SAMPLES_PER_CLASS = 300
NOISE_STD = 0.015


def base_hand():
    """A neutral, relaxed open-hand base pose (21 landmarks x,y,z), palm facing camera."""
    lm = np.zeros((21, 3))
    lm[0] = [0.50, 0.85, 0.0]                      # wrist

    # thumb
    lm[1] = [0.40, 0.78, -0.01]
    lm[2] = [0.33, 0.68, -0.02]
    lm[3] = [0.28, 0.58, -0.03]
    lm[4] = [0.24, 0.50, -0.04]

    # index
    lm[5] = [0.42, 0.62, 0.0]
    lm[6] = [0.41, 0.46, -0.01]
    lm[7] = [0.40, 0.34, -0.02]
    lm[8] = [0.40, 0.24, -0.03]

    # middle
    lm[9]  = [0.50, 0.60, 0.0]
    lm[10] = [0.50, 0.42, -0.01]
    lm[11] = [0.50, 0.28, -0.02]
    lm[12] = [0.50, 0.16, -0.03]

    # ring
    lm[13] = [0.58, 0.62, 0.0]
    lm[14] = [0.59, 0.46, -0.01]
    lm[15] = [0.60, 0.34, -0.02]
    lm[16] = [0.60, 0.24, -0.03]

    # pinky
    lm[17] = [0.66, 0.66, 0.0]
    lm[18] = [0.68, 0.54, -0.01]
    lm[19] = [0.69, 0.45, -0.02]
    lm[20] = [0.70, 0.37, -0.03]
    return lm


def curl_finger(lm, mcp_i, pip_i, dip_i, tip_i, curl_amount=0.7):
    """Bend a finger toward the palm by pulling PIP/DIP/TIP back toward MCP+wrist."""
    wrist = lm[0]
    mcp = lm[mcp_i]
    direction_to_wrist = wrist - mcp
    lm[pip_i] = mcp + 0.35 * direction_to_wrist * curl_amount + (lm[pip_i] - mcp) * (1 - curl_amount)
    lm[dip_i] = mcp + 0.55 * direction_to_wrist * curl_amount + (lm[dip_i] - mcp) * (1 - curl_amount)
    lm[tip_i] = mcp + 0.70 * direction_to_wrist * curl_amount + (lm[tip_i] - mcp) * (1 - curl_amount)
    return lm


FINGERS = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}


def make_gesture(name):
    lm = base_hand()

    if name == "open_palm":
        pass  # all fingers extended already

    elif name == "fist":
        for f in ["thumb", "index", "middle", "ring", "pinky"]:
            lm = curl_finger(lm, *FINGERS[f], curl_amount=0.85)

    elif name == "thumbs_up":
        for f in ["index", "middle", "ring", "pinky"]:
            lm = curl_finger(lm, *FINGERS[f], curl_amount=0.85)
        lm[4] = [0.50, 0.15, -0.02]   # thumb pointing straight up
        lm[3] = [0.48, 0.35, -0.02]
        lm[2] = [0.45, 0.55, -0.01]

    elif name == "thumbs_down":
        for f in ["index", "middle", "ring", "pinky"]:
            lm = curl_finger(lm, *FINGERS[f], curl_amount=0.85)
        lm[4] = [0.50, 0.98, -0.02]   # thumb pointing down
        lm[3] = [0.48, 0.90, -0.02]
        lm[2] = [0.45, 0.82, -0.01]

    elif name == "peace":
        for f in ["thumb", "ring", "pinky"]:
            lm = curl_finger(lm, *FINGERS[f], curl_amount=0.85)
        # index & middle stay extended, slightly spread apart
        lm[6][0] -= 0.03; lm[7][0] -= 0.045; lm[8][0] -= 0.06
        lm[10][0] += 0.03; lm[11][0] += 0.045; lm[12][0] += 0.06

    elif name == "ok_sign":
        for f in ["ring", "pinky"]:
            lm = curl_finger(lm, *FINGERS[f], curl_amount=0.75)
        # thumb & index tips touch (circle shape)
        lm[8] = [0.38, 0.55, -0.02]
        lm[7] = [0.39, 0.60, -0.02]
        lm[4] = [0.37, 0.56, -0.02]
        lm[3] = [0.35, 0.62, -0.02]

    elif name == "pointing":
        for f in ["thumb", "middle", "ring", "pinky"]:
            lm = curl_finger(lm, *FINGERS[f], curl_amount=0.85)
        # index stays fully extended (already in base pose)

    elif name == "call_me":
        for f in ["index", "middle", "ring"]:
            lm = curl_finger(lm, *FINGERS[f], curl_amount=0.85)
        lm[4] = [0.50, 0.15, -0.02]   # thumb out
        lm[3] = [0.48, 0.35, -0.02]
        lm[20] = [0.75, 0.20, -0.03]  # pinky out
        lm[19] = [0.72, 0.40, -0.02]

    return lm


def augment(lm, noise_std=NOISE_STD):
    """Add small random noise + slight rotation/scale/translation for realism/variety."""
    lm = lm.copy()

    # random small rotation in the image plane
    theta = np.random.uniform(-0.15, 0.15)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    center = lm[0][:2]
    xy = lm[:, :2] - center
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    lm[:, :2] = xy @ rot.T + center

    # random scale (hand closer/farther from camera)
    scale = np.random.uniform(0.9, 1.1)
    lm[:, :2] = center + (lm[:, :2] - center) * scale

    # random translation (hand position in frame)
    shift = np.random.uniform(-0.05, 0.05, size=2)
    lm[:, :2] += shift

    # gaussian noise (detection jitter)
    lm += np.random.normal(0, noise_std, lm.shape)

    return np.clip(lm, -0.2, 1.2)


def main():
    os.makedirs("dataset", exist_ok=True)
    rows = []
    for gesture in GESTURES:
        base = make_gesture(gesture)
        for _ in range(SAMPLES_PER_CLASS):
            sample = augment(base)
            rows.append(list(sample.flatten()) + [gesture])

    columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv("dataset/gesture_data.csv", index=False)
    print(f"Synthetic dataset saved: dataset/gesture_data.csv ({len(df)} rows)")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
