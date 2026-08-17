"""
collect_data.py
----------------
STEP 2 & 3 : Dataset Collection (Custom Dataset)

Run this file on YOUR OWN machine (it needs a real webcam, which is why it
can't be run inside this chat). It uses MediaPipe Hands to detect your hand
in real time, extracts 21 hand landmarks (x, y, z for each = 63 features),
and saves them to a CSV file, one row per captured frame.

HOW TO USE
----------
1. Run:  python collect_data.py
2. A webcam window opens.
3. Hold up ONE of the 8 gestures listed below.
4. Press the number key shown next to that gesture to start capturing
   frames for it (it will auto-capture SAMPLES_PER_CLASS frames).
5. Repeat for all 8 gestures, changing hand position/angle/distance
   slightly between captures for a more robust dataset.
6. Press 'q' to quit and save everything to dataset/gesture_data.csv

TIP: Capture in a few different lighting conditions and hand orientations
for a more generalizable model.
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GESTURES = {
    "1": "thumbs_up",
    "2": "thumbs_down",
    "3": "open_palm",
    "4": "fist",
    "5": "peace",
    "6": "ok_sign",
    "7": "pointing",
    "8": "call_me",
}

SAMPLES_PER_CLASS = 200          # frames captured per key press
OUTPUT_CSV = "dataset/gesture_data.csv"

# ---------------------------------------------------------------------------
# MEDIAPIPE SETUP
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def extract_landmarks(hand_landmarks):
    """Flatten 21 (x, y, z) landmarks into a single 63-length feature vector."""
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return row


def main():
    os.makedirs("dataset", exist_ok=True)
    cap = cv2.VideoCapture(0)
    data_rows = []

    capturing_label = None
    capture_count = 0

    print("Press the number key for a gesture to start capturing:")
    for k, v in GESTURES.items():
        print(f"  [{k}] {v}")
    print("Press 'q' to quit and save.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if capturing_label is not None and capture_count < SAMPLES_PER_CLASS:
                    features = extract_landmarks(hand_landmarks)
                    features.append(capturing_label)
                    data_rows.append(features)
                    capture_count += 1

        # UI overlay
        status = f"Capturing: {capturing_label} ({capture_count}/{SAMPLES_PER_CLASS})" \
            if capturing_label else "Press a number key to start capturing"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key) if key != 255 else ""

        if key_char in GESTURES:
            capturing_label = GESTURES[key_char]
            capture_count = 0
        elif key_char == "q":
            break

        if capturing_label is not None and capture_count >= SAMPLES_PER_CLASS:
            print(f"Finished capturing '{capturing_label}' ({SAMPLES_PER_CLASS} samples)")
            capturing_label = None
            capture_count = 0

    cap.release()
    cv2.destroyAllWindows()

    # Save to CSV
    columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
    df = pd.DataFrame(data_rows, columns=columns)

    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} total rows to {OUTPUT_CSV}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
