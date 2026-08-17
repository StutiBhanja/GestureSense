"""
preprocessing.py
-----------------
STEP 5 : Data Preprocessing
STEP 6 : Feature Selection / Feature Extraction
STEP 7 : Input-Output Separation
STEP 8 : Train-Test Split
STEP 8a: Feature Scaling

This module exposes a single function `load_and_preprocess()` that
train_model.py and predictor.py both import, so preprocessing logic lives
in exactly ONE place (no duplicated code between training and inference).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "dataset/gesture_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_and_preprocess(data_path: str = DATA_PATH, test_size: float = TEST_SIZE):
    """
    Loads the landmark dataset and returns everything needed for training:
    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_columns

    STEP 5 (Data Preprocessing for image-derived data):
        The 'images' here have already been converted to 21 MediaPipe hand
        landmarks (x, y, z) by collect_data.py, so there is no raw pixel
        resizing/normalizing to do here — MediaPipe already normalizes
        coordinates to a 0-1 range relative to the image frame. We still
        apply StandardScaler in Step 8a for stable/fast ANN convergence.

    STEP 6 (Feature Extraction):
        Feature extraction (Flatten of 21 landmarks x 3 coords = 63 features)
        already happened during data collection. No further extraction
        (e.g. HOG/CNN features) is needed since landmarks are already a
        compact, highly discriminative representation of hand shape.
    """
    df = pd.read_csv(data_path)

    # ---- STEP 7: Input / Output separation -------------------------------
    feature_columns = [c for c in df.columns if c != "label"]
    X = df[feature_columns].values
    y_raw = df["label"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # ---- STEP 8: Train / Test split ---------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,          # keep class balance identical in train & test
    )

    # ---- STEP 8a: Feature Scaling ------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, label_encoder, feature_columns


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, le, cols = load_and_preprocess()
    print("STEP 5-8 : PREPROCESSING SUMMARY")
    print("-" * 50)
    print(f"Total features           : {len(cols)}")
    print(f"Classes                  : {list(le.classes_)}")
    print(f"X_train shape            : {X_train.shape}")
    print(f"X_test shape             : {X_test.shape}")
    print(f"Train class balance      : {np.bincount(y_train)}")
    print(f"Test class balance       : {np.bincount(y_test)}")
    print(f"Feature mean after scale (train) : {X_train.mean():.4f}")
    print(f"Feature std  after scale (train) : {X_train.std():.4f}")
