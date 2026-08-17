# 🖐️ GestureSense: Touchless Human-Computer Interaction using ANN & MediaPipe

A touchless human-computer interaction system that recognizes 8 hand gestures in
real time using MediaPipe hand-landmark detection and an Artificial Neural Network
(ANN), deployed as an interactive Streamlit web application.

---

## 1. Problem Statement

Traditional device interaction relies on physical touch. This project builds a
system that recognizes hand gestures from a webcam feed in real time, enabling
touchless control for accessibility, hygiene-sensitive environments, and
hands-free interfaces.

## 2. Business Objective

Deliver a lightweight, real-time gesture classifier deployable for touchless
media control, presentation navigation, smart-home commands, or accessibility
tooling — using only a standard webcam, no specialized hardware.

## 3. Dataset

- **Type:** Custom, self-collected via webcam using `collect_data.py`
- **Source:** MediaPipe Hands landmark extraction (21 landmarks × x,y,z = 63 features/sample)
- **Classes (8):** `thumbs_up`, `thumbs_down`, `open_palm`, `fist`, `peace`, `ok_sign`, `pointing`, `call_me`
- **Target Variable:** `label` (categorical, 8 classes)
- **Input Features:** 63 numeric landmark coordinates per sample

> **Note:** `dataset/gesture_data.csv` currently contains a synthetically generated
> placeholder dataset (see `generate_synthetic_dataset.py`) so the full pipeline
> can be demonstrated end-to-end. Replace it with your own webcam-collected data
> using `collect_data.py` before final submission.

## 4. Project Workflow

| Step | Description | Script |
|------|-------------|--------|
| 1-2  | Problem Selection & Dataset Collection | `collect_data.py` |
| 3-4  | Dataset Import & EDA | `eda.py` |
| 5-8  | Preprocessing, Feature Prep, Train/Test Split, Scaling | `preprocessing.py` |
| 9-12 | Model Building, Optuna Tuning, Evaluation, Saving | `train_model.py` |
| 12a  | MediaPipe Inference Integration | `predictor.py` |
| 13   | Application Development | `app.py` |

## 5. Model & Results

- **Architecture:** Feed-forward ANN (Dense + BatchNormalization + Dropout layers)
- **Tuning:** Optuna (20 trials) — tuned layers, units, dropout, learning rate, batch size, activation
- **Final Test Metrics:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.994 |
| Precision (weighted) | 0.994 |
| Recall (weighted) | 0.994 |
| F1 Score (weighted) | 0.994 |
| ROC-AUC (weighted, OvR) | 1.000 |

See `screenshots/model/` for the confusion matrix, training curves, and
baseline-vs-tuned comparison chart. See `screenshots/eda/` for exploratory
data analysis visualizations.

## 6. Folder Structure

```
gesture_project/
│── app.py                        # Streamlit application
│── train_model.py                # Model building, tuning, evaluation, saving
│── preprocessing.py              # Preprocessing / train-test split / scaling
│── predictor.py                  # Inference wrapper (MediaPipe + model)
│── eda.py                        # Exploratory Data Analysis
│── collect_data.py               # Webcam-based custom dataset collection
│── generate_synthetic_dataset.py # Demo-only synthetic dataset generator
│── requirements.txt
│── README.md
│── models/                       # Saved model, scaler, label encoder
│── dataset/                      # gesture_data.csv
│── screenshots/                  # EDA and model evaluation plots
```

## 7. Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Collect your own gesture data via webcam
python collect_data.py

# 3. Run EDA
python eda.py

# 4. Train the model (includes Optuna tuning)
python train_model.py

# 5. Launch the app
streamlit run app.py
```

## 8. Application Features

- **Home** — Overview, key metrics, supported gestures
- **Project Description** — Problem statement, objective, workflow
- **Prediction** — Upload an image OR use your webcam; view detected landmarks,
  predicted gesture, confidence score, and full probability breakdown

## 9. Challenges Faced

- Distinguishing visually similar gestures (e.g., `fist` vs `thumbs_down`) required
  careful feature scaling and sufficient model capacity to separate subtle
  landmark differences.
- Ensuring robustness to hand rotation, scale, and position in frame required
  building augmentation (random rotation/scale/translation) into the data
  pipeline.

## 10. Future Scope

- Expand to a larger gesture vocabulary (e.g., full ASL alphabet)
- Add temporal modeling (LSTM/GRU) to recognize dynamic gestures (motion-based,
  not just static poses)
- Support two-hand gesture combinations
- Package as a browser extension or OS-level accessibility tool
