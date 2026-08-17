"""
train_model.py
---------------
STEP 9  : Model Building (baseline ANN)
STEP 10 : Hyperparameter Tuning (Optuna)
STEP 11 : Model Evaluation
STEP 12 : Model Saving (best model + scaler + label encoder)

Run:  python train_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import label_binarize

from preprocessing import load_and_preprocess

RANDOM_STATE = 42
os.makedirs("models", exist_ok=True)
os.makedirs("screenshots/model", exist_ok=True)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Load & preprocess data (Steps 5-8, reused from preprocessing.py)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test, scaler, label_encoder, feature_columns = load_and_preprocess()
n_features = X_train.shape[1]
n_classes = len(label_encoder.classes_)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples      : {X_test.shape[0]}")
print(f"Features          : {n_features}")
print(f"Classes           : {list(label_encoder.classes_)}")


# ---------------------------------------------------------------------------
# STEP 9 : MODEL BUILDING (baseline ANN)
# ---------------------------------------------------------------------------
def build_ann(n_layers=2, units=64, dropout=0.3, learning_rate=1e-3, activation="relu"):
    """Builds a simple feed-forward ANN classifier."""
    model = keras.Sequential(name="Gesture_ANN")
    model.add(layers.Input(shape=(n_features,)))

    for i in range(n_layers):
        model.add(layers.Dense(units, activation=activation, name=f"hidden_{i+1}"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(n_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


print("\n" + "=" * 70)
print("STEP 9 : BASELINE MODEL TRAINING")
print("=" * 70)

baseline_model = build_ann()
baseline_history = baseline_model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=40,
    batch_size=32,
    verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
)
baseline_test_acc = baseline_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Baseline ANN test accuracy: {baseline_test_acc:.4f}")


# ---------------------------------------------------------------------------
# STEP 10 : HYPERPARAMETER TUNING (OPTUNA)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10 : HYPERPARAMETER TUNING WITH OPTUNA")
print("=" * 70)


def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_categorical("units", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    model = build_ann(
        n_layers=n_layers, units=units, dropout=dropout,
        learning_rate=learning_rate, activation=activation,
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=batch_size,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)],
    )

    val_acc = max(history.history["val_accuracy"])
    return val_acc


study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=20, show_progress_bar=False)

print(f"\nBest trial validation accuracy : {study.best_value:.4f}")
print("Best hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

with open("models/best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)


# ---------------------------------------------------------------------------
# Rebuild & train FINAL model using best combination of parameters
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Training FINAL model with best hyperparameters")
print("=" * 70)

best = study.best_params
final_model = build_ann(
    n_layers=best["n_layers"],
    units=best["units"],
    dropout=best["dropout"],
    learning_rate=best["learning_rate"],
    activation=best["activation"],
)

final_history = final_model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=60,
    batch_size=best["batch_size"],
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
)

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(final_history.history["accuracy"], label="Train")
axes[0].plot(final_history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()

axes[1].plot(final_history.history["loss"], label="Train")
axes[1].plot(final_history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss over Epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend()

plt.tight_layout()
plt.savefig("screenshots/model/training_curves.png", dpi=120)
plt.close()


# ---------------------------------------------------------------------------
# STEP 11 : MODEL EVALUATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 11 : MODEL EVALUATION")
print("=" * 70)

y_pred_proba = final_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

y_test_bin = label_binarize(y_test, classes=range(n_classes))
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average="weighted", multi_class="ovr")

print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC-AUC    : {roc_auc:.4f}")

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("\nClassification Report:\n", report)

with open("models/evaluation_report.txt", "w") as f:
    f.write(f"Accuracy   : {accuracy:.4f}\n")
    f.write(f"Precision  : {precision:.4f}\n")
    f.write(f"Recall     : {recall:.4f}\n")
    f.write(f"F1 Score   : {f1:.4f}\n")
    f.write(f"ROC-AUC    : {roc_auc:.4f}\n\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Gesture Recognition ANN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("screenshots/model/confusion_matrix.png", dpi=120)
plt.close()

# Baseline vs Tuned comparison
plt.figure(figsize=(7, 5))
plt.bar(["Baseline ANN", "Optuna-Tuned ANN"], [baseline_test_acc, accuracy],
        color=["#94a3b8", "#2563eb"])
plt.ylim(0, 1)
plt.ylabel("Test Accuracy")
plt.title("Baseline vs Hyperparameter-Tuned Model")
for i, v in enumerate([baseline_test_acc, accuracy]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("screenshots/model/baseline_vs_tuned.png", dpi=120)
plt.close()


# ---------------------------------------------------------------------------
# STEP 12 : MODEL SAVING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 12 : MODEL SAVING")
print("=" * 70)

final_model.save("models/gesture_ann_model.keras")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

print("Saved:")
print("  models/gesture_ann_model.keras")
print("  models/scaler.pkl")
print("  models/label_encoder.pkl")
print("  models/feature_columns.json")
print("  models/best_params.json")
print("  models/evaluation_report.txt")
print("\nTraining pipeline complete.")
