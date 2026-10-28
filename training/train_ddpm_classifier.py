
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

OUTPUT_DIR = Path("/content/drive/MyDrive/SecureSight_Final")
FEATURE_PATH = OUTPUT_DIR / "ddpm_train_features.npz"

data = np.load(FEATURE_PATH)
X = data["X"]
y = data["y"]

print("Features:", X.shape)
print("Labels:", np.bincount(y))

# Split the extracted training features into classifier train/validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

classifier = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

classifier.fit(X_train, y_train)

val_probs = classifier.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.50).astype(int)

print("\nDDPM classifier validation accuracy:",
      f"{accuracy_score(y_val, val_preds) * 100:.2f}%")

print("\nClassification report:")
print(classification_report(
    y_val, val_preds,
    labels=[0, 1],
    target_names=["FAKE", "REAL"],
    digits=4,
    zero_division=0
))

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_val, val_preds, labels=[0, 1]))

classifier_path = OUTPUT_DIR / "ddpm_classifier.joblib"
joblib.dump(classifier, classifier_path)

metadata = {
    "model_id": "google/ddpm-cifar10-32",
    "timesteps": [100, 300, 500, 700, 900],
    "feature_count": 40,
    "labels": {"FAKE": 0, "REAL": 1},
    "threshold": 0.50,
    "seed": 42,
    "validation_accuracy": float(accuracy_score(y_val, val_preds))
}

with open(OUTPUT_DIR / "ddpm_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved classifier:", classifier_path)
print("Saved metadata:", OUTPUT_DIR / "ddpm_metadata.json")

