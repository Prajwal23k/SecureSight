import os
import json
import random
import time
import numpy as np
import torch
import joblib

from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from diffusers import UNet2DModel, DDPMScheduler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
DATA_DIR = Path("/content/SecureSight/dataset/test")
OUT_DIR = Path("/content/drive/MyDrive/SecureSight_Final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUT_DIR / "resnet50_best.pth"
CLASSIFIER_PATH = OUT_DIR / "ddpm_classifier.joblib"
CONFIG_PATH = OUT_DIR / "fusion_config.json"

PROGRESS_PATH = OUT_DIR / "final_test_progress.npz"
RESULTS_PATH = OUT_DIR / "final_test_results.json"
PREDICTIONS_PATH = OUT_DIR / "final_test_predictions.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN_WEIGHT = 0.70
DDPM_WEIGHT = 0.30
THRESHOLD = 0.50
TIMESTEPS = [100, 300, 500, 700, 900]
SAVE_EVERY = 500

# --------------------------------------------------
# 2. Verify configuration and create fixed test order
# --------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

assert config["labels"] == {"FAKE": 0, "REAL": 1}
assert config["ddpm_feature_count"] == 40
assert config["ddpm_timesteps"] == TIMESTEPS

paths = []
labels = []

for class_name, label in [("FAKE", 0), ("REAL", 1)]:
    folder = DATA_DIR / class_name
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in
        [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    )
    paths.extend([str(p) for p in files])
    labels.extend([label] * len(files))

paths = np.array(paths)
y_true = np.array(labels, dtype=np.int64)

assert len(paths) == 20000, f"Expected 20,000 test images, found {len(paths)}"

# --------------------------------------------------
# 3. Load ResNet-50 checkpoint
# --------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

cnn = models.resnet50(weights=None)
cnn.fc = torch.nn.Linear(cnn.fc.in_features, 2)
cnn.load_state_dict(checkpoint["model_state_dict"])
cnn = cnn.to(DEVICE).eval()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# 4. Load DDPM and its trained classifier
# --------------------------------------------------
ddpm_classifier = joblib.load(CLASSIFIER_PATH)

model_id = config["ddpm_model_id"]
ddpm = UNet2DModel.from_pretrained(model_id).to(DEVICE).eval()
scheduler = DDPMScheduler.from_pretrained(model_id)

ddpm_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --------------------------------------------------
# 5. Exact 40-feature extraction used for DDPM classifier
# --------------------------------------------------
@torch.inference_mode()
def extract_ddpm_features(image):
    x = ddpm_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    features = []

    for t in TIMESTEPS:
        # Match the deterministic noise-seeding convention
        # used during DDPM feature extraction.
        torch.manual_seed(1000 + t)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1000 + t)

        noise = torch.randn_like(x)
        t_tensor = torch.tensor([t], device=DEVICE, dtype=torch.long)
        noisy_x = scheduler.add_noise(x, noise, t_tensor)

        predicted_noise = ddpm(
            noisy_x, t_tensor
        ).sample

        error = predicted_noise - noise

        stats = [
            error.mean().item(),
            error.std(unbiased=False).item(),
            error.abs().mean().item(),
            error.abs().std(unbiased=False).item(),
            error.min().item(),
            error.max().item(),
            torch.mean(error ** 2).item(),
            torch.sqrt(torch.mean(error ** 2)).item()
        ]
        features.extend(stats)

    result = np.asarray(features, dtype=np.float32)
    assert result.shape == (40,)
    return result

# --------------------------------------------------
# 6. Resume previous progress, if available
# --------------------------------------------------
n = len(paths)

cnn_scores = np.full(n, np.nan, dtype=np.float32)
ddpm_scores = np.full(n, np.nan, dtype=np.float32)
completed = np.zeros(n, dtype=bool)

if PROGRESS_PATH.exists():
    saved = np.load(PROGRESS_PATH, allow_pickle=False)

    # Resume only when the saved test order matches this run.
    if (
        len(saved["paths"]) == n
        and np.array_equal(saved["paths"], paths)
    ):
        cnn_scores = saved["cnn_fake_scores"].copy()
        ddpm_scores = saved["ddpm_fake_scores"].copy()
        completed = saved["completed"].astype(bool).copy()
        print(f"Resuming: {completed.sum()}/{n} images already processed.")
    else:
        print("Existing checkpoint has a different test order; starting fresh.")

# --------------------------------------------------
# 7. Evaluate the fixed hybrid model
# --------------------------------------------------
start_time = time.time()
last_save = time.time()

for i in range(n):
    if completed[i]:
        continue

    try:
        with Image.open(paths[i]) as im:
            image = im.convert("RGB")

            # CNN: class index 0 is FAKE.
            x_cnn = cnn_transform(image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                logits = cnn(x_cnn)
                probs = torch.softmax(logits, dim=1)
                cnn_fake = probs[0, 0].item()

            # DDPM classifier was trained with FAKE=0, REAL=1.
            features = extract_ddpm_features(image)
            probabilities = ddpm_classifier.predict_proba(
                features.reshape(1, -1)
            )[0]

            # Locate columns using the classifier's actual class labels.
            classes = list(ddpm_classifier.classes_)
            fake_column = classes.index(0)
            ddpm_fake = float(probabilities[fake_column])

            cnn_scores[i] = cnn_fake
            ddpm_scores[i] = ddpm_fake
            completed[i] = True

    except Exception as e:
        # Do not silently count failed images as evaluated.
        print(f"\nERROR at image {i}: {paths[i]}\n{type(e).__name__}: {e}")
        raise

    if (i + 1) % SAVE_EVERY == 0 or i == n - 1:
        np.savez_compressed(
            PROGRESS_PATH,
            paths=paths,
            y_true=y_true,
            cnn_fake_scores=cnn_scores,
            ddpm_fake_scores=ddpm_scores,
            completed=completed
        )

        elapsed = time.time() - start_time
        print(
            f"Loop position {i + 1}/{n} | "
            f"completed {completed.sum()}/{n} | "
            f"elapsed {elapsed / 60:.1f} min"
        )

assert completed.all(), "Some test images remain incomplete."

# --------------------------------------------------
# 8. Calculate metrics using the fixed 70:30 fusion
# --------------------------------------------------
fusion_scores = CNN_WEIGHT * cnn_scores + DDPM_WEIGHT * ddpm_scores

# Label 0 = FAKE, label 1 = REAL.
y_pred = np.where(fusion_scores >= THRESHOLD, 0, 1)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["FAKE", "REAL"],
    output_dict=True,
    zero_division=0
)

results = {
    "project": "SecureSight",
    "architecture": "ResNet-50 + pretrained DDPM",
    "test_images": int(n),
    "test_class_counts": {
        "FAKE": int((y_true == 0).sum()),
        "REAL": int((y_true == 1).sum())
    },
    "cnn_weight": CNN_WEIGHT,
    "ddpm_weight": DDPM_WEIGHT,
    "fake_threshold": THRESHOLD,
    "accuracy_percent": round(float(accuracy_score(y_true, y_pred) * 100), 4),
    "macro_f1_percent": round(float(f1_score(y_true, y_pred, average="macro") * 100), 4),
    "confusion_matrix_labels": ["FAKE", "REAL"],
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "real_false_positive_rate_percent": round(
        float(cm[1, 0] / max(cm[1].sum(), 1) * 100), 4
    ),
    "notes": (
        "Evaluation on the separate test directory. "
        "Fusion weights were fixed before test evaluation."
    )
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    PREDICTIONS_PATH,
    paths=paths,
    y_true=y_true,
    cnn_fake_scores=cnn_scores,
    ddpm_fake_scores=ddpm_scores,
    fusion_fake_scores=fusion_scores,
    y_pred=y_pred
)

print("\n========== SECURESIGHT FINAL TEST RESULTS ==========")
print(f"Test images: {n}")
print(f"Accuracy: {results['accuracy_percent']:.2f}%")
print(f"Macro F1: {results['macro_f1_percent']:.2f}%")
print(f"REAL false-positive rate: {results['real_false_positive_rate_percent']:.2f}%")
print("\nConfusion matrix (rows=true, columns=predicted; FAKE, REAL):")
print(cm)
print("\nClassification report:")
print(classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["FAKE", "REAL"],
    zero_division=0
))
print("\nSaved:")
print(RESULTS_PATH)
print(PREDICTIONS_PATH)
print(PROGRESS_PATH)
