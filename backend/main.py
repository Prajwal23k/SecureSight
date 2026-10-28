
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from diffusers import UNet2DModel, DDPMScheduler

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"

CNN_PATH = MODEL_DIR / "resnet50_best.pth"
DDPM_CLASSIFIER_PATH = MODEL_DIR / "ddpm_classifier.joblib"
METADATA_PATH = MODEL_DIR / "ddpm_metadata.json"
CONFIG_PATH = MODEL_DIR / "fusion_config.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecureSight")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------
for path in [
    CNN_PATH,
    DDPM_CLASSIFIER_PATH,
    METADATA_PATH,
    CONFIG_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(f"Required model file not found: {path}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    DDPM_METADATA = json.load(f)

CNN_WEIGHT = float(CONFIG["cnn_weight"])
DDPM_WEIGHT = float(CONFIG["ddpm_weight"])
FAKE_THRESHOLD = float(CONFIG["fake_threshold"])
TIMESTEPS = [int(t) for t in CONFIG["ddpm_timesteps"]]
DDPM_MODEL_ID = CONFIG["ddpm_model_id"]

if not np.isclose(CNN_WEIGHT + DDPM_WEIGHT, 1.0):
    raise ValueError("CNN and DDPM weights must sum to 1.")

if int(CONFIG["ddpm_feature_count"]) != 40:
    raise ValueError("Expected a 40-feature DDPM classifier.")

# ---------------------------------------------------------
# FastAPI and CORS
# ---------------------------------------------------------
app = FastAPI(
    title="SecureSight API",
    description="AI-generated image detection using ResNet-50 and DDPM fusion",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

ddpm_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])

# ---------------------------------------------------------
# Load ResNet-50
# ---------------------------------------------------------
logger.info("Loading ResNet-50 checkpoint...")

checkpoint = torch.load(
    CNN_PATH,
    map_location=DEVICE,
    weights_only=False,
)

cnn_model = models.resnet50(weights=None)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
cnn_model.load_state_dict(checkpoint["model_state_dict"])
cnn_model = cnn_model.to(DEVICE).eval()

# ---------------------------------------------------------
# Load DDPM and classifier
# ---------------------------------------------------------
logger.info("Loading DDPM classifier...")
ddpm_classifier = joblib.load(DDPM_CLASSIFIER_PATH)

logger.info("Loading pretrained DDPM model: %s", DDPM_MODEL_ID)
ddpm_model = UNet2DModel.from_pretrained(DDPM_MODEL_ID)
ddpm_model = ddpm_model.to(DEVICE).eval()
ddpm_scheduler = DDPMScheduler.from_pretrained(DDPM_MODEL_ID)

# Check classifier label ordering.
classifier_classes = list(ddpm_classifier.classes_)
if 0 not in classifier_classes or 1 not in classifier_classes:
    raise ValueError(
        f"Unexpected DDPM classifier classes: {classifier_classes}"
    )

logger.info("SecureSight models loaded on %s", DEVICE)

# ---------------------------------------------------------
# CNN prediction
# Class mapping: FAKE=0, REAL=1
# ---------------------------------------------------------
@torch.inference_mode()
def get_cnn_fake_score(image: Image.Image) -> float:
    tensor = cnn_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = cnn_model(tensor)
    probabilities = torch.softmax(logits, dim=1)

    return float(probabilities[0, 0].item())


# ---------------------------------------------------------
# DDPM 40-feature extraction
# Eight statistics at each of five timesteps.
# ---------------------------------------------------------
@torch.inference_mode()
def extract_ddpm_features(image: Image.Image) -> np.ndarray:
    tensor = (
        ddpm_transform(image.convert("RGB"))
        .unsqueeze(0)
        .to(DEVICE)
    )

    features = []

    for timestep in TIMESTEPS:
        # Match the deterministic noise convention used during
        # training and final hybrid evaluation.
        torch.manual_seed(1000 + timestep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1000 + timestep)

        noise = torch.randn_like(tensor)
        timestep_tensor = torch.tensor(
            [timestep], dtype=torch.long, device=DEVICE
        )

        noisy_image = ddpm_scheduler.add_noise(
            tensor, noise, timestep_tensor
        )

        predicted_noise = ddpm_model(
            noisy_image, timestep_tensor
        ).sample

        error = predicted_noise - noise

        statistics = [
            error.mean().item(),
            error.std(unbiased=False).item(),
            error.abs().mean().item(),
            error.abs().std(unbiased=False).item(),
            error.min().item(),
            error.max().item(),
            torch.mean(error ** 2).item(),
            torch.sqrt(torch.mean(error ** 2)).item(),
        ]

        features.extend(statistics)

    features = np.asarray(features, dtype=np.float32)

    if features.shape != (40,):
        raise ValueError(
            f"Expected 40 DDPM features, got {features.shape}"
        )

    return features


# ---------------------------------------------------------
# DDPM classifier prediction
# Class mapping: FAKE=0, REAL=1
# ---------------------------------------------------------
def get_ddpm_fake_score(image: Image.Image) -> float:
    features = extract_ddpm_features(image).reshape(1, -1)
    probabilities = ddpm_classifier.predict_proba(features)[0]

    fake_index = classifier_classes.index(0)
    return float(probabilities[fake_index])


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "SecureSight API is running",
        "architecture": "ResNet-50 + pretrained DDPM",
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cnn_loaded": True,
        "ddpm_loaded": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Prevent excessively large uploads.
        if len(contents) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 15 MB.",
            )

        import io
        image = Image.open(io.BytesIO(contents)).convert("RGB")

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be decoded as an image.",
        )

    try:
        cnn_fake = get_cnn_fake_score(image)
        ddpm_fake = get_ddpm_fake_score(image)

        fusion_fake = (
            CNN_WEIGHT * cnn_fake
            + DDPM_WEIGHT * ddpm_fake
        )

        prediction = "FAKE" if fusion_fake >= FAKE_THRESHOLD else "REAL"

        confidence = (
            fusion_fake if prediction == "FAKE"
            else 1.0 - fusion_fake
        )

        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 6),
            "cnn_score": round(float(cnn_fake), 6),
            "diffusion_score": round(float(ddpm_fake), 6),
            "fusion_score": round(float(fusion_fake), 6),
            "cnn_weight": CNN_WEIGHT,
            "diffusion_weight": DDPM_WEIGHT,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Image prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend terminal for details.",
        )
