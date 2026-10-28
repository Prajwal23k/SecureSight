
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

DATA_DIR = Path("/content/SecureSight/dataset")
OUTPUT_DIR = Path("/content/drive/MyDrive/SecureSight_Final")
FEATURE_PATH = OUTPUT_DIR / "ddpm_train_features.npz"

SEED = 42
random.seed(SEED)
torch.manual_seed(1000)

TIMESTEPS = [100, 300, 500, 700, 900]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match the original DDPM pipeline: 32x32 and [-1, 1] normalization
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Balanced subset from the TRAIN split only
selected = []
for label_name in ["FAKE", "REAL"]:
    folder = DATA_DIR / "train" / label_name
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    chosen = random.sample(paths, min(5000, len(paths)))
    selected.extend([(p, 0 if label_name == "FAKE" else 1) for p in chosen])

random.shuffle(selected)
print("Selected images:", len(selected))
print("Expected: 10000 images, 5000 per class")

def extract_features(image_tensor):
    x = image_tensor.unsqueeze(0).to(DEVICE)
    stats = []

    for timestep_value in TIMESTEPS:
        # Preserve the existing deterministic timestep-seed approach
        torch.manual_seed(1000 + timestep_value)
        t = torch.tensor([timestep_value], device=DEVICE)
        noise = torch.randn_like(x)
        noisy = scheduler.add_noise(x, noise, t)

        with torch.no_grad():
            predicted_noise = ddpm(noisy, t).sample

        diff = predicted_noise - noise

        # Eight statistics per timestep, 40 features total
        stats.extend([
            diff.mean().item(),
            diff.std().item(),
            diff.abs().mean().item(),
            diff.abs().std().item(),
            diff.min().item(),
            diff.max().item(),
            (diff ** 2).mean().item(),
            torch.sqrt((diff ** 2).mean()).item()
        ])

    return np.asarray(stats, dtype=np.float32)

X = np.zeros((len(selected), 40), dtype=np.float32)
y = np.zeros(len(selected), dtype=np.int64)

for i, (path, label) in enumerate(tqdm(selected, desc="Extracting DDPM features")):
    with Image.open(path) as img:
        img = img.convert("RGB")
        tensor = transform(img)

    X[i] = extract_features(tensor)
    y[i] = label

    if (i + 1) % 500 == 0:
        np.savez_compressed(FEATURE_PATH, X=X[:i+1], y=y[:i+1])
        print(f"Saved progress: {i+1}/{len(selected)}")

np.savez_compressed(FEATURE_PATH, X=X, y=y)

print("Feature extraction complete.")
print("Feature shape:", X.shape)
print("Label counts:", np.bincount(y))
print("Saved:", FEATURE_PATH)
