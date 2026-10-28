
import os
import random
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_DIR = Path("/content/SecureSight/dataset")
OUTPUT_DIR = Path("/content/drive/MyDrive/SecureSight_Final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
NUM_WORKERS = 2

print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Augmentation only for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.08, contrast=0.08, saturation=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Separate dataset objects ensure validation images are not augmented
train_base = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_base = datasets.ImageFolder(DATA_DIR / "train", transform=eval_transform)

print("Class mapping:", train_base.class_to_idx)
assert train_base.class_to_idx == val_base.class_to_idx

# Deterministic 90/10 split
indices = torch.randperm(
    len(train_base), generator=torch.Generator().manual_seed(SEED)
).tolist()

val_size = int(0.10 * len(indices))
val_indices = indices[:val_size]
train_indices = indices[val_size:]

train_ds = Subset(train_base, train_indices)
val_ds = Subset(val_base, val_indices)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# Pretrained ResNet-50
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Fine-tune layer4 and fully connected layer
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4},
], weight_decay=1e-4)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

best_val_acc = 0.0
history = []
best_path = OUTPUT_DIR / "resnet50_best.pth"

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item() * labels.size(0)
        train_correct += (logits.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda")
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = 100 * train_correct / train_total
    val_loss = val_loss_sum / val_total
    val_acc = 100 * val_correct / val_total

    row = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    }
    history.append(row)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train loss {train_loss:.4f} | Train acc {train_acc:.2f}% | "
        f"Val loss {val_loss:.4f} | Val acc {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_base.class_to_idx,
            "input_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "epoch": epoch + 1,
            "val_accuracy": val_acc,
            "seed": SEED
        }, best_path)
        print("Saved best checkpoint:", best_path)

with open(OUTPUT_DIR / "resnet50_training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\nTraining completed.")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print("Checkpoint:", best_path)
