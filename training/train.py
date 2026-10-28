
import os
import sys
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TRAIN_PATH = os.path.join(BASE_DIR, "dataset_small", "train")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "saved_model")
NEW_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model_augmented.pth")

sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
from model import CNNModel

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Training augmentation: moderate changes only
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# Validation images are not augmented
val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# Separate dataset instances allow different transforms
full_train_aug = datasets.ImageFolder(
    TRAIN_PATH, transform=train_transform
)
full_train_val = datasets.ImageFolder(
    TRAIN_PATH, transform=val_transform
)

print("Class mapping:", full_train_aug.class_to_idx)

# 90% training, 10% validation
indices = list(range(len(full_train_aug)))
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(
    len(indices), generator=generator
).tolist()

split = int(0.9 * len(indices))
train_indices = indices[:split]
val_indices = indices[split:]

from torch.utils.data import Subset

train_dataset = Subset(full_train_aug, train_indices)
val_dataset = Subset(full_train_val, val_indices)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False,
    num_workers=0
)

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5
best_val_accuracy = -1
best_state = None

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        train_loss_sum += loss.item() * batch_size
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += batch_size

    train_loss = train_loss_sum / train_total
    train_accuracy = 100 * train_correct / train_total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            val_loss_sum += loss.item() * batch_size
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += batch_size

    val_loss = val_loss_sum / val_total
    val_accuracy = 100 * val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())

os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(best_state, NEW_MODEL_PATH)

print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print("New model saved to:", NEW_MODEL_PATH)
print("Original cnn_model.pth was not overwritten.")
