import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.evaluation.metrics import compute_classification_metrics

# =========================
# CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data", "processed"))
MODEL_SAVE_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "models", "baseline_model.pt"))

BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 3e-4
NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# =========================
# LOAD DATA
# =========================
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=val_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Detected classes:", class_names)

# =========================
# MODEL
# =========================
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Freeze most layers
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last 2 blocks
for param in model.features[-2:].parameters():
    param.requires_grad = True

# Replace classifier
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

model = model.to(DEVICE)

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)

# =========================
# TRAINING
# =========================
def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")
        val_acc = evaluate()
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

    print("Training complete.")


# =========================
# EVALUATION
# =========================
def evaluate(return_all=False):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    acc = accuracy_score(targets, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    if return_all:
        return acc, preds, targets
    return acc


# =========================
# SAVE MODEL
# =========================
def save_model(accuracy, metrics_payload=None):
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    torch.save(
        {"model_state_dict": model.state_dict(), "class_names": class_names},
        MODEL_SAVE_PATH,
    )

    with open("outputs/metrics/accuracy_before.txt", "w") as f:
        f.write(f"Validation Accuracy: {accuracy:.4f}\n")

    if metrics_payload:
        with open("outputs/metrics/model_metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)

    print(f"Model saved to {MODEL_SAVE_PATH}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()
    accuracy, preds, targets = evaluate(return_all=True)
    metrics_payload = compute_classification_metrics(targets, preds, class_names)
    save_model(accuracy, metrics_payload)
    final_accuracy, preds, targets = evaluate(return_all=True)
    metrics_payload = compute_classification_metrics(targets, preds, class_names)
    save_model(final_accuracy, metrics_payload)
