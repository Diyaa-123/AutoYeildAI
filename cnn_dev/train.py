"""
Standalone CNN training – no dependency on main project's src.*
Run from this folder:  python train.py
Uses data from main project: ../data/processed (change DATA_DIR below if needed)
Saves checkpoint to cnn_dev/checkpoints/ so the main app is not touched.

Goal: BEST accuracy while keeping SAME architecture (EfficientNet-B0 + Linear head).

Key fixes vs the previous version (important for "edge ring / center / local"):
- NO RandomResizedCrop (it destroys global position cues)
- NO HorizontalFlip (can break spatial semantics)
- EfficientNet/ImageNet normalization (correct pretrained preprocessing)
- Staged finetune: head-only warmup -> unfreeze last N blocks
- Discriminative LR: head higher, backbone lower
- AdamW + OneCycleLR
- Mixed precision (CUDA) for stability/speed
- Save BEST checkpoint by MACRO F1 (not just accuracy)
- Early stopping
- Optional WeightedRandomSampler (auto if imbalance detected)
"""
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# =========================
# CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use main project's data (no copy needed)
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))

# Save checkpoints here, not in main project
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Output metrics
METRICS_DIR = os.path.join(SCRIPT_DIR, "outputs", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

# Training hyperparams
BATCH_SIZE = 32
NUM_EPOCHS = 25
NUM_WORKERS = 2
SEED = 42

# Finetuning strategy (same architecture)
WARMUP_EPOCHS_HEAD_ONLY = 3
UNFREEZE_LAST_N_BLOCKS = 4  # try 4 for subtle classes; set to 2 if you want original behavior

# Learning rates (discriminative)
LR_HEAD = 8e-4
LR_BACKBONE = 2e-5
WEIGHT_DECAY = 1e-2

# Regularization
LABEL_SMOOTHING = 0.05  # used only if not using Focal Loss
# Focal Loss (down-weights easy examples, helps hard/minority classes)
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0  # higher = more focus on hard examples
FOCAL_ALPHA = None  # None = uniform; or list per-class weight (length num_classes)

# Early stopping
EARLY_STOP_PATIENCE = 6
MIN_DELTA = 1e-4

# Imbalance handling
USE_WEIGHTED_SAMPLER_IF_IMBALANCED = True
IMBALANCE_RATIO_THRESHOLD = 1.5  # if max/min >= this, sampler on

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
print("Data:", DATA_DIR)
print("Checkpoints:", CHECKPOINT_DIR)

# =========================
# Reproducibility
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# =========================
# Metrics (same as src.evaluation.metrics)
# =========================
def compute_classification_metrics(targets, preds, class_names):
    accuracy = float(accuracy_score(targets, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, labels=list(range(len(class_names))), zero_division=0
    )
    conf = confusion_matrix(targets, preds, labels=list(range(len(class_names))))

    per_class = []
    targets_np = np.array(targets)
    for idx, name in enumerate(class_names):
        per_class.append({
            "label": name,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(np.sum(targets_np == idx)),
        })

    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)) if len(precision) else 0.0,
        "macro_recall": float(np.mean(recall)) if len(recall) else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": conf.tolist(),
    }


# =========================
# Focal Loss
# =========================
class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples, focuses on hard/minority classes."""

    def __init__(self, num_classes, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,) long
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # p_t
        focal_weight = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)
        nll = -(targets_one_hot * log_probs).sum(dim=1)
        loss = focal_weight * nll
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =========================
# Transforms (POSITION-SAFE)
# =========================
# ImageNet normalization (works across torchvision versions; avoids weights.transforms() API changes)
imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),     # preserve global geometry
    transforms.RandomRotation(3),   # mild rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # mild lighting robustness
    transforms.ToTensor(),
    imagenet_norm,
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    imagenet_norm,
])

# =========================
# Data
# =========================
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transform,
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=val_transform,
)

# CRITICAL: train and val must use the same class order (same folder names + same sort).
# If they differ (e.g. "Center" vs "center"), labels are wrong and Edge ring/Local/center get 0.
train_classes = list(train_dataset.classes)
val_classes = list(val_dataset.classes)
if train_classes != val_classes:
    print("ERROR: Train and val class order mismatch!")
    print("Train classes:", train_classes)
    print("Val   classes:", val_classes)
    print("Fix: rename val folders to match train exactly (e.g. val 'Center' -> 'center').")
    raise SystemExit(1)
class_names = train_classes
num_classes = len(class_names)
print("Classes (train & val match):", class_names)

# Imbalance detection
targets_list = [y for _, y in train_dataset.samples]
class_counts = Counter(targets_list) if len(targets_list) else Counter()
min_count = min(class_counts.values()) if len(class_counts) else 1
max_count = max(class_counts.values()) if len(class_counts) else 1
imbalance_ratio = max_count / max(1, min_count)
print("Train class counts:", {class_names[k]: v for k, v in class_counts.items()})
print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")

sampler = None
shuffle = True
if USE_WEIGHTED_SAMPLER_IF_IMBALANCED and imbalance_ratio >= IMBALANCE_RATIO_THRESHOLD:
    class_weight = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weight[y] for y in targets_list]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    shuffle = False
    print("Using WeightedRandomSampler (imbalance detected).")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=shuffle,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
)

# =========================
# Model – SAME architecture
# =========================
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Replace classifier head
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

def set_trainable_last_n_blocks(n_blocks: int):
    # Freeze all backbone
    for p in model.features.parameters():
        p.requires_grad = False
    # Unfreeze last N feature blocks
    if n_blocks > 0:
        for p in model.features[-n_blocks:].parameters():
            p.requires_grad = True
    # Always train head
    for p in model.classifier.parameters():
        p.requires_grad = True

# Start head-only
set_trainable_last_n_blocks(0)

model = model.to(DEVICE)

if USE_FOCAL_LOSS:
    criterion = FocalLoss(num_classes=num_classes, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA).to(DEVICE)
    print(f"Loss: Focal (gamma={FOCAL_GAMMA})")
else:
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    print("Loss: CrossEntropy (label_smoothing={})".format(LABEL_SMOOTHING))

# =========================
# Optimizer / Scheduler
# =========================
def build_optimizer():
    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": LR_HEAD}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": LR_BACKBONE})

    return optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

def build_scheduler(optimizer, steps_per_epoch, total_epochs):
    max_lr = [g["lr"] for g in optimizer.param_groups]
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

# =========================
# Eval
# =========================
@torch.no_grad()
def evaluate(return_all=False):
    model.eval()
    preds, targets = [], []
    for images, labels in val_loader:
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        preds.extend(predicted.cpu().numpy())
        targets.extend(labels.numpy())

    acc = accuracy_score(targets, preds) if len(targets) else 0.0
    if return_all:
        return acc, preds, targets
    return acc

# =========================
# Save best model (by macro F1)
# =========================
def save_checkpoint(tag, best_score, metrics_payload):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{tag}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "best_score": float(best_score),
            "tag": tag,
        },
        ckpt_path,
    )
    with open(os.path.join(METRICS_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)
    with open(os.path.join(METRICS_DIR, f"{tag}_score.txt"), "w") as f:
        f.write(f"{tag} score: {best_score:.6f}\n")
    print(f"Saved checkpoint: {ckpt_path}")

# =========================
# Train (staged finetune + best save + early stop)
# =========================
def train():
    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    def run_stage(stage_name, total_epochs, unfreeze_blocks):
        nonlocal best_macro_f1, best_epoch, epochs_no_improve

        set_trainable_last_n_blocks(unfreeze_blocks)
        optimizer = build_optimizer()
        scheduler = build_scheduler(optimizer, steps_per_epoch=len(train_loader), total_epochs=total_epochs)

        print(f"\n{stage_name}: epochs={total_epochs}, unfreeze_last_blocks={unfreeze_blocks}")
        print("Trainable params:",
              sum(p.numel() for p in model.parameters() if p.requires_grad),
              "/", sum(p.numel() for p in model.parameters()))

        for e in range(total_epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(train_loader))
            acc, preds, targets = evaluate(return_all=True)
            metrics_payload = compute_classification_metrics(targets, preds, class_names)

            macro_f1 = metrics_payload["macro_f1"]
            global_epoch = (best_epoch + 1) if best_epoch >= 0 else 0
            print(f"Epoch Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")

            # Track best by macro F1
            if macro_f1 > best_macro_f1 + MIN_DELTA:
                best_macro_f1 = macro_f1
                best_epoch = global_epoch
                epochs_no_improve = 0
                save_checkpoint("best_model", best_macro_f1, metrics_payload)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping (no macro-F1 improvement for {EARLY_STOP_PATIENCE} epochs).")
                    return False  # stop

        return True  # continue

    # Stage 1: head-only warmup
    if WARMUP_EPOCHS_HEAD_ONLY > 0:
        cont = run_stage("Stage 1 (head-only)", WARMUP_EPOCHS_HEAD_ONLY, unfreeze_blocks=0)
        if not cont:
            return best_macro_f1

    # Stage 2: finetune last N blocks
    remaining = NUM_EPOCHS - WARMUP_EPOCHS_HEAD_ONLY
    if remaining > 0:
        run_stage(f"Stage 2 (finetune last {UNFREEZE_LAST_N_BLOCKS})", remaining, unfreeze_blocks=UNFREEZE_LAST_N_BLOCKS)

    return best_macro_f1

# =========================
# Main
# =========================
if __name__ == "__main__":
    best_score = train()

    # Load best checkpoint for final reporting
    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded best checkpoint for final evaluation.")

    accuracy, preds, targets = evaluate(return_all=True)
    metrics_payload = compute_classification_metrics(targets, preds, class_names)

    # Write a final summary file too
    with open(os.path.join(METRICS_DIR, "final_summary.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)
    with open(os.path.join(METRICS_DIR, "final_accuracy.txt"), "w") as f:
        f.write(f"Final Validation Accuracy (best checkpoint): {accuracy:.4f}\n")
        f.write(f"Final Macro F1 (best checkpoint): {metrics_payload['macro_f1']:.4f}\n")

    print("Final validation accuracy (best checkpoint):", accuracy)
    print("Final macro F1 (best checkpoint):", metrics_payload["macro_f1"])
    print("Per-class F1:", [p["f1"] for p in metrics_payload["per_class"]])
    print("Train classes:", train_dataset.class_to_idx)
    print("Val classes:", val_dataset.class_to_idx)



