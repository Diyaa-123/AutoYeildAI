# CNN development sandbox

Work on improving the CNN **here** without touching the main app. When you're happy with the model, merge the changes back.

---

## Files that define the CNN (copy these for reference)

| In main project | Purpose |
|-----------------|--------|
| **`src/training/train_classifier.py`** | Training loop + **architecture** (EfficientNet-B0, classifier head, freeze/unfreeze). |
| **`src/inference/run_inference.py`** | Loads checkpoint for **prediction**. Must match architecture. |
| **`src/inference/gradcam.py`** | Loads checkpoint for **heatmaps**. Must match architecture + `model.features[-1]`. |
| **`src/evaluation/metrics.py`** | `compute_classification_metrics()` used by training. |

**Data:** `data/processed/train` and `data/processed/val` (ImageFolder: one folder per class).

**Checkpoint format:** `{"model_state_dict": ..., "class_names": [...]}`. Keep this when you save.

---

## How to use this folder

1. **Use data from the main project** (no copy): set `DATA_DIR` in `train.py` to the main project path (see below).
2. **Run training** from this folder:
   ```bash
   cd cnn_dev
   pip install torch torchvision scikit-learn
   python train.py
   ```
3. **Change the model** in `train.py` only (different backbone, head, epochs, LR, augmentation). No API or Grad-CAM code here.
4. **When satisfied**, merge back (see below).

---

## What to merge back when you're done

| If you changed… | Copy back into main project |
|-----------------|-----------------------------|
| **Only training** (epochs, LR, augmentation, no architecture change) | `cnn_dev/train.py` → update **`src/training/train_classifier.py`** (training block + config). Copy **`cnn_dev/checkpoints/*.pt`** → **`models/baseline_model.pt`**. |
| **Architecture** (e.g. different model or classifier) | Update **all three**: **`src/training/train_classifier.py`**, **`src/inference/run_inference.py`**, **`src/inference/gradcam.py`** so they build the **same** model and load the same checkpoint. Copy the new **`.pt`** to **`models/baseline_model.pt`**. For Grad-CAM, set `target_layer` to the last conv layer of your new model (e.g. `model.features[-1]` for EfficientNet). |

Rule: **train**, **run_inference**, and **gradcam** must always build the same architecture and expect the same checkpoint keys.

---

## Optional: copy these from main project for reference

You can copy (not move) these into `cnn_dev/` to compare or diff when merging:

- `src/training/train_classifier.py` → `cnn_dev/ref_train_classifier.py`
- `src/inference/run_inference.py` → `cnn_dev/ref_run_inference.py`
- `src/inference/gradcam.py` → `cnn_dev/ref_gradcam.py`
- `src/evaluation/metrics.py` → `cnn_dev/ref_metrics.py`

Then merge from `train.py` and the ref files back into `src/` when done.
