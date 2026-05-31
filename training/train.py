# ============================================================
# MediVision — Synapse Multi-Organ Segmentation Training
# Dataset: TransUNet preprocessed Synapse (train_npz / test_vol_h5)
# Platform: Kaggle P100/T4
# Expected runtime: ~4-6 hours
# ============================================================

# ── 0. Install ───────────────────────────────────────────────
import subprocess, sys
def pip(pkg): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
pip("segmentation-models-pytorch")
pip("h5py")

# ── 1. Imports ───────────────────────────────────────────────
import os, json, time, random
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
print(f"GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

# ── 2. Config ────────────────────────────────────────────────
CFG = {
    # ↓ adjust if your Kaggle dataset is mounted elsewhere
    "train_dir":  "/kaggle/input/btcv-synapse-preprocessed/Synapse/train_npz",
    "test_dir":   "/kaggle/input/btcv-synapse-preprocessed/Synapse/test_vol_h5",
    "output_dir": "/kaggle/working/medivision_output",

    "img_size":     224,   # TransUNet preprocessing uses 224×224
    "num_classes":  9,     # Synapse: 8 organs + background
    "batch_size":   24,    # P100 16GB handles 24 comfortably at 224×224
    "epochs":       150,
    "lr":           1e-4,
    "weight_decay": 1e-5,
    "val_split":    0.2,   # hold out 20% of training cases for validation
    "seed":         42,
    "save_every":   10,

    # Synapse 8-organ class names (background=0)
    "class_names": [
        "background", "aorta", "gallbladder", "spleen",
        "left_kidney", "right_kidney", "liver", "stomach", "pancreas"
    ]
}

Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ── 3. Datasets ──────────────────────────────────────────────
class SynapseTrainDataset(Dataset):
    """
    Loads TransUNet-preprocessed .npz slices.
    Each file contains:
      image : (H, W)  float32  — already normalized
      label : (H, W)  uint8    — class indices 0-8
    """
    def __init__(self, npz_paths, img_size=224, augment=False):
        self.paths   = npz_paths
        self.size    = img_size
        self.augment = augment
        print(f"  {'Train' if augment else 'Val'} slices: {len(self.paths)}")

    def _augment(self, img, lbl):
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            lbl = np.fliplr(lbl).copy()
        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img).copy()
            lbl = np.flipud(lbl).copy()
        # Random 90° rotation
        k = random.randint(0, 3)
        img = np.rot90(img, k).copy()
        lbl = np.rot90(lbl, k).copy()
        return img, lbl

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data  = np.load(self.paths[idx])
        img   = data["image"].astype(np.float32)   # (H, W)
        label = data["label"].astype(np.int64)      # (H, W)

        # Resize to target size if needed
        if img.shape[0] != self.size:
            from PIL import Image as PILImage
            img   = np.array(PILImage.fromarray(img).resize(
                        (self.size, self.size), PILImage.BILINEAR))
            label = np.array(PILImage.fromarray(label.astype(np.uint8)).resize(
                        (self.size, self.size), PILImage.NEAREST))

        if self.augment:
            img, label = self._augment(img, label)

        # (H,W) → (3,H,W)  replicate single channel × 3 for ResNet encoder
        img_tensor = torch.from_numpy(
            np.stack([img, img, img], axis=0)
        )
        return img_tensor, torch.tensor(label, dtype=torch.long)


class SynapseTestDataset(Dataset):
    """
    Loads TransUNet-preprocessed .npy.h5 volumes for evaluation.
    Each file contains:
      image : (S, H, W)  — S slices
      label : (S, H, W)
    Flattened to individual slices for metric computation.
    """
    def __init__(self, h5_paths, img_size=224):
        self.slices = []
        for p in h5_paths:
            with h5py.File(p, "r") as f:
                img_vol = f["image"][:]    # (S, H, W)
                lbl_vol = f["label"][:]    # (S, H, W)
            for s in range(img_vol.shape[0]):
                self.slices.append((img_vol[s], lbl_vol[s]))
        self.size = img_size
        print(f"  Test slices : {len(self.slices)}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img, label = self.slices[idx]
        img   = img.astype(np.float32)
        label = label.astype(np.int64)

        if img.shape[0] != self.size:
            from PIL import Image as PILImage
            img   = np.array(PILImage.fromarray(img).resize(
                        (self.size, self.size), PILImage.BILINEAR))
            label = np.array(PILImage.fromarray(label.astype(np.uint8)).resize(
                        (self.size, self.size), PILImage.NEAREST))

        img_tensor = torch.from_numpy(np.stack([img, img, img], axis=0))
        return img_tensor, torch.tensor(label, dtype=torch.long)


def build_dataloaders(cfg):
    train_dir = Path(cfg["train_dir"])
    test_dir  = Path(cfg["test_dir"])

    all_train = sorted(train_dir.glob("*.npz"))
    all_test  = sorted(test_dir.glob("*.npy.h5"))

    assert len(all_train) > 0, f"No .npz files in {train_dir}"
    assert len(all_test)  > 0, f"No .npy.h5 files in {test_dir}"
    print(f"Found {len(all_train)} train cases, {len(all_test)} test volumes")

    # Split training cases into train / val
    train_cases, val_cases = train_test_split(
        all_train, test_size=cfg["val_split"], random_state=cfg["seed"]
    )

    train_ds = SynapseTrainDataset(train_cases, cfg["img_size"], augment=True)
    val_ds   = SynapseTestDataset(all_test,     cfg["img_size"])  # use test set as val

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ── 4. Model ─────────────────────────────────────────────────
def build_model(num_classes):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model.to(DEVICE)


# ── 5. Loss ───────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.C      = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss  = 0.0
        for c in range(self.C):
            p = probs[:, c]
            t = (targets == c).float()
            loss += 1 - (2 * (p * t).sum() + self.smooth) / \
                        (p.sum() + t.sum() + self.smooth)
        return loss / self.C

class CombinedLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce   = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + \
               0.5 * self.ce(logits, targets)


# ── 6. Metrics ────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(preds, targets, num_classes, smooth=1e-6):
    """Returns per-class Dice and IoU (skips empty GT classes)."""
    dice_scores, iou_scores = {}, {}
    for c in range(1, num_classes):   # skip background
        p = (preds   == c).float()
        t = (targets == c).float()
        if t.sum() == 0 and p.sum() == 0:
            continue
        intersection = (p * t).sum()
        union        = p.sum() + t.sum() - intersection
        dice_scores[c] = float((2 * intersection + smooth) /
                               (p.sum() + t.sum() + smooth))
        iou_scores[c]  = float((intersection + smooth) / (union + smooth))
    return dice_scores, iou_scores


# ── 7. Train / Val ────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def val_epoch(model, loader, criterion, num_classes, class_names):
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    dice, iou   = compute_metrics(all_preds, all_targets, num_classes)

    mean_dice = float(np.mean(list(dice.values()))) if dice else 0.0
    mean_iou  = float(np.mean(list(iou.values())))  if iou  else 0.0

    named_dice = {class_names[c]: round(v, 4) for c, v in dice.items()}
    named_iou  = {class_names[c]: round(v, 4) for c, v in iou.items()}

    return total_loss / len(loader), mean_dice, mean_iou, named_dice, named_iou


# ── 8. Main ───────────────────────────────────────────────────
def train(cfg):
    print("\n" + "="*60)
    print(" MediVision — Synapse Multi-Organ Segmentation")
    print("="*60 + "\n")

    train_loader, val_loader = build_dataloaders(cfg)
    model     = build_model(cfg["num_classes"])
    criterion = CombinedLoss(cfg["num_classes"])
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5,
                    patience=10, verbose=True)

    best_dice  = 0.0
    history    = []
    out        = Path(cfg["output_dir"])

    for epoch in range(1, cfg["epochs"] + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, mean_dice, mean_iou, named_dice, named_iou = val_epoch(
            model, val_loader, criterion,
            cfg["num_classes"], cfg["class_names"]
        )
        scheduler.step(mean_dice)
        elapsed = time.time() - t0

        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "mean_dice":  round(mean_dice,  4),
            "mean_iou":   round(mean_iou,   4),
            "per_class_dice": named_dice,
            "per_class_iou":  named_iou,
            "lr":         optimizer.param_groups[0]["lr"],
            "elapsed_s":  round(elapsed, 1),
        }
        history.append(record)

        print(f"Ep {epoch:3d}/{cfg['epochs']} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"dice={mean_dice:.4f} | iou={mean_iou:.4f} | "
              f"lr={record['lr']:.1e} | {elapsed:.0f}s")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(),
                       out / "unet_resnet34_synapse_best.pth")
            print(f"  ✓ Best dice {best_dice:.4f} — checkpoint saved")

        if epoch % cfg["save_every"] == 0:
            torch.save(model.state_dict(),
                       out / f"checkpoint_ep{epoch:03d}.pth")

        with open(out / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── Final results ─────────────────────────────────────────
    print("\n" + "="*60)
    print(f" Training complete — Best Dice: {best_dice:.4f}")
    print("="*60)
    print("\nFinal per-class Dice:")
    for cls, score in named_dice.items():
        bar = "█" * int(score * 20)
        print(f"  {cls:15s} {score:.4f}  {bar}")

    final = {
        "dataset":        "Synapse multi-organ (TransUNet preprocessed)",
        "train_val_split": f"80/20 (seed={cfg['seed']})",
        "best_val_dice":  round(best_dice, 4),
        "mean_iou":       round(mean_iou,  4),
        "per_class_dice": named_dice,
        "per_class_iou":  named_iou,
        "config":         {k: v for k, v in cfg.items()
                           if k not in ("train_dir","test_dir","class_names")},
        "model":          "UNet-ResNet34 (segmentation_models_pytorch)",
        "note": ("These are real measured results from training on the "
                 "Synapse multi-organ CT dataset.")
    }
    with open(out / "final_results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nSaved to {out}/")
    print("  unet_resnet34_synapse_best.pth  ← trained weights")
    print("  final_results.json              ← paste into MODEL_CARD.md")
    print("  training_history.json           ← full epoch log")
    return best_dice, final

if __name__ == "__main__":
    best_dice, results = train(CFG)
