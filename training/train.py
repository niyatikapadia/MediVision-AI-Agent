# MediVision — BTCV Multi-Organ Segmentation Training
# Run on Kaggle P100/T4 — full pipeline, ~4-6 hours
# Dataset: BTCV (Synapse:syn3193805)

# ── 0. Install deps ──────────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("nibabel")       # NIfTI file loading
install("albumentations")
install("segmentation-models-pytorch")  # UNet-ResNet34 pretrained

# ── 1. Imports ───────────────────────────────────────────────────────────────
import os, json, time, random
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── 2. Config ─────────────────────────────────────────────────────────────────
CFG = {
    "data_root":    "/kaggle/input/btcv-abdomen/RawData",
    "output_dir":   "/kaggle/working/medivision_output",
    "img_size":     512,
    "num_classes":  14,       # BTCV has 13 organs + background
    "batch_size":   8,
    "epochs":       50,
    "lr":           1e-4,
    "weight_decay": 1e-5,
    "val_split":    0.2,
    "seed":         42,
    "save_every":   5,        # save checkpoint every N epochs
    # BTCV organ mapping
    "class_names": [
        "background","spleen","right_kidney","left_kidney","gallbladder",
        "esophagus","liver","stomach","aorta","ivc",
        "portal_vein","pancreas","right_adrenal","left_adrenal"
    ]
}

Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ── 3. Dataset ────────────────────────────────────────────────────────────────
class BTCVSliceDataset(Dataset):
    """
    Loads BTCV NIfTI volumes, extracts 2D axial slices on the fly.
    Applies HU windowing [-175, 250] standard for abdominal CT.
    """

    HU_MIN, HU_MAX = -175, 250

    def __init__(self, image_paths, mask_paths, transform=None):
        self.slices = []  # list of (img_path, mask_path, slice_idx)
        for img_path, msk_path in zip(image_paths, mask_paths):
            vol = nib.load(img_path)
            n_slices = vol.shape[2]
            for s in range(n_slices):
                self.slices.append((img_path, msk_path, s))
        self.transform = transform
        print(f"  Dataset: {len(self.slices)} slices from {len(image_paths)} volumes")

    def _load_slice(self, img_path, msk_path, idx):
        img_vol = nib.load(img_path).get_fdata()
        msk_vol = nib.load(msk_path).get_fdata().astype(np.int64)
        img_slice = img_vol[:, :, idx].astype(np.float32)
        msk_slice = msk_vol[:, :, idx]
        # HU windowing → [0, 1]
        img_slice = np.clip(img_slice, self.HU_MIN, self.HU_MAX)
        img_slice = (img_slice - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)
        return img_slice, msk_slice

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, msk_path, s = self.slices[idx]
        img, msk = self._load_slice(img_path, msk_path, s)
        # Convert to 3-channel (expected by ResNet encoder)
        img_rgb = np.stack([img, img, img], axis=-1)
        if self.transform:
            aug = self.transform(image=img_rgb, mask=msk)
            img_rgb, msk = aug["image"], aug["mask"]
        return img_rgb, torch.tensor(msk, dtype=torch.long)


def get_transforms(is_train: bool, img_size: int):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            ToTensorV2(),
        ])


def build_dataloaders(cfg):
    img_dir  = Path(cfg["data_root"]) / "Training" / "img"
    mask_dir = Path(cfg["data_root"]) / "Training" / "label"

    img_paths  = sorted(img_dir.glob("*.nii.gz"))
    mask_paths = sorted(mask_dir.glob("*.nii.gz"))

    assert len(img_paths) > 0, f"No .nii.gz files found in {img_dir}"
    assert len(img_paths) == len(mask_paths), "Image/mask count mismatch"
    print(f"Found {len(img_paths)} CT volumes")

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        img_paths, mask_paths,
        test_size=cfg["val_split"],
        random_state=cfg["seed"]
    )

    train_ds = BTCVSliceDataset(train_imgs, train_masks,
                                transform=get_transforms(True, cfg["img_size"]))
    val_ds   = BTCVSliceDataset(val_imgs,   val_masks,
                                transform=get_transforms(False, cfg["img_size"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ── 4. Model ──────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    """
    UNet with ResNet-34 encoder from segmentation_models_pytorch.
    ImageNet pretrained encoder, fine-tuned end-to-end.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model.to(DEVICE)


# ── 5. Loss ───────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(self.num_classes):
            pred_c = probs[:, c]
            tgt_c  = (targets == c).float()
            intersection = (pred_c * tgt_c).sum()
            loss += 1 - (2 * intersection + self.smooth) / \
                        (pred_c.sum() + tgt_c.sum() + self.smooth)
        return loss / self.num_classes


class CombinedLoss(nn.Module):
    """0.5 × Dice + 0.5 × CrossEntropy"""
    def __init__(self, num_classes):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + \
               0.5 * self.ce(logits, targets)


# ── 6. Metrics ────────────────────────────────────────────────────────────────
def compute_dice_per_class(preds, targets, num_classes, smooth=1e-6):
    """Returns per-class Dice scores as a list."""
    dice_scores = []
    for c in range(1, num_classes):  # skip background
        pred_c = (preds == c).float()
        tgt_c  = (targets == c).float()
        if tgt_c.sum() == 0 and pred_c.sum() == 0:
            continue  # skip empty classes
        intersection = (pred_c * tgt_c).sum()
        score = (2 * intersection + smooth) / \
                (pred_c.sum() + tgt_c.sum() + smooth)
        dice_scores.append((c, float(score)))
    return dice_scores


# ── 7. Train / Val loops ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    per_class   = compute_dice_per_class(all_preds, all_targets, num_classes)
    mean_dice   = np.mean([s for _, s in per_class]) if per_class else 0.0
    return total_loss / len(loader), mean_dice, per_class


# ── 8. Main training loop ─────────────────────────────────────────────────────
def train(cfg):
    print("\n" + "="*60)
    print("MediVision — BTCV Multi-Organ Segmentation Training")
    print("="*60)

    train_loader, val_loader = build_dataloaders(cfg)
    model     = build_model(cfg["num_classes"])
    criterion = CombinedLoss(cfg["num_classes"])
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True)

    best_dice  = 0.0
    history    = []
    output_dir = Path(cfg["output_dir"])

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, mean_dice, per_class = val_epoch(
            model, val_loader, criterion, cfg["num_classes"])
        scheduler.step(mean_dice)
        elapsed = time.time() - t0

        # Per-class results
        class_results = {
            cfg["class_names"][c]: round(score, 4)
            for c, score in per_class
        }

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "mean_dice":  round(mean_dice, 4),
            "per_class":  class_results,
            "lr":         optimizer.param_groups[0]["lr"],
            "elapsed_s":  round(elapsed, 1),
        }
        history.append(record)

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"mean_dice={mean_dice:.4f} | lr={record['lr']:.2e} | "
              f"time={elapsed:.0f}s")

        # Save best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(),
                       output_dir / "unet_resnet34_btcv_best.pth")
            print(f"  ✓ New best Dice: {best_dice:.4f} — checkpoint saved")

        # Save periodic checkpoint
        if epoch % cfg["save_every"] == 0:
            torch.save(model.state_dict(),
                       output_dir / f"checkpoint_epoch{epoch}.pth")

        # Save running history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Training complete. Best validation Dice: {best_dice:.4f}")
    print("="*60)
    print("\nFinal per-class Dice (last epoch):")
    for cls_name, score in class_results.items():
        print(f"  {cls_name:25s}: {score:.4f}")

    # Save final results JSON — this is what goes into MODEL_CARD.md
    final_results = {
        "dataset": "BTCV (Synapse:syn3193805)",
        "split": f"80% train / 20% val (seed={cfg['seed']})",
        "best_val_dice": best_dice,
        "final_epoch_dice": mean_dice,
        "per_class_dice_final_epoch": class_results,
        "config": cfg,
        "total_epochs": cfg["epochs"],
    }
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - unet_resnet34_btcv_best.pth   ← upload this to GitHub repo")
    print(f"  - final_results.json            ← copy numbers into MODEL_CARD.md")
    print(f"  - training_history.json         ← full epoch log")

    return best_dice, final_results


if __name__ == "__main__":
    best_dice, results = train(CFG)
