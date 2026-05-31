# ============================================================
# MediVision v2 — Synapse Multi-Organ Segmentation
# Fix: consistent normalization across train/test splits
# ============================================================

import subprocess, sys
def pip(pkg): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
pip("segmentation-models-pytorch")
pip("h5py")

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
from PIL import Image as PILImage
from torch.amp import autocast, GradScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
print(f"GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

CFG = {
    "train_dir":  "/kaggle/input/datasets/nkapadia001/btcv-synapse-preprocessed/project_TransUNet/data/Synapse/train_npz",
    "test_dir":   "/kaggle/input/datasets/nkapadia001/btcv-synapse-preprocessed/project_TransUNet/data/Synapse/test_vol_h5",
    "output_dir": "/kaggle/working/medivision_output",
    "img_size":    224,
    "num_classes": 9,
    "batch_size":  24,
    "epochs":      150,
    "lr":          1e-4,
    "weight_decay":1e-5,
    "val_split":   0.2,
    "seed":        42,
    "save_every":  10,
    "early_stopping_patience": 20,
    "class_names": [
        "background","aorta","gallbladder","spleen",
        "left_kidney","right_kidney","liver","stomach","pancreas"
    ]
}

Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])
torch.backends.cudnn.benchmark = True


# ── Key fix: shared normalization function ────────────────────
def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize a single CT slice to [0, 1] using its own min/max.
    This works for BOTH .npz (already float) and .h5 (raw HU values)
    because it normalizes each slice independently.
    """
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mn) / (mx - mn)).astype(np.float32)


def to_tensor(img: np.ndarray, size: int) -> torch.Tensor:
    """Resize + replicate to 3-channel tensor."""
    if img.shape[0] != size or img.shape[1] != size:
        img = np.array(
            PILImage.fromarray(img).resize((size, size), PILImage.BILINEAR)
        )
    return torch.from_numpy(np.stack([img, img, img], axis=0)).float()


def mask_to_tensor(mask: np.ndarray, size: int) -> torch.Tensor:
    if mask.shape[0] != size or mask.shape[1] != size:
        mask = np.array(
            PILImage.fromarray(mask.astype(np.uint8)).resize(
                (size, size), PILImage.NEAREST)
        )
    return torch.tensor(mask, dtype=torch.long)


# ── Datasets ─────────────────────────────────────────────────
class SynapseTrainDataset(Dataset):
    def __init__(self, npz_paths, img_size=224, augment=False):
        self.paths   = npz_paths
        self.size    = img_size
        self.augment = augment
        print(f"  {'Train' if augment else 'Val'} slices: {len(self.paths)}")

    def _augment(self, img, lbl):
        if random.random() > 0.5:
            img, lbl = np.fliplr(img).copy(), np.fliplr(lbl).copy()
        if random.random() > 0.5:
            img, lbl = np.flipud(img).copy(), np.flipud(lbl).copy()
        k = random.randint(0, 3)
        img, lbl = np.rot90(img, k).copy(), np.rot90(lbl, k).copy()
        return img, lbl

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        data  = np.load(self.paths[idx])
        img   = normalize(data["image"])          # ← normalize here
        label = data["label"].astype(np.int64)
        if self.augment:
            img, label = self._augment(img, label)
        return to_tensor(img, self.size), mask_to_tensor(label, self.size)


class SynapseTestDataset(Dataset):
    """
    Loads .npy.h5 volumes — each contains full 3D CT volume.
    Applies the SAME normalize() function as training.
    """
    def __init__(self, h5_paths, img_size=224):
        self.slices = []
        for p in h5_paths:
            with h5py.File(p, "r") as f:
                img_vol = f["image"][:]   # (S, H, W) — raw values
                lbl_vol = f["label"][:]   # (S, H, W)
            for s in range(img_vol.shape[0]):
                self.slices.append((img_vol[s], lbl_vol[s]))
        self.size = img_size
        print(f"  Test slices: {len(self.slices)}")

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        img, label = self.slices[idx]
        img   = normalize(img)             # ← same normalize as train
        label = label.astype(np.int64)
        return to_tensor(img, self.size), mask_to_tensor(label, self.size)


def build_dataloaders(cfg):
    all_train = sorted(Path(cfg["train_dir"]).glob("*.npz"))
    all_test  = sorted(Path(cfg["test_dir"]).glob("*.npy.h5"))
    assert len(all_train) > 0, f"No .npz in {cfg['train_dir']}"
    assert len(all_test)  > 0, f"No .npy.h5 in {cfg['test_dir']}"
    print(f"Found {len(all_train)} train cases, {len(all_test)} test volumes")

    train_cases, val_cases = train_test_split(
        all_train, test_size=cfg["val_split"], random_state=cfg["seed"]
    )
    train_ds = SynapseTrainDataset(train_cases, cfg["img_size"], augment=True)
    val_ds   = SynapseTrainDataset(val_cases,   cfg["img_size"], augment=False)
    test_ds  = SynapseTestDataset(all_test,     cfg["img_size"])

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ── Model ─────────────────────────────────────────────────────
def build_model(num_classes):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(DEVICE)


# ── Loss ──────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, C, smooth=1e-6):
        super().__init__()
        self.C, self.smooth = C, smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss  = 0.0
        for c in range(1, self.C):
            p = probs[:, c]
            t = (targets == c).float()
            loss += 1 - (2*(p*t).sum() + self.smooth) / (p.sum()+t.sum()+self.smooth)
        return loss / (self.C - 1)

class CombinedLoss(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dice = DiceLoss(C)
        self.ce   = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        return 0.5*self.dice(logits, targets) + 0.5*self.ce(logits, targets)


# ── Metrics ───────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(preds, targets, C, smooth=1e-6):
    dice, iou = {}, {}
    for c in range(1, C):
        p = (preds==c).float(); t = (targets==c).float()
        if t.sum()==0 and p.sum()==0: continue
        inter = (p*t).sum()
        union = p.sum()+t.sum()-inter
        dice[c] = float((2*inter+smooth)/(p.sum()+t.sum()+smooth))
        iou[c]  = float((inter+smooth)/(union+smooth))
    return dice, iou


# ── Train / eval loops ────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train(); total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            loss = criterion(model(imgs), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, criterion, C, class_names):
    model.eval()
    total_loss, all_p, all_t = 0.0, [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        all_p.append(logits.argmax(1).cpu())
        all_t.append(masks.cpu())
    all_p = torch.cat(all_p); all_t = torch.cat(all_t)
    dice, iou = compute_metrics(all_p, all_t, C)
    mean_dice = float(np.mean(list(dice.values()))) if dice else 0.0
    mean_iou  = float(np.mean(list(iou.values())))  if iou  else 0.0
    named_dice = {class_names[c]: round(v,4) for c,v in dice.items()}
    named_iou  = {class_names[c]: round(v,4) for c,v in iou.items()}
    return total_loss/len(loader), mean_dice, mean_iou, named_dice, named_iou


# ── Main ──────────────────────────────────────────────────────
def train(cfg):
    print("\n" + "="*60)
    print(" MediVision v2 — Synapse Multi-Organ Segmentation")
    print("="*60 + "\n")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model     = build_model(cfg["num_classes"])
    criterion = CombinedLoss(cfg["num_classes"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5,
                    patience=5, min_lr=1e-6)
    scaler    = GradScaler("cuda")

    best_dice, no_improve, history = -1.0, 0, []
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs"]+1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, mean_dice, mean_iou, named_dice, named_iou = eval_epoch(
            model, val_loader, criterion, cfg["num_classes"], cfg["class_names"])
        scheduler.step(mean_dice)
        elapsed = time.time()-t0

        record = {"epoch":epoch,"train_loss":round(train_loss,4),
                  "val_loss":round(val_loss,4),"mean_dice":round(mean_dice,4),
                  "mean_iou":round(mean_iou,4),"per_class_dice":named_dice,
                  "per_class_iou":named_iou,"lr":optimizer.param_groups[0]["lr"],
                  "elapsed_s":round(elapsed,1)}
        history.append(record)

        print(f"Ep {epoch:3d}/{cfg['epochs']} | train={train_loss:.4f} | "
              f"val={val_loss:.4f} | dice={mean_dice:.4f} | "
              f"iou={mean_iou:.4f} | lr={record['lr']:.1e} | {elapsed:.0f}s")

        if mean_dice > best_dice:
            best_dice = mean_dice; no_improve = 0
            torch.save(model.state_dict(), out/"unet_resnet34_synapse_best.pth")
            print(f"  ✓ Best val dice {best_dice:.4f} — saved")
        else:
            no_improve += 1

        if epoch % cfg["save_every"] == 0:
            torch.save(model.state_dict(), out/f"checkpoint_ep{epoch:03d}.pth")

        with open(out/"training_history.json","w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch}"); break

    # ── Official test evaluation ──────────────────────────────
    print("\n" + "="*60)
    print(" Evaluating on official test set (test_vol_h5)")
    print("="*60)
    model.load_state_dict(torch.load(out/"unet_resnet34_synapse_best.pth",
                                     map_location=DEVICE))
    _, test_dice, test_iou, test_named_dice, test_named_iou = eval_epoch(
        model, test_loader, criterion,
        cfg["num_classes"], cfg["class_names"])

    print(f"\nTest Dice : {test_dice:.4f}")
    print(f"Test IoU  : {test_iou:.4f}")
    print("\nPer-class Dice:")
    for cls, score in test_named_dice.items():
        bar = "█" * int(score * 20)
        print(f"  {cls:15s} {score:.4f}  {bar}")

    final = {
        "dataset":           "Synapse multi-organ CT (TransUNet preprocessed)",
        "train_val_split":   f"80/20 of train_npz (seed={cfg['seed']})",
        "best_val_dice":     round(best_dice, 4),
        "test_dice":         round(test_dice, 4),
        "test_iou":          round(test_iou,  4),
        "per_class_dice":    test_named_dice,
        "per_class_iou":     test_named_iou,
        "model":             "UNet-ResNet34 (segmentation_models_pytorch)",
        "normalization":     "per-slice min-max [0,1] applied to both train and test",
        "note":              "Real measured results — evaluated on official test_vol_h5 set"
    }
    with open(out/"final_results.json","w") as f:
        json.dump(final, f, indent=2)

    print(f"\nAll outputs saved to {out}/")
    print("  unet_resnet34_synapse_best.pth  ← trained weights")
    print("  final_results.json              ← copy into MODEL_CARD.md")
    return best_dice, final

if __name__ == "__main__":
    best_dice, results = train(CFG)
