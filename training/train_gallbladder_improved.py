# ============================================================
# MediVision — Gallbladder & Liver Improvement Training
# 
# Targets the two weakest classes from v2:
#   Gallbladder: 0.585 → target 0.70+
#   Liver:       0.543 → target 0.75+
#
# Three improvements over v2:
#   1. Class-weighted loss (gallbladder/liver weighted 3x)
#   2. Focal Tversky loss (penalizes false negatives harder)
#   3. Test-time augmentation (TTA) for final evaluation
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
print(f"Device: {DEVICE} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

CFG = {
    "train_dir":  "/kaggle/input/datasets/nkapadia001/btcv-synapse-preprocessed/project_TransUNet/data/Synapse/train_npz",
    "test_dir":   "/kaggle/input/datasets/nkapadia001/btcv-synapse-preprocessed/project_TransUNet/data/Synapse/test_vol_h5",
    "output_dir": "/kaggle/working/medivision_v3_output",
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

    # ── Improvement 1: class weights ─────────────────────────
    # Gallbladder=class 2, Liver=class 6 weighted 3x
    # All others weighted 1x, background 0.1x
    "class_weights": [0.1, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.5],

    # ── Improvement 2: Focal Tversky params ──────────────────
    "tversky_alpha": 0.7,   # weight false negatives more (recall focus)
    "tversky_beta":  0.3,   # weight false positives less
    "focal_gamma":   0.75,  # focal exponent

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


# ── Normalization (same as v2) ────────────────────────────────
def normalize(img: np.ndarray) -> np.ndarray:
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mn) / (mx - mn)).astype(np.float32)

def to_tensor(img, size):
    if img.shape[0] != size or img.shape[1] != size:
        img = np.array(PILImage.fromarray(img).resize((size,size), PILImage.BILINEAR))
    return torch.from_numpy(np.stack([img,img,img], axis=0)).float()

def mask_to_tensor(mask, size):
    if mask.shape[0] != size or mask.shape[1] != size:
        mask = np.array(PILImage.fromarray(mask.astype(np.uint8)).resize((size,size), PILImage.NEAREST))
    return torch.tensor(mask, dtype=torch.long)


# ── Datasets ─────────────────────────────────────────────────
class SynapseTrainDataset(Dataset):
    def __init__(self, npz_paths, img_size=224, augment=False):
        self.paths = npz_paths; self.size = img_size; self.augment = augment
        print(f"  {'Train' if augment else 'Val'} slices: {len(self.paths)}")

    def _augment(self, img, lbl):
        if random.random() > 0.5: img,lbl = np.fliplr(img).copy(),np.fliplr(lbl).copy()
        if random.random() > 0.5: img,lbl = np.flipud(img).copy(),np.flipud(lbl).copy()
        k = random.randint(0,3)
        img,lbl = np.rot90(img,k).copy(),np.rot90(lbl,k).copy()
        # ── Improvement: elastic-like random scale crop ───────
        if random.random() > 0.5:
            scale = random.uniform(0.85, 1.0)
            h,w   = img.shape
            nh,nw = int(h*scale), int(w*scale)
            y0    = random.randint(0, h-nh)
            x0    = random.randint(0, w-nw)
            img   = img[y0:y0+nh, x0:x0+nw]
            lbl   = lbl[y0:y0+nh, x0:x0+nw]
        return img, lbl

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        data  = np.load(self.paths[idx])
        img   = normalize(data["image"])
        label = data["label"].astype(np.int64)
        if self.augment: img,label = self._augment(img,label)
        return to_tensor(img,self.size), mask_to_tensor(label,self.size)


class SynapseTestDataset(Dataset):
    def __init__(self, h5_paths, img_size=224):
        self.slices = []
        for p in h5_paths:
            with h5py.File(p,"r") as f:
                iv,lv = f["image"][:], f["label"][:]
            for s in range(iv.shape[0]):
                self.slices.append((iv[s], lv[s]))
        self.size = img_size
        print(f"  Test slices: {len(self.slices)}")

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        img,label = self.slices[idx]
        return to_tensor(normalize(img.astype(np.float32)),self.size), \
               mask_to_tensor(label.astype(np.int64),self.size)


def build_dataloaders(cfg):
    all_train = sorted(Path(cfg["train_dir"]).glob("*.npz"))
    all_test  = sorted(Path(cfg["test_dir"]).glob("*.npy.h5"))
    assert len(all_train)>0 and len(all_test)>0
    print(f"Found {len(all_train)} train, {len(all_test)} test")
    tr,va = train_test_split(all_train,test_size=cfg["val_split"],random_state=cfg["seed"])
    kw = dict(num_workers=4,pin_memory=True,persistent_workers=True)
    return (DataLoader(SynapseTrainDataset(tr,cfg["img_size"],True), cfg["batch_size"],True,**kw),
            DataLoader(SynapseTrainDataset(va,cfg["img_size"],False),cfg["batch_size"],False,**kw),
            DataLoader(SynapseTestDataset(all_test,cfg["img_size"]),  cfg["batch_size"],False,**kw))


# ── Model ─────────────────────────────────────────────────────
def build_model(n):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n,
        # ── Improvement 3: add attention gates in decoder ─────
        decoder_attention_type="scse",  # squeeze-excite attention
    ).to(DEVICE)


# ── Improvement 2: Focal Tversky Loss ────────────────────────
class FocalTverskyLoss(nn.Module):
    """
    Tversky loss with focal exponent.
    alpha > beta → penalizes false negatives more → better recall on small organs.
    Focal exponent γ < 1 → focuses training on easy examples less, hard more.

    Paper: Abraham & Khan (2019) — A Novel Focal Tversky Loss Function
    """
    def __init__(self, C, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.C=C; self.alpha=alpha; self.beta=beta
        self.gamma=gamma; self.smooth=smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss  = 0.0
        for c in range(1, self.C):
            p  = probs[:,c]
            t  = (targets==c).float()
            tp = (p*t).sum()
            fp = (p*(1-t)).sum()
            fn = ((1-p)*t).sum()
            tversky = (tp+self.smooth)/(tp+self.alpha*fn+self.beta*fp+self.smooth)
            loss += (1-tversky)**self.gamma
        return loss/(self.C-1)


class ImprovedLoss(nn.Module):
    """
    0.4 × Focal Tversky  +  0.3 × Dice  +  0.3 × Weighted CE
    """
    def __init__(self, cfg):
        super().__init__()
        self.ft  = FocalTverskyLoss(cfg["num_classes"],
                                    cfg["tversky_alpha"],
                                    cfg["tversky_beta"],
                                    cfg["focal_gamma"])
        # standard dice
        from torch import nn as _nn
        self.ce_weights = torch.tensor(cfg["class_weights"],
                                       dtype=torch.float32, device=DEVICE)
        self.C = cfg["num_classes"]

    def _dice(self, logits, targets, smooth=1e-6):
        probs = torch.softmax(logits,1); loss=0.0
        for c in range(1,self.C):
            p=probs[:,c]; t=(targets==c).float()
            loss+=1-(2*(p*t).sum()+smooth)/(p.sum()+t.sum()+smooth)
        return loss/(self.C-1)

    def forward(self, logits, targets):
        ce  = nn.functional.cross_entropy(logits, targets, weight=self.ce_weights)
        return 0.4*self.ft(logits,targets) + 0.3*self._dice(logits,targets) + 0.3*ce


# ── Metrics ───────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(preds, targets, C, smooth=1e-6):
    dice,iou={},{}
    for c in range(1,C):
        p=(preds==c).float(); t=(targets==c).float()
        if t.sum()==0 and p.sum()==0: continue
        inter=(p*t).sum(); union=p.sum()+t.sum()-inter
        dice[c]=float((2*inter+smooth)/(p.sum()+t.sum()+smooth))
        iou[c] =float((inter+smooth)/(union+smooth))
    return dice,iou


# ── Improvement 4: Test-Time Augmentation ────────────────────
@torch.no_grad()
def tta_predict(model, imgs):
    """
    Average predictions over 4 augmentations:
    original, hflip, vflip, hflip+vflip
    """
    preds = []
    for hf in [False, True]:
        for vf in [False, True]:
            x = imgs.clone()
            if hf: x = torch.flip(x, [3])
            if vf: x = torch.flip(x, [2])
            logits = model(x)
            if hf: logits = torch.flip(logits, [3])
            if vf: logits = torch.flip(logits, [2])
            preds.append(torch.softmax(logits, dim=1))
    return torch.stack(preds).mean(0).argmax(1)


# ── Train/eval ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train(); total=0.0
    for imgs,masks in loader:
        imgs,masks=imgs.to(DEVICE),masks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            loss=criterion(model(imgs),masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total+=loss.item()
    return total/len(loader)

@torch.no_grad()
def eval_epoch(model, loader, criterion, C, class_names, use_tta=False):
    model.eval()
    total_loss,all_p,all_t=0.0,[],[]
    for imgs,masks in loader:
        imgs,masks=imgs.to(DEVICE),masks.to(DEVICE)
        logits=model(imgs)
        total_loss+=criterion(logits,masks).item()
        preds = tta_predict(model,imgs) if use_tta else logits.argmax(1)
        all_p.append(preds.cpu()); all_t.append(masks.cpu())
    all_p=torch.cat(all_p); all_t=torch.cat(all_t)
    dice,iou=compute_metrics(all_p,all_t,C)
    mean_dice=float(np.mean(list(dice.values()))) if dice else 0.0
    mean_iou =float(np.mean(list(iou.values())))  if iou  else 0.0
    named_dice={class_names[c]:round(v,4) for c,v in dice.items()}
    named_iou ={class_names[c]:round(v,4) for c,v in iou.items()}
    return total_loss/len(loader),mean_dice,mean_iou,named_dice,named_iou


# ── Main ──────────────────────────────────────────────────────
def train(cfg):
    print("\n"+"="*60)
    print(" MediVision v3 — Gallbladder & Liver Improvement")
    print("="*60+"\n")
    print("Key improvements over v2:")
    print("  1. Class-weighted loss (gallbladder ×3, liver ×3)")
    print("  2. Focal Tversky loss (α=0.7 — recall focus)")
    print("  3. Squeeze-Excite attention in decoder")
    print("  4. Test-time augmentation (TTA) at evaluation")
    print("  5. Random scale-crop augmentation\n")

    tr_loader,va_loader,te_loader = build_dataloaders(cfg)
    model     = build_model(cfg["num_classes"])
    criterion = ImprovedLoss(cfg)
    optimizer = optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,"max",factor=0.5,patience=5,min_lr=1e-6)
    scaler    = GradScaler("cuda")

    best_dice,no_improve,history=-1.0,0,[]
    out=Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)

    for epoch in range(1,cfg["epochs"]+1):
        t0=time.time()
        tr_loss=train_epoch(model,tr_loader,optimizer,criterion,scaler)
        va_loss,mean_dice,mean_iou,named_dice,named_iou=eval_epoch(
            model,va_loader,criterion,cfg["num_classes"],cfg["class_names"])
        scheduler.step(mean_dice); elapsed=time.time()-t0

        record={"epoch":epoch,"train_loss":round(tr_loss,4),"val_loss":round(va_loss,4),
                "mean_dice":round(mean_dice,4),"mean_iou":round(mean_iou,4),
                "per_class_dice":named_dice,"lr":optimizer.param_groups[0]["lr"],
                "elapsed_s":round(elapsed,1)}
        history.append(record)

        print(f"Ep {epoch:3d}/{cfg['epochs']} | train={tr_loss:.4f} | val={va_loss:.4f} | "
              f"dice={mean_dice:.4f} | gb={named_dice.get('gallbladder',0):.4f} | "
              f"liver={named_dice.get('liver',0):.4f} | {elapsed:.0f}s")

        if mean_dice>best_dice:
            best_dice=mean_dice; no_improve=0
            torch.save(model.state_dict(), out/"unet_resnet34_v3_best.pth")
            print(f"  ✓ Best dice {best_dice:.4f}")
        else:
            no_improve+=1

        if epoch%cfg["save_every"]==0:
            torch.save(model.state_dict(), out/f"checkpoint_ep{epoch:03d}.pth")
        with open(out/"training_history_v3.json","w") as f:
            json.dump(history,f,indent=2)
        if no_improve>=cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch}"); break

    # ── Official test evaluation WITH TTA ─────────────────────
    print("\n"+"="*60)
    print(" Test evaluation WITH Test-Time Augmentation")
    print("="*60)
    model.load_state_dict(torch.load(out/"unet_resnet34_v3_best.pth",
                                     map_location=DEVICE, weights_only=True))

    # Without TTA
    _,td_no_tta,ti_no_tta,nd_no_tta,ni_no_tta = eval_epoch(
        model,te_loader,criterion,cfg["num_classes"],cfg["class_names"],use_tta=False)
    # With TTA
    _,td_tta,ti_tta,nd_tta,ni_tta = eval_epoch(
        model,te_loader,criterion,cfg["num_classes"],cfg["class_names"],use_tta=True)

    print(f"\n{'Metric':<25} {'No TTA':>10} {'With TTA':>10} {'v2 baseline':>12}")
    print("-"*60)
    v2 = {"aorta":0.8483,"gallbladder":0.5850,"spleen":0.8911,"left_kidney":0.8415,
          "right_kidney":0.9433,"liver":0.5425,"stomach":0.8197,"pancreas":0.7367}
    for cls in cfg["class_names"][1:]:
        v2s  = v2.get(cls,0)
        ntta = nd_no_tta.get(cls,0)
        wtta = nd_tta.get(cls,0)
        diff = wtta - v2s
        sign = "▲" if diff>0.01 else ("▼" if diff<-0.01 else "~")
        print(f"  {cls:<23} {ntta:>10.4f} {wtta:>10.4f} {v2s:>10.4f}  {sign}{abs(diff):.3f}")
    print(f"\n  {'Mean Dice':<23} {td_no_tta:>10.4f} {td_tta:>10.4f} {'0.7760':>10}  "
          f"{'▲' if td_tta>0.776 else '▼'}{abs(td_tta-0.776):.3f}")

    final={
        "dataset":        "Synapse multi-organ CT (TransUNet preprocessed)",
        "model":          "UNet-ResNet34 + SCSE attention (v3)",
        "improvements":   ["class-weighted loss","focal tversky","SCSE attention","TTA","scale-crop aug"],
        "test_dice_no_tta":  round(td_no_tta,4),
        "test_dice_with_tta":round(td_tta,4),
        "test_iou_with_tta": round(ti_tta,4),
        "per_class_dice_tta":nd_tta,
        "v2_baseline_dice":  0.7760,
        "improvement_over_v2": round(td_tta-0.7760,4),
        "note": "Real measured results — v3 with gallbladder/liver improvements"
    }
    with open(out/"final_results_v3.json","w") as f:
        json.dump(final,f,indent=2)
    print(f"\nSaved to {out}/final_results_v3.json")
    return best_dice, final

if __name__=="__main__":
    best_dice, results = train(CFG)
