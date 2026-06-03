# ================================================================
# MediVision v3 -> v3.1 Fine-tuning
#
# Purpose: fix liver/right_kidney boundary confusion
# Strategy: start from v3 checkpoint, fine-tune 50 epochs with
#   - higher liver weight (2.0 -> 4.0)
#   - lower right_kidney weight (1.5 -> 0.8)
#   - lower LR (1e-5) — fine-tuning, not relearning
#   - same augmentation and loss as v3
#
# Expected: liver Dice 0.71 -> 0.78+, right_kidney stays 0.94+
# Runtime: ~75 minutes on Kaggle P100
# ================================================================

import subprocess, sys
def pip(pkg): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
pip("segmentation-models-pytorch")
pip("h5py")

import json, time, random
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
    "train_dir":   "/kaggle/input/datasets/nkapadia001/btcv-synapse-preprocessed/project_TransUNet/data/Synapse/train_npz",
    "test_dir":    "/kaggle/input/datasets/nkapadia001/btcv-synapse-preprocessed/project_TransUNet/data/Synapse/test_vol_h5",
    "checkpoint":  "/kaggle/input/YOUR_V3_DATASET/unet_resnet34_v3_best.pth",  # <-- update this path
    "output_dir":  "/kaggle/working/medivision_v31",

    "img_size":    512,
    "batch_size":  8,
    "num_classes": 9,
    "epochs":      50,       # fine-tuning only
    "lr":          1e-5,     # 10x lower than v3 — preserve learned features
    "weight_decay":1e-4,
    "val_split":   0.2,
    "seed":        42,
    "early_stopping_patience": 15,

    # Key change: liver weight 2.0->4.0, right_kidney 1.5->0.8
    # This directly addresses the liver/kidney boundary confusion
    "class_weights": [0.05, 1.5, 2.5, 1.0, 1.5, 0.8, 4.0, 1.2, 3.0],
    #                  bg    ao   gb   sp   lk   rk   li   st   pa

    "tversky_alpha": 0.7,   # even more recall-focused for liver
    "tversky_beta":  0.3,
    "focal_gamma":   0.75,

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


def normalize(img):
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8: return np.zeros_like(img, dtype=np.float32)
    return ((img - mn) / (mx - mn)).astype(np.float32)

def to_tensor(img, size):
    if img.shape[0] != size or img.shape[1] != size:
        img = np.array(PILImage.fromarray(img).resize((size, size), PILImage.BILINEAR))
    return torch.from_numpy(np.stack([img, img, img], axis=0)).float()

def mask_to_tensor(mask, size):
    if mask.shape[0] != size or mask.shape[1] != size:
        mask = np.array(PILImage.fromarray(mask.astype(np.uint8)).resize((size, size), PILImage.NEAREST))
    return torch.tensor(mask, dtype=torch.long)


def augment(img, lbl):
    if random.random() > 0.5: img,lbl = np.fliplr(img).copy(),np.fliplr(lbl).copy()
    if random.random() > 0.5: img,lbl = np.flipud(img).copy(),np.flipud(lbl).copy()
    k = random.randint(0,3)
    img,lbl = np.rot90(img,k).copy(),np.rot90(lbl,k).copy()
    if random.random() > 0.4:
        scale = random.uniform(0.8, 1.0)
        h,w   = img.shape
        nh,nw = int(h*scale), int(w*scale)
        y0,x0 = random.randint(0,h-nh), random.randint(0,w-nw)
        img,lbl = img[y0:y0+nh,x0:x0+nw], lbl[y0:y0+nh,x0:x0+nw]
    if random.random() > 0.5:
        gamma = random.uniform(0.7, 1.5)
        img   = np.clip(img, 1e-8, 1.0)
        img   = np.power(img, gamma)
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(0.01, 0.04), img.shape)
        img   = np.clip(img + noise, 0, 1).astype(np.float32)
    return img, lbl


class SynapseTrainDataset(Dataset):
    def __init__(self, paths, size=512, aug=False):
        self.paths,self.size,self.aug = paths,size,aug
        print(f"  Dataset: {len(self.paths)} slices")
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        d = np.load(self.paths[idx])
        img,lbl = normalize(d["image"]), d["label"].astype(np.int64)
        if self.aug: img,lbl = augment(img,lbl)
        return to_tensor(img,self.size), mask_to_tensor(lbl,self.size)

class SynapseTestDataset(Dataset):
    def __init__(self, h5_paths, size=512):
        self.slices,self.size = [],size
        for p in h5_paths:
            with h5py.File(p,"r") as f:
                iv,lv = f["image"][:],f["label"][:]
            for s in range(iv.shape[0]):
                self.slices.append((iv[s],lv[s]))
        print(f"  Test slices: {len(self.slices)}")
    def __len__(self): return len(self.slices)
    def __getitem__(self, idx):
        img,lbl = self.slices[idx]
        return to_tensor(normalize(img.astype(np.float32)),self.size), mask_to_tensor(lbl.astype(np.int64),self.size)

def build_dataloaders(cfg):
    all_train = sorted(Path(cfg["train_dir"]).glob("*.npz"))
    all_test  = sorted(Path(cfg["test_dir"]).glob("*.npy.h5"))
    tr,va = train_test_split(all_train, test_size=cfg["val_split"], random_state=cfg["seed"])
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    return (
        DataLoader(SynapseTrainDataset(tr,cfg["img_size"],True), cfg["batch_size"],True,**kw),
        DataLoader(SynapseTrainDataset(va,cfg["img_size"],False),cfg["batch_size"],False,**kw),
        DataLoader(SynapseTestDataset(all_test,cfg["img_size"]),  cfg["batch_size"],False,**kw),
    )

def build_model(cfg):
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=3, classes=cfg["num_classes"],
        decoder_attention_type="scse",
    ).to(DEVICE)
    ckpt = Path(cfg["checkpoint"])
    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        print(f"  Loaded v3 checkpoint: {ckpt}")
    else:
        print(f"  WARNING: checkpoint not found at {ckpt}")
        print(f"  Update CFG['checkpoint'] to the path of your v3 .pth file")
    return model


class FocalTverskyLoss(nn.Module):
    def __init__(self, C, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.C,self.alpha,self.beta,self.gamma,self.smooth = C,alpha,beta,gamma,smooth
    def forward(self, logits, targets):
        probs = torch.softmax(logits,1); loss=0.0
        for c in range(1,self.C):
            p=probs[:,c]; t=(targets==c).float()
            tp=(p*t).sum(); fp=(p*(1-t)).sum(); fn=((1-p)*t).sum()
            tv=(tp+self.smooth)/(tp+self.alpha*fn+self.beta*fp+self.smooth)
            loss+=(1-tv)**self.gamma
        return loss/(self.C-1)

class FineTuneLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.C  = cfg["num_classes"]
        self.ft = FocalTverskyLoss(self.C, cfg["tversky_alpha"], cfg["tversky_beta"], cfg["focal_gamma"])
        w = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=DEVICE)
        self.ce = nn.CrossEntropyLoss(weight=w)
    def _dice(self, logits, targets, smooth=1e-6):
        probs=torch.softmax(logits,1); loss=0.0
        for c in range(1,self.C):
            p=probs[:,c]; t=(targets==c).float()
            loss+=1-(2*(p*t).sum()+smooth)/(p.sum()+t.sum()+smooth)
        return loss/(self.C-1)
    def forward(self, logits, targets):
        return 0.4*self.ft(logits,targets)+0.3*self._dice(logits,targets)+0.3*self.ce(logits,targets)


@torch.no_grad()
def compute_metrics(preds, targets, C, smooth=1e-6):
    dice,iou={},{}
    for c in range(1,C):
        p=(preds==c).float(); t=(targets==c).float()
        if t.sum()==0 and p.sum()==0: continue
        inter=(p*t).sum(); union=p.sum()+t.sum()-inter
        dice[c]=float((2*inter+smooth)/(p.sum()+t.sum()+smooth))
        iou[c]=float((inter+smooth)/(union+smooth))
    return dice,iou

def train_epoch(model,loader,optimizer,criterion,scaler):
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
def eval_epoch(model,loader,criterion,C,class_names):
    model.eval()
    total_loss,all_p,all_t=0.0,[],[]
    for imgs,masks in loader:
        imgs,masks=imgs.to(DEVICE),masks.to(DEVICE)
        logits=model(imgs)
        total_loss+=criterion(logits,masks).item()
        all_p.append(logits.argmax(1).cpu()); all_t.append(masks.cpu())
    all_p=torch.cat(all_p); all_t=torch.cat(all_t)
    dice,iou=compute_metrics(all_p,all_t,C)
    mean_dice=float(np.mean(list(dice.values()))) if dice else 0.0
    mean_iou=float(np.mean(list(iou.values()))) if iou else 0.0
    named_dice={class_names[c]:round(v,4) for c,v in dice.items()}
    named_iou={class_names[c]:round(v,4) for c,v in iou.items()}
    return total_loss/len(loader),mean_dice,mean_iou,named_dice,named_iou


def train(cfg):
    print("\n"+"="*65)
    print(" MediVision v3.1 — Liver/Kidney Boundary Fine-tuning")
    print("="*65)
    print(f"  Starting from: v3 checkpoint")
    print(f"  LR: {cfg['lr']} (10x lower than v3)")
    print(f"  Liver weight: 2.0 -> 4.0 | Right kidney: 1.5 -> 0.8")
    print(f"  Epochs: {cfg['epochs']} (fine-tuning only)\n")

    train_loader,val_loader,test_loader = build_dataloaders(cfg)
    model     = build_model(cfg)
    criterion = FineTuneLoss(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-7)
    scaler    = GradScaler("cuda")

    best_dice,no_improve,history = -1.0,0,[]
    out = Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)

    v3_baseline = {"liver":0.7126,"right_kidney":0.9499,"gallbladder":0.7350,"pancreas":0.8412}

    for epoch in range(1,cfg["epochs"]+1):
        t0 = time.time()
        tr_loss = train_epoch(model,train_loader,optimizer,criterion,scaler)
        va_loss,mean_dice,mean_iou,named_dice,named_iou = eval_epoch(
            model,val_loader,criterion,cfg["num_classes"],cfg["class_names"])
        scheduler.step()
        elapsed = time.time()-t0

        # Show the organs we care about most
        watch = {k:v for k,v in named_dice.items() if k in ("liver","right_kidney","gallbladder","pancreas")}
        watch_str = " | ".join(f"{k[:2]}:{v:.3f}(v3:{v3_baseline.get(k,0):.3f})" for k,v in watch.items())
        print(f"Ep {epoch:3d}/{cfg['epochs']} | train={tr_loss:.4f} | val={va_loss:.4f} | "
              f"dice={mean_dice:.4f} | [{watch_str}] | {elapsed:.0f}s")

        record = {"epoch":epoch,"train_loss":round(tr_loss,4),"val_loss":round(va_loss,4),
                  "mean_dice":round(mean_dice,4),"per_class_dice":named_dice,
                  "lr":optimizer.param_groups[0]["lr"]}
        history.append(record)

        if mean_dice > best_dice:
            best_dice=mean_dice; no_improve=0
            torch.save(model.state_dict(), out/"unet_resnet34_v31_best.pth")
            print(f"  ✓ Best: {best_dice:.4f}")
        else:
            no_improve+=1

        with open(out/"history_v31.json","w") as f:
            json.dump(history,f,indent=2)

        if no_improve>=cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch}"); break

    # Test evaluation
    print("\n"+"="*65)
    print(" Test evaluation")
    print("="*65)
    model.load_state_dict(torch.load(out/"unet_resnet34_v31_best.pth",
                                     map_location=DEVICE,weights_only=True))
    _,test_dice,test_iou,nd,ni = eval_epoch(
        model,test_loader,criterion,cfg["num_classes"],cfg["class_names"])

    print(f"\nTest Dice: {test_dice:.4f}  (v3: 0.8593,  v2: 0.7760)")
    print("\nPer-class vs v3:")
    v3 = {"aorta":0.8916,"gallbladder":0.7350,"spleen":0.9162,"left_kidney":0.9019,
          "right_kidney":0.9499,"liver":0.7126,"stomach":0.9260,"pancreas":0.8412}
    for cls,score in nd.items():
        diff=score-v3.get(cls,0)
        sign="▲" if diff>0.005 else ("▼" if diff<-0.005 else "~")
        bar="█"*int(score*20)
        print(f"  {cls:15s} {score:.4f}  {sign}{abs(diff):.3f}  {bar}")

    final = {
        "version":"v3.1","dataset":"Synapse multi-organ CT",
        "test_dice":round(test_dice,4),"test_iou":round(test_iou,4),
        "per_class_dice":nd,"per_class_iou":ni,
        "v3_baseline":0.8593,"improvement":round(test_dice-0.8593,4),
        "finetune_changes":{"liver_weight":"2.0->4.0","right_kidney_weight":"1.5->0.8",
                            "lr":"1e-4->1e-5","epochs":50,"start":"v3 checkpoint"},
        "note":"Fine-tuned from v3 to fix liver/right_kidney boundary confusion"
    }
    with open(out/"final_results_v31.json","w") as f:
        json.dump(final,f,indent=2)

    print(f"\nOutputs: {out}/")
    print("  unet_resnet34_v31_best.pth  <- new weights")
    print("  final_results_v31.json      <- update MODEL_CARD.md")
    return best_dice,final

if __name__=="__main__":
    best_dice,results = train(CFG)
