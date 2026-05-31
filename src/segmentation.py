"""
Segmentation module — wraps your trained UNet-ResNet34 v2 checkpoint.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp

CLASS_NAMES = [
    "background","aorta","gallbladder","spleen",
    "left_kidney","right_kidney","liver","stomach","pancreas"
]

NORMAL_RANGES = {
    "liver":       {"volume_cm3": (1200, 1800)},
    "spleen":      {"volume_cm3": (100,  250)},
    "left_kidney": {"volume_cm3": (120,  200)},
    "right_kidney":{"volume_cm3": (120,  200)},
    "pancreas":    {"volume_cm3": (60,   120)},
}

PIXEL_SPACING_MM = 0.7


class SegmentationModel:
    def __init__(self, checkpoint: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Same architecture as training
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,   # no imagenet here — loading your weights
            in_channels=3,
            classes=9,
        ).to(self.device)

        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  ⚠️  Checkpoint not found at {ckpt_path} — running with random weights (demo mode)")

        self.model.eval()

    @torch.no_grad()
    def run(self, pil_image) -> dict:
        """Run segmentation on a PIL image. Returns masks + anomalies."""
        # Normalize same as training: per-slice min-max
        img_np = np.array(pil_image.convert("L")).astype(np.float32)
        mn, mx = img_np.min(), img_np.max()
        if mx - mn > 1e-8:
            img_np = (img_np - mn) / (mx - mn)

        # Resize to 224×224, replicate to 3 channels
        img_resized = np.array(
            Image.fromarray(img_np).resize((224, 224), Image.BILINEAR)
        )
        tensor = torch.from_numpy(
            np.stack([img_resized, img_resized, img_resized], axis=0)
        ).unsqueeze(0).float().to(self.device)

        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred   = np.argmax(probs, axis=0)

        # Extract per-organ results
        organ_masks = {}
        for class_id in range(1, 9):
            name  = CLASS_NAMES[class_id]
            mask  = (pred == class_id).astype(np.uint8)
            count = int(mask.sum())
            if count > 50:
                conf = float(probs[class_id][pred == class_id].mean())
                organ_masks[name] = {
                    "detected":    True,
                    "pixel_count": count,
                    "confidence":  round(conf, 3),
                    "mask":        mask,
                }

        anomalies = self._detect_anomalies(pred, probs)

        return {
            "pred_mask":   pred,
            "probs":       probs,
            "organ_masks": organ_masks,
            "anomalies":   anomalies,
        }

    def _detect_anomalies(self, pred, probs):
        anomalies = []
        # Flag low-confidence predictions on detected organs as potential anomalies
        for class_id in range(1, 9):
            mask = (pred == class_id)
            if mask.sum() > 50:
                conf = float(probs[class_id][mask].mean())
                if conf < 0.55:
                    anomalies.append({
                        "type":       f"uncertain_{CLASS_NAMES[class_id]}",
                        "pixel_count": int(mask.sum()),
                        "confidence":  round(conf, 3),
                        "severity":    "low",
                    })
        return anomalies

    def measure(self, seg_output: dict) -> dict:
        px = PIXEL_SPACING_MM
        measurements = {}
        for organ, data in seg_output.get("organ_masks", {}).items():
            area_mm2 = data["pixel_count"] * (px ** 2)
            vol_cm3  = round((area_mm2 * px * 5) / 1000, 2)
            measurements[organ] = {
                "estimated_area_mm2":   round(area_mm2, 1),
                "estimated_volume_cm3": vol_cm3,
                "note": "Single-slice estimate only — not a true volume"
            }
        for a in seg_output.get("anomalies", []):
            diam = round(2 * np.sqrt(a["pixel_count"] * (px**2) / np.pi), 1)
            measurements[f"anomaly_{a['type']}"] = {"estimated_diameter_mm": diam}
        return measurements

    def compare_to_normals(self, measurements: dict) -> dict:
        comparison = {}
        for organ, norms in NORMAL_RANGES.items():
            if organ not in measurements:
                continue
            vol  = measurements[organ].get("estimated_volume_cm3", 0)
            lo,hi = norms["volume_cm3"]
            if vol < lo:   status = "below_normal"
            elif vol > hi: status = "above_normal"
            else:          status = "normal"
            comparison[organ] = {
                "measured_volume_cm3": vol,
                "normal_range":        [lo, hi],
                "status":              status,
                "deviation_pct":       round(((vol-(lo+hi)/2)/((lo+hi)/2))*100, 1),
                "note":                "Single-slice estimate — interpret cautiously"
            }
        return comparison
