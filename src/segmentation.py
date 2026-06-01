"""
Segmentation inference — UNet-ResNet34 v2 checkpoint.
Volumes are NOT reported — requires full DICOM series with pixel spacing metadata.
Only pixel coverage (%) and detection confidence are reported.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp

CLASS_NAMES = [
    "background","aorta","gallbladder","spleen",
    "left_kidney","right_kidney","liver","stomach","pancreas"
]

# Approximate % of 224x224 slice a healthy organ occupies
# Used only for "relative size" assessment — not volume
EXPECTED_COVERAGE_PCT = {
    "liver":        (15, 35),
    "spleen":       (3,  10),
    "left_kidney":  (2,  8),
    "right_kidney": (2,  8),
    "pancreas":     (1,  5),
    "aorta":        (0.2, 2),
    "gallbladder":  (0.5, 4),
    "stomach":      (2,  12),
}

class SegmentationModel:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=9,
        ).to(self.device)

        ckpt = Path(checkpoint)
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            print(f"  Loaded checkpoint: {ckpt}")
        else:
            print(f"  WARNING: checkpoint not found at {ckpt}")
        self.model.eval()

    @torch.no_grad()
    def run(self, pil_image) -> dict:
        img_np = np.array(pil_image.convert("L")).astype(np.float32)
        mn, mx = img_np.min(), img_np.max()
        if mx - mn > 1e-8:
            img_np = (img_np - mn) / (mx - mn)

        img_r = np.array(Image.fromarray(img_np).resize((224,224), Image.BILINEAR))
        tensor = torch.from_numpy(
            np.stack([img_r, img_r, img_r], axis=0)
        ).unsqueeze(0).float().to(self.device)

        probs = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu().numpy()
        pred  = np.argmax(probs, axis=0)
        total_pixels = 224 * 224

        organ_masks = {}
        for cid in range(1, 9):
            name  = CLASS_NAMES[cid]
            mask  = (pred == cid).astype(np.uint8)
            count = int(mask.sum())
            if count > 30:
                conf     = float(probs[cid][pred == cid].mean())
                coverage = round(count / total_pixels * 100, 2)
                organ_masks[name] = {
                    "detected":       True,
                    "pixel_count":    count,
                    "coverage_pct":   coverage,
                    "confidence":     round(conf, 3),
                    "mask":           mask,
                }

        return {
            "pred_mask":   pred,
            "probs":       probs,
            "organ_masks": organ_masks,
            "anomalies":   self._detect_anomalies(pred, probs),
        }

    def _detect_anomalies(self, pred, probs):
        anomalies = []
        for cid in range(1, 9):
            mask = (pred == cid)
            if mask.sum() > 30:
                conf = float(probs[cid][mask].mean())
                if conf < 0.55:
                    anomalies.append({
                        "type":        f"low_confidence_{CLASS_NAMES[cid]}",
                        "pixel_count":  int(mask.sum()),
                        "confidence":   round(conf, 3),
                        "severity":     "low",
                    })
        return anomalies

    def measure(self, seg_output: dict) -> dict:
        """
        Returns pixel coverage percentage only.
        Volume measurement requires full DICOM series + pixel spacing metadata.
        This is intentionally NOT computed to avoid misleading clinical values.
        """
        measurements = {}
        total = 224 * 224
        for organ, data in seg_output.get("organ_masks", {}).items():
            coverage = data["pixel_count"] / total * 100
            exp      = EXPECTED_COVERAGE_PCT.get(organ)
            measurements[organ] = {
                "coverage_pct":          round(coverage, 2),
                "volume_cm3":            "N/A — requires full DICOM series",
                "volume_note":           "Volumetry requires multi-slice DICOM with pixel spacing metadata",
            }
        return measurements

    def compare_to_normals(self, measurements: dict) -> dict:
        """
        Compares slice coverage % against expected ranges.
        Returns relative assessment only — NOT a clinical volume measurement.
        """
        comparison = {}
        for organ, data in measurements.items():
            coverage = data.get("coverage_pct", 0)
            exp      = EXPECTED_COVERAGE_PCT.get(organ)
            if not exp:
                continue
            lo, hi = exp
            if coverage < lo * 0.5:
                status = "smaller_than_expected"
            elif coverage > hi * 1.5:
                status = "larger_than_expected"
            else:
                status = "within_expected_range"
            comparison[organ] = {
                "coverage_pct":    coverage,
                "expected_range":  f"{lo}–{hi}%",
                "status":          status,
                "note":            "Coverage-based only — not a volume measurement",
            }
        return comparison
