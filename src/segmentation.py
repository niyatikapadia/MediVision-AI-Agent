"""
Segmentation inference — UNet-ResNet34 v2 checkpoint.

Volume/Area methodology:
  We report cross-sectional AREA (cm²) on this single slice, not whole-organ volume.
  This is the correct measurement for single-slice CT analysis.

  Reference ranges are single-slice cross-sectional areas at standard anatomical levels,
  derived from published CT anatomy literature:
    - Liver cross-section at porta hepatis: 150–280 cm²
    - Kidney cross-section at hilum: 15–35 cm²
    - Pancreas cross-section at body: 8–25 cm²
    - Spleen cross-section at hilum: 20–60 cm²
    - Aorta: diameter 1.5–2.5 cm (not area)

  Pixel spacing estimated from standard abdominal CT FOV (370mm/512px = 0.684mm/px).
  Uncertainty ±25% due to unknown actual scanner protocol.

  For whole-organ volumetry, provide full DICOM series with pixel spacing metadata.
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

# Single-slice cross-sectional area reference ranges (cm²)
# At standard anatomical level for each organ
# Source: CT anatomy atlases + Heymsfield et al. 1997
SINGLE_SLICE_RANGES = {
    "liver":        {"area_cm2": (80,  220), "level": "porta hepatis",
                     "ref": "Heymsfield et al. 1997"},
    "spleen":       {"area_cm2": (15,  55),  "level": "splenic hilum",
                     "ref": "Dittmar et al. 2021"},
    "left_kidney":  {"area_cm2": (12,  30),  "level": "renal hilum",
                     "ref": "Emamian et al. 1993"},
    "right_kidney": {"area_cm2": (12,  30),  "level": "renal hilum",
                     "ref": "Emamian et al. 1993"},
    "pancreas":     {"area_cm2": (5,   20),  "level": "pancreatic body",
                     "ref": "Saisho et al. 2007"},
    "gallbladder":  {"area_cm2": (2,   15),  "level": "gallbladder fossa",
                     "ref": "Everson et al. 1980"},
    "stomach":      {"area_cm2": (10,  60),  "level": "variable — depends on filling",
                     "ref": "variable"},
    "aorta":        {"diameter_mm": (15, 25), "level": "infrarenal",
                     "ref": "ACR Guidelines 2024"},
}

# Standard abdominal CT acquisition assumptions
FOV_MM           = 370.0   # typical adult abdominal CT field-of-view
NATIVE_SIZE_PX   = 512     # standard CT reconstruction matrix
PIXEL_SPACING_MM = FOV_MM / NATIVE_SIZE_PX   # 0.684 mm/px
UNCERTAINTY_PCT  = 25      # ±25% for unknown scanner protocol


class SegmentationModel:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device   = torch.device(device)
        self.img_size = 224
        # Scale pixel spacing to our 224px inference size
        self.px_mm    = PIXEL_SPACING_MM * (NATIVE_SIZE_PX / self.img_size)

        self.model = smp.Unet(
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

        img_r  = np.array(
            Image.fromarray(img_np).resize((self.img_size, self.img_size), Image.BILINEAR)
        )
        tensor = torch.from_numpy(
            np.stack([img_r, img_r, img_r], axis=0)
        ).unsqueeze(0).float().to(self.device)

        probs = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu().numpy()
        pred  = np.argmax(probs, axis=0)

        organ_masks = {}
        for cid in range(1, 9):
            name  = CLASS_NAMES[cid]
            mask  = (pred == cid).astype(np.uint8)
            count = int(mask.sum())
            if count > 30:
                conf = float(probs[cid][pred == cid].mean())
                organ_masks[name] = {
                    "detected":    True,
                    "pixel_count": count,
                    "confidence":  round(conf, 3),
                    "mask":        mask,
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
                        "severity":    "low",
                    })
        return anomalies

    def measure(self, seg_output: dict) -> dict:
        """
        Compute single-slice cross-sectional area for each detected organ.
        This is the correct measurement for 2D CT slice analysis.
        Pixel spacing estimated from standard abdominal CT FOV.
        """
        measurements = {}
        for organ, data in seg_output.get("organ_masks", {}).items():
            px       = data["pixel_count"]
            area_mm2 = px * (self.px_mm ** 2)
            area_cm2 = area_mm2 / 100.0

            if organ == "aorta":
                # Report diameter for tubular structures
                diam_mm = round(2 * np.sqrt(area_mm2 / np.pi), 1)
                measurements[organ] = {
                    "diameter_mm":     diam_mm,
                    "diam_range_mm":   (
                        round(diam_mm * (1 - UNCERTAINTY_PCT/100), 1),
                        round(diam_mm * (1 + UNCERTAINTY_PCT/100), 1),
                    ),
                    "pixel_count":     px,
                    "confidence":      data["confidence"],
                    "method":          f"estimated — FOV={FOV_MM}mm assumed, ±{UNCERTAINTY_PCT}%",
                }
            else:
                measurements[organ] = {
                    "area_cm2":        round(area_cm2, 1),
                    "area_range_cm2":  (
                        round(area_cm2 * (1 - UNCERTAINTY_PCT/100), 1),
                        round(area_cm2 * (1 + UNCERTAINTY_PCT/100), 1),
                    ),
                    "pixel_count":     px,
                    "confidence":      data["confidence"],
                    "method":          f"estimated — FOV={FOV_MM}mm assumed, ±{UNCERTAINTY_PCT}%",
                }
        return measurements

    def compare_to_normals(self, measurements: dict) -> dict:
        """
        Compare single-slice cross-sectional areas to published reference ranges.
        Uses conservative flagging — only flags outside uncertainty bounds.
        """
        comparison = {}
        for organ, meas in measurements.items():
            ref = SINGLE_SLICE_RANGES.get(organ)
            if not ref:
                continue

            if "area_cm2" in ref:
                area     = meas.get("area_cm2", 0)
                lo, hi   = ref["area_cm2"]
                a_lo, a_hi = meas.get("area_range_cm2", (area, area))

                # Conservative: only flag if outside even accounting for uncertainty
                if a_hi < lo:
                    status = "below_normal"
                elif a_lo > hi:
                    status = "above_normal"
                elif area < lo or area > hi:
                    status = "borderline"
                else:
                    status = "normal"

                comparison[organ] = {
                    "area_cm2":           area,
                    "area_range_cm2":     meas.get("area_range_cm2"),
                    "reference_range_cm2":[lo, hi],
                    "anatomical_level":   ref["level"],
                    "status":             status,
                    "reference":          ref["ref"],
                    "uncertainty":        f"±{UNCERTAINTY_PCT}%",
                }

            elif "diameter_mm" in ref:
                diam   = meas.get("diameter_mm", 0)
                lo, hi = ref["diameter_mm"]
                d_lo, d_hi = meas.get("diam_range_mm", (diam, diam))

                if d_hi < lo:   status = "below_normal"
                elif d_lo > hi: status = "above_normal — possible aneurysm"
                elif diam < lo or diam > hi: status = "borderline"
                else:           status = "normal"

                comparison[organ] = {
                    "diameter_mm":         diam,
                    "diam_range_mm":       meas.get("diam_range_mm"),
                    "reference_range_mm":  [lo, hi],
                    "anatomical_level":    ref["level"],
                    "status":              status,
                    "reference":           ref["ref"],
                    "uncertainty":         f"±{UNCERTAINTY_PCT}%",
                }

        return comparison
