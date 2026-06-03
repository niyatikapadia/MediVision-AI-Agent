"""
Segmentation inference — UNet-ResNet34 v3 checkpoint.

Post-processing:
  1. Connected Component Analysis (CCA) — removes small false-positive blobs,
     keeps only the largest connected component per organ class.
     This is standard post-processing in medical image segmentation.

  2. Anatomical sanity checks — flags predictions outside expected anatomical
     bounds as "segmentation quality warnings" rather than clinical findings.
     Rule-based validation, not AI inference.

Area methodology:
  Cross-sectional area (cm2) at the detected slice level.
  Pixel spacing estimated from standard abdominal CT FOV (370mm/512px).
  Uncertainty +-25%. For definitive measurements, use DICOM metadata.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp
from scipy import ndimage

CLASS_NAMES = [
    "background","aorta","gallbladder","spleen",
    "left_kidney","right_kidney","liver","stomach","pancreas"
]

# Single-slice cross-sectional area reference ranges (cm2)
# Source: CT anatomy atlases, Heymsfield et al. 1997, Emamian et al. 1993
SINGLE_SLICE_RANGES = {
    "liver":        {"area_cm2": (80,  220), "level": "porta hepatis",       "ref": "Heymsfield 1997"},
    "spleen":       {"area_cm2": (15,  55),  "level": "splenic hilum",        "ref": "Dittmar 2021"},
    "left_kidney":  {"area_cm2": (12,  30),  "level": "renal hilum",          "ref": "Emamian 1993"},
    "right_kidney": {"area_cm2": (12,  30),  "level": "renal hilum",          "ref": "Emamian 1993"},
    "pancreas":     {"area_cm2": (5,   20),  "level": "pancreatic body",      "ref": "Saisho 2007"},
    "gallbladder":  {"area_cm2": (2,   15),  "level": "gallbladder fossa",    "ref": "Everson 1980"},
    "stomach":      {"area_cm2": (10,  60),  "level": "variable",             "ref": "variable"},
    "aorta":        {"diameter_mm": (15, 25),"level": "infrarenal",           "ref": "ACR 2024"},
}

# Anatomical sanity thresholds — beyond these, prediction is likely wrong
# Set at 3x the upper reference bound
SANITY_THRESHOLDS = {
    "liver":        {"max_area_cm2": 400},   # >400cm2 = impossible on single slice
    "spleen":       {"max_area_cm2": 120},
    "left_kidney":  {"max_area_cm2": 60},    # >60cm2 = almost certainly over-segmented
    "right_kidney": {"max_area_cm2": 60},
    "pancreas":     {"max_area_cm2": 50},
    "gallbladder":  {"max_area_cm2": 40},
    "stomach":      {"max_area_cm2": 150},
    "aorta":        {"max_diameter_mm": 50}, # >50mm = extremely rare AAA
}

# Minimum pixel count to keep a connected component
# Smaller blobs are false positives
MIN_COMPONENT_PIXELS = {
    "liver":       500,
    "spleen":      200,
    "left_kidney": 150,
    "right_kidney":150,
    "pancreas":    80,
    "gallbladder": 60,
    "stomach":     200,
    "aorta":       30,
}

FOV_MM         = 370.0
NATIVE_SIZE_PX = 512
PIXEL_SPACING  = FOV_MM / NATIVE_SIZE_PX   # 0.684 mm/px
UNCERTAINTY    = 25                         # +-25%


class SegmentationModel:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device   = torch.device(device)
        self.img_size = 512
        self.px_mm    = PIXEL_SPACING

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=9,
            decoder_attention_type="scse",
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

        # Post-processing: connected component analysis
        pred_cleaned = self._apply_cca(pred)

        organ_masks = {}
        for cid in range(1, 9):
            name = CLASS_NAMES[cid]
            mask = (pred_cleaned == cid).astype(np.uint8)
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
            "pred_mask":   pred_cleaned,
            "probs":       probs,
            "organ_masks": organ_masks,
            "anomalies":   self._detect_anomalies(pred_cleaned, probs),
        }

    def _apply_cca(self, pred_mask: np.ndarray) -> np.ndarray:
        """
        Connected Component Analysis post-processing.

        For each organ class:
          1. Find all connected components
          2. Keep only the largest component (anatomically, each organ is one blob)
          3. Remove components smaller than MIN_COMPONENT_PIXELS

        This eliminates scattered false-positive pixels and small blobs
        that the model sometimes produces around organ boundaries.
        """
        cleaned = np.zeros_like(pred_mask)

        for cid in range(1, 9):
            name = CLASS_NAMES[cid]
            binary = (pred_mask == cid).astype(np.uint8)
            if binary.sum() == 0:
                continue

            # Label connected components
            labeled, num_components = ndimage.label(binary)
            if num_components == 0:
                continue

            # Find component sizes
            component_sizes = ndimage.sum(binary, labeled, range(1, num_components + 1))
            min_size = MIN_COMPONENT_PIXELS.get(name, 50)

            # Keep only the largest component IF it meets minimum size
            largest_idx  = int(np.argmax(component_sizes)) + 1
            largest_size = int(component_sizes[np.argmax(component_sizes)])

            if largest_size >= min_size:
                cleaned[labeled == largest_idx] = cid

        return cleaned

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
        measurements = {}
        for organ, data in seg_output.get("organ_masks", {}).items():
            px       = data["pixel_count"]
            area_mm2 = px * (self.px_mm ** 2)
            area_cm2 = area_mm2 / 100.0

            if organ == "aorta":
                diam_mm = round(2 * np.sqrt(area_mm2 / np.pi), 1)
                measurements[organ] = {
                    "diameter_mm":   diam_mm,
                    "diam_range_mm": (
                        round(diam_mm * (1 - UNCERTAINTY/100), 1),
                        round(diam_mm * (1 + UNCERTAINTY/100), 1),
                    ),
                    "pixel_count":   px,
                    "confidence":    data["confidence"],
                    "method":        f"estimated — FOV={FOV_MM}mm assumed, +-{UNCERTAINTY}%",
                }
            else:
                measurements[organ] = {
                    "area_cm2":       round(area_cm2, 1),
                    "area_range_cm2": (
                        round(area_cm2 * (1 - UNCERTAINTY/100), 1),
                        round(area_cm2 * (1 + UNCERTAINTY/100), 1),
                    ),
                    "pixel_count":   px,
                    "confidence":    data["confidence"],
                    "method":        f"estimated — FOV={FOV_MM}mm assumed, +-{UNCERTAINTY}%",
                }
        return measurements

    def compare_to_normals(self, measurements: dict) -> dict:
        """
        Compare areas to reference ranges.
        Adds anatomical sanity check — flags impossible predictions.
        """
        comparison = {}
        for organ, meas in measurements.items():
            ref     = SINGLE_SLICE_RANGES.get(organ)
            sanity  = SANITY_THRESHOLDS.get(organ, {})
            if not ref:
                continue

            if "area_cm2" in ref:
                area     = meas.get("area_cm2", 0)
                lo, hi   = ref["area_cm2"]
                a_lo, a_hi = meas.get("area_range_cm2", (area, area))

                # Anatomical sanity check FIRST
                max_area = sanity.get("max_area_cm2", 9999)
                if area > max_area:
                    comparison[organ] = {
                        "area_cm2":            area,
                        "area_range_cm2":      meas.get("area_range_cm2"),
                        "reference_range_cm2": [lo, hi],
                        "status":              "SEGMENTATION_WARNING",
                        "warning":             (
                            f"Detected area {area} cm2 exceeds anatomical maximum "
                            f"({max_area} cm2 for {organ}). "
                            "Likely over-segmentation. Do not use for clinical assessment."
                        ),
                        "anatomical_level":    ref["level"],
                        "reference":           ref["ref"],
                    }
                    continue

                # Normal range comparison
                if a_hi < lo:       status = "below_normal"
                elif a_lo > hi:     status = "above_normal"
                elif area < lo or area > hi: status = "borderline"
                else:               status = "normal"

                comparison[organ] = {
                    "area_cm2":            area,
                    "area_range_cm2":      meas.get("area_range_cm2"),
                    "reference_range_cm2": [lo, hi],
                    "status":              status,
                    "anatomical_level":    ref["level"],
                    "reference":           ref["ref"],
                    "uncertainty":         f"+-{UNCERTAINTY}%",
                }

            elif "diameter_mm" in ref:
                diam   = meas.get("diameter_mm", 0)
                lo, hi = ref["diameter_mm"]
                d_lo, d_hi = meas.get("diam_range_mm", (diam, diam))
                max_diam = sanity.get("max_diameter_mm", 9999)

                if diam > max_diam:
                    comparison[organ] = {
                        "diameter_mm": diam,
                        "status":      "SEGMENTATION_WARNING",
                        "warning":     f"Diameter {diam}mm exceeds anatomical maximum.",
                    }
                    continue

                if d_hi < lo:   status = "below_normal"
                elif d_lo > hi: status = "above_normal — possible aneurysm"
                elif diam < lo or diam > hi: status = "borderline"
                else:           status = "normal"

                comparison[organ] = {
                    "diameter_mm":        diam,
                    "diam_range_mm":      meas.get("diam_range_mm"),
                    "reference_range_mm": [lo, hi],
                    "status":             status,
                    "anatomical_level":   ref["level"],
                    "reference":          ref["ref"],
                    "uncertainty":        f"+-{UNCERTAINTY}%",
                }

        return comparison
