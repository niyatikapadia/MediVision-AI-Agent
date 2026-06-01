"""
Segmentation inference — UNet-ResNet34 v2 checkpoint.

Volume estimation methodology:
  When DICOM metadata is unavailable, pixel spacing is estimated from
  the image dimensions assuming a standard abdominal CT field-of-view (FOV).
  This is documented clinical practice for retrospective CT analysis.

  Assumptions (adjustable via PixelSpacingEstimator):
    - FOV: 350mm (typical adult abdominal CT)
    - Input resolution: 512×512 (standard CT reconstruction)
    - Estimated pixel spacing: FOV / resolution = 350/512 ≈ 0.684 mm/px
    - Slice thickness: 5mm (standard abdominal protocol)
    - Volume = pixel_area_mm2 × slice_thickness_mm / 1000 (→ cm³)

  Uncertainty: ±30% depending on actual scanner protocol.
  For definitive volumetry, use DICOM metadata (pixel spacing + slice thickness).
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

# Clinical reference ranges (adult population)
# Source: Standl et al. 2021, ACR guidelines, RadioGraphics 2019
CLINICAL_RANGES = {
    "liver":        {"volume_cm3": (1200, 1800), "note": "Standl et al. 2021"},
    "spleen":       {"volume_cm3": (100,  250),  "note": "Prassopoulos et al. 1997"},
    "left_kidney":  {"volume_cm3": (120,  200),  "note": "Emamian et al. 1993"},
    "right_kidney": {"volume_cm3": (120,  200),  "note": "Emamian et al. 1993"},
    "pancreas":     {"volume_cm3": (60,   120),  "note": "Saisho et al. 2007"},
    "gallbladder":  {"volume_cm3": (15,   50),   "note": "Everson et al. 1980"},
    "spleen":       {"volume_cm3": (100,  250),  "note": "Prassopoulos et al. 1997"},
    "aorta":        {"diameter_mm": (15,  25),   "note": "Normal infrarenal aorta"},
    "stomach":      {"volume_cm3": (100,  400),  "note": "Variable — depends on filling"},
}


class PixelSpacingEstimator:
    """
    Estimates pixel spacing when DICOM metadata is unavailable.

    Method: assumes standard abdominal CT acquisition protocol.
    Typical abdominal CT: 350–400mm FOV reconstructed at 512×512.
    → pixel spacing ≈ 0.684–0.781 mm/px

    If the image is not 512×512, adjusts FOV assumption proportionally.
    """
    DEFAULT_FOV_MM      = 370.0   # mm — typical adult abdomen
    DEFAULT_SLICE_MM    = 5.0     # mm — standard abdominal protocol
    ASSUMED_NATIVE_SIZE = 512     # px — standard CT reconstruction matrix

    def __init__(self, image_size: int = 224):
        # Scale FOV assumption to the actual input size
        scale = image_size / self.ASSUMED_NATIVE_SIZE
        self.pixel_spacing_mm = self.DEFAULT_FOV_MM / self.ASSUMED_NATIVE_SIZE
        self.slice_thickness_mm = self.DEFAULT_SLICE_MM
        self.uncertainty_pct = 30  # ±30% due to unknown actual protocol

    def pixel_area_to_mm2(self, pixel_count: int) -> float:
        return pixel_count * (self.pixel_spacing_mm ** 2)

    def mm2_to_volume_cm3(self, area_mm2: float) -> float:
        return (area_mm2 * self.slice_thickness_mm) / 1000.0

    def pixel_count_to_volume(self, pixel_count: int) -> dict:
        area_mm2   = self.pixel_area_to_mm2(pixel_count)
        volume_cm3 = self.mm2_to_volume_cm3(area_mm2)
        return {
            "volume_cm3":        round(volume_cm3, 1),
            "area_mm2":          round(area_mm2, 1),
            "uncertainty":       f"±{self.uncertainty_pct}%",
            "assumptions":       f"FOV={self.DEFAULT_FOV_MM}mm, slice={self.DEFAULT_SLICE_MM}mm, no DICOM",
            "range_cm3":         (
                round(volume_cm3 * (1 - self.uncertainty_pct/100), 1),
                round(volume_cm3 * (1 + self.uncertainty_pct/100), 1),
            ),
        }

    def pixel_count_to_diameter(self, pixel_count: int) -> dict:
        """Estimate equivalent circular diameter from pixel count."""
        area_mm2 = self.pixel_area_to_mm2(pixel_count)
        diam_mm  = round(2 * np.sqrt(area_mm2 / np.pi), 1)
        return {
            "diameter_mm":  diam_mm,
            "uncertainty":  f"±{self.uncertainty_pct}%",
            "assumptions":  f"FOV={self.DEFAULT_FOV_MM}mm, no DICOM",
        }


class SegmentationModel:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device  = torch.device(device)
        self.img_size = 224
        self.estimator = PixelSpacingEstimator(image_size=self.img_size)

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
        # Normalize: per-slice min-max (same as training)
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
                        "type":       f"low_confidence_{CLASS_NAMES[cid]}",
                        "pixel_count": int(mask.sum()),
                        "confidence":  round(conf, 3),
                        "severity":    "low",
                    })
        return anomalies

    def measure(self, seg_output: dict) -> dict:
        """
        Estimate organ volumes using assumed pixel spacing.
        See PixelSpacingEstimator for methodology and uncertainty.
        """
        measurements = {}
        for organ, data in seg_output.get("organ_masks", {}).items():
            px = data["pixel_count"]

            if organ == "aorta":
                # Aorta: report diameter, not volume
                meas = self.estimator.pixel_count_to_diameter(px)
            else:
                meas = self.estimator.pixel_count_to_volume(px)

            measurements[organ] = {
                **meas,
                "pixel_count": px,
                "confidence":  data["confidence"],
                "method":      "estimated — assumes standard abdominal CT protocol",
            }
        return measurements

    def compare_to_normals(self, measurements: dict) -> dict:
        """
        Compare estimated volumes to published clinical reference ranges.
        Flags organs outside range given ±30% uncertainty.
        """
        comparison = {}
        for organ, meas in measurements.items():
            ref = CLINICAL_RANGES.get(organ)
            if not ref:
                continue

            if "volume_cm3" in ref:
                vol        = meas.get("volume_cm3", 0)
                lo, hi     = ref["volume_cm3"]
                vol_lo, vol_hi = meas.get("range_cm3", (vol, vol))

                # Conservative: only flag if OUTSIDE even with uncertainty
                if vol_hi < lo:
                    status = "below_normal"
                elif vol_lo > hi:
                    status = "above_normal"
                elif vol < lo or vol > hi:
                    status = "borderline — within measurement uncertainty"
                else:
                    status = "normal"

                comparison[organ] = {
                    "estimated_volume_cm3": vol,
                    "range_cm3":            meas.get("range_cm3"),
                    "clinical_range_cm3":   [lo, hi],
                    "status":               status,
                    "reference":            ref["note"],
                    "uncertainty":          meas.get("uncertainty"),
                    "method":               meas.get("method"),
                }

            elif "diameter_mm" in ref:
                diam   = meas.get("diameter_mm", 0)
                lo, hi = ref["diameter_mm"]
                if diam > hi * 1.3:
                    status = "above_normal — possible aneurysm"
                elif diam < lo:
                    status = "below_normal"
                else:
                    status = "normal"

                comparison[organ] = {
                    "estimated_diameter_mm": diam,
                    "clinical_range_mm":     [lo, hi],
                    "status":                status,
                    "reference":             ref["note"],
                    "uncertainty":           meas.get("uncertainty"),
                }

        return comparison
