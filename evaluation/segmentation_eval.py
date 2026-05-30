"""
Segmentation evaluation script.

Computes Dice score and IoU for each organ class on a held-out validation set.
Run this to reproduce the numbers reported in the README.

Usage:
    python evaluation/segmentation_eval.py \
        --images data/val/images/ \
        --masks data/val/masks/ \
        --checkpoint models/unet_resnet34_multiorgan.pth \
        --output evaluation/results/seg_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

ORGAN_CLASSES = {
    0: "background",
    1: "liver",
    2: "pancreas",
    3: "kidney_left",
    4: "kidney_right",
    5: "tumor",
    6: "spleen",
}


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient between two binary masks."""
    pred_flat = pred.flatten().astype(float)
    target_flat = target.flatten().astype(float)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union between two binary masks."""
    pred_flat = pred.flatten().astype(bool)
    target_flat = target.flatten().astype(bool)
    intersection = (pred_flat & target_flat).sum()
    union = (pred_flat | target_flat).sum()
    return float(intersection + smooth) / float(union + smooth)


def evaluate_on_dataset(
    image_dir: Path,
    mask_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, dict]:
    """
    Run evaluation loop over all images in image_dir.

    Expects mask files named identically to image files.
    Mask pixel values correspond to ORGAN_CLASSES keys.

    Returns per-class metrics dict.
    """
    from src.vision.segmentation import UNetResNet34
    import torchvision.transforms as T

    model = UNetResNet34(num_classes=len(ORGAN_CLASSES)).to(device)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path} — using random weights (results will be meaningless)")
    model.eval()

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    logger.info(f"Evaluating on {len(image_paths)} images...")

    # Per-class accumulators
    class_dice: dict[int, list[float]] = {k: [] for k in ORGAN_CLASSES if k != 0}
    class_iou: dict[int, list[float]] = {k: [] for k in ORGAN_CLASSES if k != 0}

    for img_path in image_paths:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            logger.warning(f"No mask found for {img_path.name}, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        mask_gt = np.array(Image.open(mask_path).convert("L").resize((512, 512), Image.NEAREST))

        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        pred_mask = logits.squeeze(0).argmax(dim=0).cpu().numpy()

        for class_id in class_dice:
            pred_binary = (pred_mask == class_id).astype(np.uint8)
            gt_binary = (mask_gt == class_id).astype(np.uint8)

            # Only score if GT has pixels for this class (avoid inflating with empty slices)
            if gt_binary.sum() > 0 or pred_binary.sum() > 0:
                class_dice[class_id].append(dice_score(pred_binary, gt_binary))
                class_iou[class_id].append(iou_score(pred_binary, gt_binary))

    # Aggregate
    results = {}
    all_dice, all_iou = [], []

    for class_id, organ_name in ORGAN_CLASSES.items():
        if class_id == 0:
            continue
        dices = class_dice[class_id]
        ious = class_iou[class_id]
        if not dices:
            continue
        mean_dice = float(np.mean(dices))
        mean_iou = float(np.mean(ious))
        results[organ_name] = {
            "dice": round(mean_dice, 4),
            "iou": round(mean_iou, 4),
            "n_slices_evaluated": len(dices),
            "std_dice": round(float(np.std(dices)), 4),
        }
        all_dice.append(mean_dice)
        all_iou.append(mean_iou)
        logger.info(f"  {organ_name:15} Dice={mean_dice:.4f}  IoU={mean_iou:.4f}  (n={len(dices)})")

    results["overall"] = {
        "dice": round(float(np.mean(all_dice)), 4),
        "iou": round(float(np.mean(all_iou)), 4),
        "note": "Unweighted mean across all organ classes with non-empty GT",
    }
    logger.info(f"
  Overall Dice: {results['overall']['dice']:.4f}  IoU: {results['overall']['iou']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--images", type=Path, required=True, help="Path to validation images directory")
    parser.add_argument("--masks", type=Path, required=True, help="Path to ground truth masks directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/unet_resnet34_multiorgan.pth"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/seg_eval.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    logger.info(f"Using device: {device}")

    results = evaluate_on_dataset(args.images, args.masks, args.checkpoint, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"
Results saved to {args.output}")


if __name__ == "__main__":
    main()
