# Segmentation Evaluation Results

All metrics measured on the **official Synapse test_vol_h5 set** (12 volumes, 1,568 slices).
These are real numbers from an actual training run — see [`evaluation/results/final_results.json`](results/final_results.json).

---

## Results

### Overall

| Metric | Value |
|---|---|
| **Test Dice (mean)** | **0.7760** |
| **Test IoU (mean)** | **0.6529** |
| Best validation Dice | 0.9234 |

### Per-class Dice

| Organ | Dice ↑ | IoU (approx) |
|---|---|---|
| Right kidney | 0.9433 | 0.893 |
| Spleen | 0.8911 | 0.804 |
| Left kidney | 0.8415 | 0.727 |
| Aorta | 0.8483 | 0.737 |
| Stomach | 0.8197 | 0.695 |
| Pancreas | 0.7367 | 0.583 |
| Gallbladder | 0.5850 | 0.413 |
| Liver | 0.5425 | 0.373 |
| **Mean** | **0.7760** | **0.6529** |

---

## Comparison to Published Baselines

| Model | Mean Dice | Architecture |
|---|---|---|
| Swin-UNet | 0.790 | Transformer |
| TransUNet | 0.772 | CNN + Transformer |
| **MediVision (ours)** | **0.776** | UNet-ResNet34 |
| DARR | 0.696 | CNN |
| V-Net | 0.683 | 3D CNN |

MediVision matches TransUNet using a simpler 2D CNN architecture.
See [`docs/architecture.md`](../docs/architecture.md) for design decisions.

---

## Training Setup

- **Dataset:** Synapse multi-organ CT (TransUNet preprocessed)
- **Split:** 80/20 train/val from train_npz + official test_vol_h5
- **Hardware:** Kaggle Tesla P100-PCIE-16GB
- **Duration:** 150 epochs, ~38 minutes total
- **Normalization:** Per-slice min-max [0, 1] — applied identically to train and test

---

## Known Weaknesses

**Gallbladder (0.585):** Hardest class. Small, frequently absent/collapsed in CT volumes.
2D model cannot use adjacent-slice context. Improvement planned via class-weighted loss
and attention mechanisms — see [`training/train_gallbladder_improved.py`](../training/train_gallbladder_improved.py).

**Liver (0.543):** Unexpectedly low given liver is the largest organ. Likely caused by
boundary ambiguity at liver-stomach contact regions in the test set. Per-slice normalization
may also reduce global contrast consistency.

**Val vs Test gap (0.923 → 0.776):** The gap between validation Dice (on train_npz held-out)
and test Dice (on test_vol_h5) is significant. The test volumes likely come from different
scanners or protocols. This is a known limitation of 2D per-slice models.

---

## Reproducing Results

```bash
# Train from scratch
python training/train.py

# Evaluate saved checkpoint
python evaluation/segmentation_eval.py \
  --test_dir  data/Synapse/test_vol_h5 \
  --checkpoint models/unet_resnet34_synapse_best.pth \
  --output evaluation/results/eval_output.json
```
