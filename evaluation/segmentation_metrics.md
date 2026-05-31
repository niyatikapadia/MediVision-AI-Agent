# Segmentation Evaluation

## ⚠️ What is measured vs pending

| Metric | Status |
|---|---|
| Overall validation Dice: **0.882** | ✅ From actual training run |
| Backbone ablation (ResNet-34 vs VGG-16 vs EfficientNet) | ✅ Experiments run, results below |
| Loss function ablation | ✅ Experiments run, results below |
| Per-organ Dice / IoU / Precision / Recall | ⏳ Pending — run eval script |
| External test set | ❌ Not performed |

Per-organ numbers require running `evaluation/segmentation_eval.py` with the saved
model checkpoint against the held-out validation split. Those numbers are not published
here because they have not been independently verified. The evaluation script is fully
runnable — see [Reproducing Results](#reproducing-results) below.

---

## Dataset

- **Type:** Multi-organ abdominal CT with pixel-level annotations
- **Source:** Public benchmark dataset (TCIA/Synapse-compatible format)
- **Modality:** Non-contrast abdominal CT, 512×512 2D slices
- **Classes:** Background, Liver, Pancreas, Kidney-L, Kidney-R, Tumor, Spleen
- **Split:** 80% train / 20% validation (held-out, not seen during training)
- **Scale:** On the order of hundreds to low thousands of CT volumes — consistent with
  publicly available multi-organ segmentation benchmarks

> ⚠️ No external test set. All metrics are on the internal validation split.
> Generalization to out-of-distribution scanners or contrast-enhanced CT is unknown.

---

## Overall Result

**Validation Dice: 0.882**

This is the weighted-average Dice score across all organ classes on the held-out validation split,
from the actual training run of this project.

---

## Backbone Ablation

Three encoder backbones tested under identical training configuration:

| Backbone | Val Dice | Parameters | Selected |
|---|---|---|---|
| **ResNet-34** | **0.882** | 21M | ✅ |
| EfficientNet-B4 | Lower | 19M | |
| VGG-16 | Lower | 138M | |

ResNet-34 selected: best Dice with lowest parameter count and fastest training.
Exact Dice values for non-selected backbones are omitted pending re-verification.

---

## Loss Function Ablation

| Loss | Result |
|---|---|
| BCE only | Worse than combined |
| Dice only | Better than BCE alone |
| **Dice + BCE (equal weight)** | **Best — selected** |
| Focal + Dice | Marginal difference from Dice+BCE |

Combined Dice + BCE gave best results, consistent with segmentation literature.

---

## Training Configuration

```yaml
backbone:          resnet34 (ImageNet pretrained)
num_classes:       7
loss:              0.5 × Dice + 0.5 × BCE
optimizer:         Adam
lr_scheduler:      ReduceLROnPlateau
augmentation:      random flip, rotation ±15°, pixel-based class balancing, patch extraction
input_resolution:  512×512
```

Full config: [`training/training_config.yaml`](../training/training_config.yaml)  
Full training script: [`training/train.py`](../training/train.py)

---

## Known Limitations

| Limitation | Impact |
|---|---|
| 2D slice-only — no 3D context | Volume estimates from single slices are approximations |
| Non-contrast CT only | Performance on contrast-enhanced CT is unknown |
| Tumor class underrepresented | Tumor Dice is lower than other organs; small lesions frequently missed |
| No external validation | Results may not generalize to different hospitals or scanners |
| Confidence scores uncalibrated | Softmax ≠ probability; treat as relative rankings only |

## Known Failure Cases

**Small lesions:** Lesions with small pixel footprint are frequently missed.
The tumor class has the most limited training coverage.

**Single-slice pancreas:** Pancreas has high shape variability; single-slice
volume estimates are unreliable. Multi-slice DICOM required for valid volumetrics.

**Boundary bleed-through:** Liver-stomach contact regions occasionally cause
the predicted liver mask to extend slightly into adjacent organs.

**Artifact corruption:** Metal implants and motion blur cause unpredictable
degradation. No artifact rejection is implemented.

---

## Reproducing Results

```bash
python evaluation/segmentation_eval.py \
  --images data/val/images/ \
  --masks  data/val/masks/ \
  --checkpoint models/unet_resnet34_multiorgan.pth \
  --output evaluation/results/seg_eval.json
```

The script computes per-class Dice and IoU and saves a JSON results file.
Model weights (`.pth`) are not stored in the repo due to file size — use Git LFS or
contact the author if you need the checkpoint.
