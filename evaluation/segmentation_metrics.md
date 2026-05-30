# Segmentation Evaluation

## Dataset

- **Source:** Multi-organ CT annotation dataset (derived from public TCIA/Synapse splits)
- **Split:** 80% train / 20% validation (held-out, not used during training)
- **Scan type:** Non-contrast abdominal CT, 512×512 2D slices
- **Annotations:** Liver, pancreas, left/right kidneys, spleen, tumor regions

> ⚠️ **Transparency note:** Validation was performed on an internal held-out split.
> No external test set or clinical validation has been done. These numbers reflect
> in-distribution performance only and should not be interpreted as clinical accuracy.

---

## Results

### Per-organ Dice Score (validation set)

| Organ | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ | n slices |
|---|---|---|---|---|---|
| Liver | **0.914** | 0.842 | 0.931 | 0.898 | 214 |
| Kidney (left) | **0.903** | 0.824 | 0.918 | 0.889 | 198 |
| Kidney (right) | **0.899** | 0.832 | 0.912 | 0.887 | 198 |
| Spleen | **0.891** | 0.814 | 0.904 | 0.878 | 187 |
| Pancreas | **0.837** | 0.746 | 0.854 | 0.821 | 201 |
| Tumor | **0.786** | 0.683 | 0.802 | 0.771 | 89 |
| **Overall (weighted avg)** | **0.882** | 0.775 | 0.901 | 0.867 | — |

### Training configuration

```
Backbone:      ResNet-34 (ImageNet pretrained, fine-tuned)
Loss:          0.5 × Dice + 0.5 × BCE  (ablation in experiments/loss_ablation.md)
Optimizer:     Adam, lr=1e-4, weight decay=1e-5
LR schedule:   ReduceLROnPlateau (patience=5)
Batch size:    8
Input size:    512×512
Augmentation:  Random horizontal/vertical flip, rotation ±15°,
               pixel-based class balancing, patch extraction
Hardware:      CUDA GPU, ~6 hrs training
```

---

## Backbone Ablation

Three encoders tested on identical training config. Full log in `experiments/backbone_comparison.md`.

| Backbone | Val Dice | Params | Train time | Selected |
|---|---|---|---|---|
| ResNet-34 | **0.882** | 21M | 6 hrs | ✅ |
| EfficientNet-B4 | 0.871 | 19M | 7 hrs | |
| VGG-16 | 0.863 | 138M | 11 hrs | |

ResNet-34 selected: best Dice, smallest footprint, fastest training.

---

## Known Limitations

| Limitation | Impact |
|---|---|
| 2D slice-only — no 3D volumetric context | Adjacent-slice information ignored; volume estimates are approximations |
| Trained on non-contrast CT only | Performance degrades on contrast-enhanced CT (different HU distributions) |
| Small tumor training set (89 slices) | Tumor class confidence is lower; small lesions (<1cm equivalent) frequently missed |
| No external validation | Results may not generalize to scans from different hospitals or scanners |
| Not validated on MRI | Architecture could support MRI; no experiments run |

## Failure Cases

**Small pancreatic tumors:** lesions with pixel count below ~100 (< ~7mm estimated diameter) are missed in most test cases. Pancreas has the highest shape variability.

**Organ boundary bleed:** liver-stomach contact regions occasionally produce boundary bleed-through — predicted liver mask extends slightly into stomach territory.

**Motion/artifact corruption:** heavily artifact-corrupted slices (metal implants, motion blur) cause unreliable predictions. No artifact rejection is implemented.

**Confidence miscalibration:** softmax confidence is not calibrated. A 0.92 confidence does not reliably equal 92% accuracy. Temperature scaling was not applied.

---

## Reproducing These Results

```bash
python evaluation/segmentation_eval.py \
  --images data/val/images/ \
  --masks  data/val/masks/ \
  --checkpoint models/unet_resnet34_multiorgan.pth \
  --output evaluation/results/seg_eval.json
```

See `evaluation/segmentation_eval.py` for the full evaluation script.
