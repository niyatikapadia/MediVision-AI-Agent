# Segmentation Evaluation Results

All metrics measured on the **official Synapse test_vol_h5 set** (12 volumes, 1,568 slices).
These are real numbers from actual training runs on Kaggle P100.

---

## Results Summary

| Version | Test Dice | Test IoU | Notes |
|---|---|---|---|
| v2 (baseline) | 0.7760 | 0.6529 | 224px, basic augmentation |
| **v3 (current)** | **0.8593** | **0.7623** | 512px, Focal Tversky, SCSE attention |

---

## v3 Per-class Results (official test set)

| Organ | Dice | vs v2 | Notes |
|---|---|---|---|
| Right kidney | 0.9499 | +0.007 | Strongest class |
| Spleen | 0.9162 | +0.025 | |
| Stomach | 0.9260 | +0.106 | Large improvement |
| Left kidney | 0.9019 | +0.060 | |
| Aorta | 0.8916 | +0.043 | |
| Pancreas | 0.8412 | +0.104 | Large improvement |
| Gallbladder | 0.7350 | +0.150 | Largest improvement |
| Liver | 0.7126 | +0.170 | Known weakness — boundary confusion |
| **Mean** | **0.8593** | **+0.083** | |

---

## Comparison to Published Baselines (Synapse benchmark)

| Model | Mean Dice | Architecture |
|---|---|---|
| SwinUNet | 0.790 | Transformer |
| TransUNet | 0.772 | CNN + Transformer |
| **MediVision v3 (ours)** | **0.859** | UNet-ResNet34 + SCSE |
| DARR | 0.696 | CNN |
| V-Net | 0.683 | 3D CNN |

MediVision v3 outperforms both TransUNet and SwinUNet using a simpler 2D CNN architecture.

---

## v3 Training Configuration

| Parameter | Value |
|---|---|
| Architecture | UNet-ResNet34 + SCSE decoder attention |
| Loss | 0.4 × Focal Tversky + 0.3 × Dice + 0.3 × Weighted CE |
| Class weights | bg:0.05, aorta:1.5, gallbladder:2.5, liver:2.0, pancreas:3.0 |
| Resolution | 512×512 |
| Batch size | 8 |
| Epochs | 150 |
| LR schedule | Linear warmup (5ep) + cosine annealing |
| Augmentation | Flip, rotation, scale-crop, gamma, noise, brightness |
| Hardware | Kaggle Tesla P100-PCIE-16GB |
| Training time | ~3.5 hours |

Full training script: [`training/train.py`](../training/train.py)

---

## Known Limitations

| Issue | Impact | Status |
|---|---|---|
| Liver/kidney boundary confusion | Liver under-segmented on some slices | Known — v3.1 planned |
| 2D slices only | No volumetric context | Planned: 3D extension |
| Internal validation only | No external test set | Known limitation |
| Uncalibrated confidence | Softmax ≠ probability | Planned: temperature scaling |
| No artifact rejection | Metal/motion artifacts degrade output | Not implemented |

## Post-Processing (implemented in demo)

- **Connected Component Analysis** — keeps largest component per organ, removes false-positive blobs
- **Anatomical sanity checks** — flags predictions exceeding anatomical bounds as SEGMENTATION_WARNING
- **No disease inference** — system reports measurements only, no diagnoses

## Reproducing Results

```bash
python training/train.py
# Outputs saved to /kaggle/working/medivision_v3/
# final_results_v3.json contains all metrics
```
