# Model Card — MediVision Segmentation Model

> Following Mitchell et al. (2019) model card format.

---

## Status: Research Prototype

Not validated for clinical use.
All numbers are from real training runs on the Synapse multi-organ CT dataset.

---

## Model Overview

| Field | Value |
|---|---|
| **Model** | UNet with ResNet-34 encoder + SCSE decoder attention |
| **Version** | v3 |
| **Task** | Multi-class semantic segmentation of abdominal CT scans |
| **Input** | 512×512 grayscale CT slice (normalized, replicated to 3 channels) |
| **Output** | Per-pixel class mask (9 classes: background + 8 organs) |
| **Framework** | PyTorch 2.5 + segmentation-models-pytorch 0.5 |
| **Status** | Research prototype — NOT validated for clinical use |

---

## Intended Use

**Intended:** AI/ML research, computer vision benchmarking, portfolio demonstration.

**Not intended:** Clinical diagnosis, medical decision-making, patient-facing deployment.

---

## Training Data

| Field | Value |
|---|---|
| Dataset | Synapse Multi-Organ CT (TransUNet preprocessed) |
| Source | synapse.org/Synapse:syn3193805 |
| Modality | Abdominal CT, 2D axial slices |
| Classes | Background + 8 organs |
| Train slices | 1,768 (80% of train_npz, seed=42) |
| Val slices | 443 (20% of train_npz) |
| Test volumes | 12 volumes / 1,568 slices (official test_vol_h5) |

---

## Evaluation Results (v3)

### Overall

| Metric | Value |
|---|---|
| **Test Dice (mean)** | **0.8593** |
| **Test IoU (mean)** | **0.7623** |
| Best val Dice | 0.9348 |

### Per-class Dice (official test set)

| Organ | Dice | Status |
|---|---|---|
| Right kidney | 0.9499 | Strong |
| Stomach | 0.9260 | Strong |
| Spleen | 0.9162 | Strong |
| Left kidney | 0.9019 | Strong |
| Aorta | 0.8916 | Strong |
| Pancreas | 0.8412 | Good |
| Gallbladder | 0.7350 | Acceptable |
| Liver | 0.7126 | Weak — known boundary issue |

### vs Published Baselines

| Model | Dice |
|---|---|
| **MediVision v3** | **0.859** |
| SwinUNet | 0.790 |
| TransUNet | 0.772 |

---

## Known Limitations

- No external validation set
- 2D slices only — no 3D volumetric context
- Liver/kidney boundary confusion on some slices (documented)
- Confidence scores uncalibrated (raw softmax)
- Trained on non-contrast CT only
- No artifact rejection

## Failure Cases

**Liver boundary:** On slices where liver and kidney are adjacent, the model
occasionally misassigns liver pixels to the kidney class. Dice 0.71 reflects this.
Post-processing (CCA + anatomical sanity checks) catches the worst cases.

**Gallbladder when absent:** Gallbladder is sometimes absent or collapsed in CT.
The model occasionally produces a small false-positive in the gallbladder region.
CCA post-processing removes most of these.

---

## Post-Processing

All inference runs through:
1. **Connected Component Analysis** — largest component per organ only
2. **Anatomical sanity checks** — flags impossible predictions as SEGMENTATION_WARNING
3. **No disease inference** — measurements reported, no diagnoses made

---

## Citation

```bibtex
@misc{kapadia2026medivision,
  author    = {Kapadia, Niyati},
  title     = {MediVision AI Agent},
  year      = {2026},
  url       = {https://github.com/niyatikapadia/MediVision-AI-Agent}
}
```

---

## References

- Mitchell et al. (2019) — Model Cards for Model Reporting
- Ronneberger et al. (2015) — U-Net
- He et al. (2016) — Deep Residual Learning
- Chen et al. (2021) — TransUNet
- Abraham & Khan (2019) — Focal Tversky Loss
