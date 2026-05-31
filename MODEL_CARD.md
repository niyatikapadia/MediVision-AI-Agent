# Model Card — MediVision Segmentation Model

> Following the model card format proposed by Mitchell et al. (2019).

---

## ⚠️ Status: Research Prototype

Not validated for clinical use. All numbers below are from a real training run
on the Synapse multi-organ CT dataset (Kaggle P100, 150 epochs).

---

## Model Overview

| Field | Value |
|---|---|
| **Model name** | MediVision-Seg v1.0 |
| **Architecture** | UNet with ResNet-34 encoder |
| **Task** | Multi-class semantic segmentation of abdominal CT scans |
| **Input** | 224×224 RGB tensor (CT slice, per-slice min-max normalized) |
| **Output** | Per-pixel class mask (9 classes) |
| **Framework** | PyTorch 2.x + segmentation-models-pytorch |
| **Status** | Research prototype — NOT validated for clinical use |

---

## Intended Use

### Intended users
- AI/ML researchers studying medical image segmentation
- Computer vision engineers evaluating segmentation architectures
- Students building portfolio projects in healthcare AI

### Out-of-scope
⛔ Clinical diagnosis, medical decision-making, or any patient-facing deployment.

---

## Training Data

| Field | Value |
|---|---|
| **Dataset** | Synapse Multi-Organ CT (TransUNet preprocessed) |
| **Source** | synapse.org/Synapse:syn3193805 |
| **Modality** | Abdominal CT, 2D axial slices |
| **Classes** | Background + 8 organs (aorta, gallbladder, spleen, kidneys ×2, liver, stomach, pancreas) |
| **Train cases** | 1,768 slices (80% of train_npz, seed=42) |
| **Val cases** | 443 slices (20% of train_npz, seed=42) |
| **Test volumes** | 12 volumes / 1,568 slices (official test_vol_h5) |
| **Split** | Standard Synapse benchmark split |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Backbone | ResNet-34 (ImageNet pretrained, fine-tuned) |
| Loss | 0.5 × Dice + 0.5 × CrossEntropy |
| Optimizer | AdamW |
| Learning rate | 1e-4 → reduced on plateau (min 1e-6) |
| Batch size | 24 |
| Epochs | 150 (full run, no early stopping triggered) |
| Augmentation | Random flip (H+V), random 90° rotation |
| Normalization | Per-slice min-max to [0, 1] |
| Hardware | Kaggle Tesla P100-PCIE-16GB |
| Training time | ~38 minutes (150 epochs × ~15s/epoch) |

Full script: [`training/train.py`](training/train.py)

---

## Evaluation Results

**All numbers are from the official Synapse test_vol_h5 set — not the training split.**

### Summary

| Metric | Value |
|---|---|
| **Test Dice (mean)** | **0.7760** |
| **Test IoU (mean)** | **0.6529** |
| Best val Dice | 0.9234 |

### Per-class Dice (official test set)

| Organ | Dice ↑ | Notes |
|---|---|---|
| Right kidney | **0.9433** | Strongest class |
| Spleen | **0.8911** | |
| Aorta | **0.8483** | |
| Stomach | **0.8197** | |
| Left kidney | **0.8415** | |
| Pancreas | **0.7367** | Small, variable shape |
| Gallbladder | **0.5850** | Hardest class — see limitations |
| Liver | **0.5425** | Unexpectedly low — see limitations |
| **Mean** | **0.7760** | |

### Comparison to published baselines (Synapse benchmark)

| Model | Mean Dice |
|---|---|
| TransUNet (Chen et al., 2021) | 0.772 |
| Swin-UNet (Cao et al., 2021) | 0.790 |
| **MediVision UNet-ResNet34** | **0.776** |
| V-Net (Milletari et al., 2016) | 0.683 |
| DARR (Fu et al., 2020) | 0.696 |

> MediVision is competitive with TransUNet using a significantly simpler architecture.
> Published baselines use 3D context and larger training sets in some cases — direct
> comparison should be made cautiously.

---

## Known Limitations

| Limitation | Severity | Notes |
|---|---|---|
| 2D slices only — no 3D context | High | Gallbladder and liver suffer most |
| Gallbladder Dice 0.585 | Medium | Small, variable, often absent/collapsed in dataset |
| Liver Dice 0.543 | Medium | Unexpected — likely boundary ambiguity in test set |
| No external validation | High | Results on out-of-distribution scanners unknown |
| Per-slice normalization | Medium | Loses global HU context across slices |
| Confidence scores uncalibrated | Medium | Softmax ≠ probability |

## Failure Cases

**Gallbladder:** Frequently absent or collapsed in test volumes. 2D models cannot
use adjacent-slice context to confirm presence. This is a known hard class — even
TransUNet gets only 0.630.

**Liver boundary:** The liver-stomach boundary is ambiguous on some test slices.
Predicted liver mask occasionally bleeds into stomach territory.

**Small structures on single slices:** Aorta and pancreas tail visible only on
a few slices — single-slice predictions are noisy without 3D context.

---

## Ethical Considerations

- Trained on public benchmark data — no new patient data collected
- Must not be used with real patient data without IRB/ethics approval
- Not clinically validated — do not use in diagnostic workflows
- Demographic distribution of training data is unknown

---

## Citation

```bibtex
@misc{kapadia2026medivision,
  author    = {Kapadia, Niyati},
  title     = {MediVision AI Agent: Multimodal Medical Imaging Analysis},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/niyatikapadia/MediVision-AI-Agent}
}
```

---

## References

- Mitchell et al. (2019) — [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- Ronneberger et al. (2015) — [U-Net](https://arxiv.org/abs/1505.04597)
- He et al. (2016) — [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- Chen et al. (2021) — [TransUNet](https://arxiv.org/abs/2102.04306)
