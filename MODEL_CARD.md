# Model Card — MediVision Segmentation Model

> Following the model card format proposed by Mitchell et al. (2019).  
> This card describes the UNet-ResNet34 segmentation component of the MediVision pipeline.

---

## Model Overview

| Field | Value |
|---|---|
| **Model name** | MediVision-Seg v1.0 |
| **Architecture** | UNet with ResNet-34 encoder |
| **Task** | Multi-class semantic segmentation of abdominal CT scans |
| **Input** | 512×512 RGB PNG (CT slice, normalized) |
| **Output** | Per-pixel class mask (7 classes) |
| **Framework** | PyTorch 2.2 |
| **Status** | Research prototype — NOT validated for clinical use |

---

## Intended Use

### Intended users
- AI/ML researchers studying medical image segmentation
- Computer vision engineers evaluating multi-class segmentation architectures
- Students building portfolio projects in healthcare AI

### Intended use cases
- Organ localization as a preprocessing step for downstream AI analysis
- Anomaly region flagging for human radiologist review (with expert oversight)
- Research experimentation and architecture benchmarking

### Out-of-scope use cases
⛔ **This model must NOT be used for:**
- Clinical diagnosis or any medical decision-making
- Unsupervised deployment in any healthcare setting
- Replacing radiologist review
- Any application involving real patient data without appropriate IRB/ethics approval

---

## Training Data

### Dataset
- **Type:** Multi-organ abdominal CT scan dataset
- **Source:** Derived from publicly available medical image segmentation benchmarks (TCIA/Synapse-compatible annotation format)
- **Size:** ~1,000 annotated CT volumes, extracted as 512×512 2D slices
- **Modality:** Non-contrast abdominal CT
- **Annotation classes:** Background (0), Liver (1), Pancreas (2), Kidney-L (3), Kidney-R (4), Tumor (5), Spleen (6)
- **Annotators:** Ground truth from existing dataset annotations; not independently re-annotated

### Data split
| Split | Proportion | Use |
|---|---|---|
| Train | 80% | Model weight updates |
| Validation | 20% | Metric reporting, early stopping |
| External test | None | ⚠️ Not performed |

> ⚠️ **No external test set was used.** All reported metrics are on the internal validation split. 
> Performance on out-of-distribution scanners, contrast-enhanced CT, or MRI is unknown.

### Known data biases
- Dataset is likely skewed toward adult patients (pediatric CT is rare in public datasets)
- Non-contrast CT only — model has not seen contrast-enhanced scans during training
- Limited tumor annotations (89 positive slices) — tumor class is underrepresented
- Geographic/scanner diversity of the source dataset is unknown

---

## Evaluation

### Metrics

| Metric | Value | Status |
|---|---|---|
| Overall validation Dice | **0.882** | ✅ From actual training run |
| Per-organ Dice / IoU / Precision / Recall | Not published | ⏳ Pending — run `evaluation/segmentation_eval.py` |
| Slice counts per class | Not published | ⏳ Pending verification |
| External test set | Not performed | ⚠️ Known limitation |

Per-organ numbers have not been independently verified and are not stated here.
Run the evaluation script with the saved model checkpoint to generate them.
See [`evaluation/segmentation_metrics.md`](evaluation/segmentation_metrics.md).

Full evaluation script: [`evaluation/segmentation_eval.py`](evaluation/segmentation_eval.py)  
Full results discussion: [`evaluation/segmentation_metrics.md`](evaluation/segmentation_metrics.md)

---

## Training Configuration

Full training script: [`training/train.py`](training/train.py)  
Full config: [`training/training_config.yaml`](training/training_config.yaml)

```yaml
model:
  backbone: resnet34
  num_classes: 7
  pretrained_encoder: true   # ImageNet weights

training:
  epochs: 50
  batch_size: 8
  optimizer: adam
  lr: 0.0001
  weight_decay: 0.00001
  lr_scheduler: reduce_on_plateau
  scheduler_patience: 5
  early_stopping_patience: 10

loss:
  type: combined
  dice_weight: 0.5
  bce_weight: 0.5

augmentation:
  random_horizontal_flip: true
  random_vertical_flip: true
  rotation_degrees: 15
  class_balancing: pixel_weighted
  patch_extraction: true

hardware:
  device: cuda
  approx_train_time: 6 hours
```

---

## Known Limitations

| Limitation | Severity | Mitigation status |
|---|---|---|
| No external validation | High | Not mitigated — future work |
| 2D slices only (no 3D context) | High | Partial: planned 3D extension |
| Non-contrast CT only | High | Not mitigated |
| Small tumor training set | Medium | Dataset expansion planned |
| Confidence scores uncalibrated | Medium | Temperature scaling planned |
| No artifact rejection | Medium | Not implemented |
| No pediatric validation | Unknown | Not studied |

---

## Failure Cases

**Small lesions:** Tumors with pixel count < ~100 (estimated < 7mm diameter) are frequently missed. Precision on the tumor class drops significantly for small lesions.

**Pancreatic segmentation on single 2D slices:** The pancreas has high shape variability. Single-slice volume estimates are unreliable and should not be used for clinical measurements. Multi-slice DICOM series is required for valid volumetric analysis.

**Boundary bleed-through:** At organ contact boundaries (e.g., liver-stomach), the model occasionally extends the predicted mask of one organ into the adjacent organ's territory.

**Artifact-corrupted slices:** Metal implants, severe motion blur, or truncation artifacts cause unpredictable prediction degradation. No artifact detection or rejection is implemented.

---

## Ethical Considerations

**Data privacy:** The model was trained on datasets from existing public benchmarks. No new patient data was collected. Users must not feed real patient data into this system without appropriate institutional approval and data use agreements.

**Clinical safety:** This model has not undergone any form of clinical validation, regulatory review, or medical device certification. It must not be used in any clinical or diagnostic workflow.

**Bias and fairness:** The training data's demographic distribution is unknown. Performance may differ across patient populations, body types, and scanner manufacturers in ways that have not been studied.

**Transparency:** This model card documents known limitations honestly. Users are responsible for understanding these limitations before any application.

---

## Citation

If you use this model or codebase in research, please cite:

```
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
- Ronneberger et al. (2015) — [U-Net: CNNs for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- He et al. (2016) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
