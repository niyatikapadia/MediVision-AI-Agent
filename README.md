<div align="center">

# MediVision AI Agent

**Multi-organ CT segmentation + Medical RAG + Local LLM reasoning**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*[Niyati Kapadia](https://niyatinikunjkapadia.wixsite.com/portfolio) · [LinkedIn](https://www.linkedin.com/in/niyati-nikunj-k-ab47861a4/)*

</div>

---

## What This Is

MediVision is a local, privacy-first medical imaging analysis pipeline.
Upload an abdominal CT scan slice — the system segments organs, retrieves
relevant medical literature, and generates a structured findings report.
Everything runs locally. No data leaves your machine.

**Honest scope:** Research prototype. Not clinically validated.
See [MODEL_CARD.md](MODEL_CARD.md) for full limitations.

---

## Pipeline

```
CT scan (PNG/JPG)
      ↓
UNet-ResNet34 segmentation
(trained on Synapse, Test Dice 0.859)
      ↓
CCA post-processing + anatomical sanity checks
      ↓
Medical RAG (BM25, 12 documents, organ-aware retrieval)
      ↓
Llama3 local reasoning (Ollama, fully offline)
      ↓
Structured findings report (measurements only, no diagnoses)
```

---

## Segmentation Results

Trained on Synapse multi-organ CT dataset. Evaluated on official test set.

| Organ | Dice | Organ | Dice |
|---|---|---|---|
| Right kidney | 0.950 | Pancreas | 0.841 |
| Stomach | 0.926 | Gallbladder | 0.735 |
| Spleen | 0.916 | **Liver** | **0.713** |
| Left kidney | 0.902 | Aorta | 0.892 |
| **Overall** | **0.859** | | |

Outperforms TransUNet (0.772) and SwinUNet (0.790) using a simpler 2D architecture.
Full evaluation: [`evaluation/segmentation_metrics.md`](evaluation/segmentation_metrics.md)

---

## Quickstart

```bash
git clone https://github.com/niyatikapadia/MediVision-AI-Agent.git
cd MediVision-AI-Agent
pip install -r requirements.txt

# Place model weights
cp unet_resnet34_v3_best.pth models/

# Start Ollama (required for agent reasoning)
ollama serve   # separate terminal

# Run demo
python app.py
# Open http://localhost:7860
```

---

## Project Structure

```
MediVision-AI-Agent/
├── app.py                          # Gradio demo app
├── src/
│   ├── segmentation.py             # UNet-ResNet34 inference + CCA + sanity checks
│   ├── rag_module.py               # BM25 retrieval, 12-doc medical KB
│   ├── agent.py                    # Ollama agent (findings only, no diagnoses)
│   └── visualize.py                # Segmentation overlay rendering
├── training/
│   ├── train.py                    # v3 training (512px, Focal Tversky, SCSE)
│   └── train_finetune_liver.py     # v3.1 liver fine-tuning (planned)
├── evaluation/
│   ├── segmentation_metrics.md     # Real results + comparison to baselines
│   ├── rag_benchmark.md            # RAG retrieval quality
│   ├── agent_reasoning_tests.md    # Agent trace examples
│   └── segmentation_eval.py       # Runnable evaluation script
├── examples/
│   ├── example_case_1.md           # Full pipeline trace: hepatic lesion
│   └── example_case_2.md           # Full pipeline trace: normal baseline
├── docs/
│   └── architecture.md             # Design decisions + references
├── data/
│   ├── generate_sample_data.py     # Synthetic test image generator
│   └── sample_data/                # Sample PNG scans for testing
├── models/
│   └── .gitkeep                    # Place unet_resnet34_v3_best.pth here
├── MODEL_CARD.md                   # Full model documentation
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Key Design Decisions

**Why local LLM?** Documents never leave the machine. Safe for confidential contracts, medical records, legal documents. No API costs, no rate limits.

**Why 2D slices?** Full 3D DICOM processing is a planned v2 extension. 2D is a known limitation — documented in MODEL_CARD.md.

**Why findings-only reports?** The agent is explicitly prohibited from making diagnoses. It reports measurements and observations. Expert radiologist review is always recommended.

See [`docs/architecture.md`](docs/architecture.md) for full reasoning.

---

## Limitations

- Single 2D slice — no volumetric context
- Liver/kidney boundary confusion on some slices
- Internal validation only — no external test set
- 12-document RAG KB (not full PubMed)
- Uncalibrated confidence scores

Full details: [MODEL_CARD.md](MODEL_CARD.md)

---

## Roadmap

- [ ] v3.1 — liver fine-tuning (train_finetune_liver.py ready)
- [ ] Full DICOM series support
- [ ] Temperature scaling for confidence calibration
- [ ] Hausdorff distance metrics
- [ ] nnU-Net benchmark comparison

---

## License

MIT — see [LICENSE](LICENSE).

<div align="center">
<sub>Research prototype. Not for clinical use.<br>
Built by <a href="https://niyatinikunjkapadia.wixsite.com/portfolio">Niyati Kapadia</a></sub>
</div>
