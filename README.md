<div align="center">

# MediVision AI Agent

**Multimodal medical imaging analysis pipeline**  
UNet-ResNet34 segmentation · Hybrid medical RAG · LangGraph agent reasoning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-1C3C3C?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![CI](https://github.com/niyatikapadia/MediVision-AI-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/niyatikapadia/MediVision-AI-Agent/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*[Niyati Kapadia](https://niyatinikunjkapadia.wixsite.com/portfolio) · [LinkedIn](https://www.linkedin.com/in/niyati-nikunj-k-ab47861a4/)*

</div>

---

## What This Is

MediVision is a **research prototype** that orchestrates multiple AI components into a single clinical analysis pipeline:

1. A CT scan image enters the system
2. A trained **UNet-ResNet34** segmentation model identifies organs and flags anomalies
3. A **hybrid RAG pipeline** (BM25 + BioBERT + FAISS) retrieves relevant medical literature
4. A **LangGraph agent loop** reasons across all inputs and produces a structured differential diagnosis

The key idea: the LLM doesn't just answer one question — it uses tools iteratively, deciding which tool to call next based on what it found, up to a configurable maximum of iterations.

**Honest scope:** this is a working prototype. The segmentation model is genuinely trained (88.2% val Dice). The agent orchestration is fully implemented. The RAG knowledge base currently has 5 sample documents — not live PubMed. See [Current Status](#-current-status) for the complete picture.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MediVision Pipeline                       │
│                                                             │
│  Input: CT scan PNG + optional clinical notes               │
│                          │                                  │
│              ┌───────────▼───────────┐                      │
│              │   LangGraph Agent     │                      │
│              │                       │                      │
│              │  ┌─────────────────┐  │                      │
│              │  │  LLM (reason)   │◄─┼── tool results       │
│              │  └────────┬────────┘  │                      │
│              │           │ tool_call │                      │
│              │  ┌────────▼────────┐  │                      │
│              │  │   Tool Router   │  │                      │
│              │  └──┬──────────┬───┘  │                      │
│              └─────┼──────────┼──────┘                      │
│                    │          │                             │
│           ┌────────▼──┐  ┌───▼────────┐                    │
│           │  Vision   │  │    RAG     │                    │
│           │  Tools    │  │   Tools    │                    │
│           └────────┬──┘  └───┬────────┘                    │
│                    │         │                             │
│            UNet-ResNet34   FAISS + BM25                    │
│            segmentation    hybrid retrieval                │
│                                                             │
│  Output: findings · differential diagnosis · confidence     │
└─────────────────────────────────────────────────────────────┘
```

**Agent tools:**

| Tool | What it does | Implemented |
|---|---|---|
| `segment_scan` | UNet-ResNet34 inference, returns organ masks + anomaly flags | ✅ |
| `measure_anomalies` | Pixel-area to mm² / estimated volume conversion | ✅ |
| `compare_to_normals` | Volume vs age/sex-adjusted reference ranges | ✅ |
| `retrieve_medical_knowledge` | Hybrid BM25 + BioBERT FAISS search | ✅ |

---

## Segmentation Results

Trained on a multi-organ CT dataset. Evaluated on a held-out internal validation split (not used during training).

| Metric | Value | Status |
|---|---|---|
| Overall validation Dice | **0.882** | ✅ From actual training run |
| Per-organ breakdown | See eval script | ⏳ Run `evaluation/segmentation_eval.py` to reproduce |
| External test set | Not performed | ⚠️ Known limitation |

> Per-organ Dice/IoU numbers are pending independent verification.
> Run the evaluation script with the model checkpoint to generate them.
> See [`evaluation/segmentation_metrics.md`](evaluation/segmentation_metrics.md).

> ⚠️ Internal validation only — no external test set. Per-organ metrics pending verification.
> See [`evaluation/segmentation_metrics.md`](evaluation/segmentation_metrics.md) for full methodology, limitations, and how to reproduce.

**Backbone ablation** — why ResNet-34:

| Backbone | Val Dice | Params | Train time |
|---|---|---|---|
| ResNet-34 ✅ | **0.882** | 21M | 6 hrs |
| EfficientNet-B4 | 0.871 | 19M | 7 hrs |
| VGG-16 | 0.863 | 138M | 11 hrs |

---

## Workflow Trace

This is a real step-by-step execution trace from the LangGraph agent:

```
Input: ct_abdomen_slice.png + "58yo male, elevated ALT, abdominal discomfort"

[1] Agent → segment_scan("ct_abdomen_slice.png")
    Reasoning: "Need visual findings first."
    Returns: liver 0.921 conf · pancreas 0.847 · anomaly 0.762 (14px equiv)

[2] Agent → measure_anomalies(output)
    Reasoning: "Anomaly found. Need size before retrieval query."
    Returns: liver 1520cm³ · anomaly diameter ~14.2mm

[3] Agent → retrieve_medical_knowledge("14mm hepatic lesion elevated LFTs CT")
    Returns (RRF top-3):
      [0.032] "Liver volumetry CT validation" — normal 1200–1800cm³ ✓
      [0.024] "ACR Criteria: Liver Lesion" — lesions >1cm → contrast MRI
      [0.018] "AI differential diagnosis CT" — AI reduces errors 31%

[4] Agent → compare_to_normals(measurements, age=58)
    Returns: liver NORMAL (1520, range 1200–1800) · pancreas below (single-slice artifact)

[5] Agent finalizes
    Differential: hemangioma 0.58 · cyst 0.22 · HCC cannot exclude 0.12
    Followup: contrast-enhanced MRI (ACR criteria)
    Iterations used: 5 of 6 max
```

Full traces with JSON outputs in [`examples/example_case_1.md`](examples/example_case_1.md) and [`examples/example_case_2.md`](examples/example_case_2.md).

---

## Quickstart

**Prerequisites:** Python 3.10+, 4GB RAM, optional GPU

```bash
git clone https://github.com/niyatikapadia/MediVision-AI-Agent.git
cd MediVision-AI-Agent

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add ANTHROPIC_API_KEY or OPENAI_API_KEY (or use local Ollama)

# Generate synthetic test images
python data/generate_sample_data.py
```

```python
from src.agents.medivision_agent import MediVisionAgent

agent = MediVisionAgent(llm_backend="claude")   # or "gpt4o" or "local"

result = agent.analyze(
    scan_path="data/sample_data/ct_abdomen_slice.png",
    clinical_notes="58yo male. Elevated liver enzymes. Abdominal discomfort."
)

print(result.differential_diagnosis)
result.save_report("output/report.json")
```

**Or with Docker:**

```bash
docker compose up
# API at http://localhost:8000/docs
```

**API endpoints:**
- `POST /analyze` — full pipeline (scan + clinical notes → report)
- `POST /segment` — segmentation only
- `POST /search` — RAG search only
- `GET /health` — model status

---

## Project Structure

```
MediVision-AI-Agent/
│
├── src/
│   ├── agents/medivision_agent.py    # LangGraph graph + tool registration
│   ├── vision/segmentation.py        # UNet-ResNet34 definition + inference
│   ├── rag/pipeline.py               # BM25 + FAISS + RRF hybrid retrieval
│   ├── api/main.py                   # FastAPI endpoints
│   └── utils/report_generator.py    # Output formatting + FHIR structure
│
├── evaluation/
│   ├── segmentation_metrics.md       # Dice/IoU results, failure cases, methodology
│   ├── rag_benchmark.md              # Retrieval quality + traces
│   ├── agent_reasoning_tests.md      # 10-case manual evaluation with traces
│   └── segmentation_eval.py         # Runnable evaluation script
│
├── examples/
│   ├── example_case_1.md             # Full trace: hepatic lesion case
│   └── example_case_2.md             # Full trace: normal baseline case
│
├── docs/
│   └── architecture.md               # Design decisions + references
│
├── data/
│   ├── generate_sample_data.py        # Generates synthetic test images
│   └── sample_data/                   # Synthetic PNG scans (NOT real patient data)
│
├── tests/
│   └── test_full_suite.py             # Unit tests: segmentation, RAG, reports, API
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## ✅ Current Status

### Fully implemented

| Component | Location | Notes |
|---|---|---|
| UNet-ResNet34 architecture | `src/vision/segmentation.py` | Full forward pass, trained weights not in repo (100MB+) |
| Segmentation inference | `src/vision/segmentation.py` | Preprocessing, mask extraction, anomaly detection |
| Organ measurement | `src/vision/segmentation.py` | Pixel-area estimation, normal range comparison |
| BM25 retrieval | `src/rag/pipeline.py` | `rank_bm25`, keyword matching |
| BioBERT dense retrieval | `src/rag/pipeline.py` | `sentence-transformers`, FAISS index |
| RRF fusion | `src/rag/pipeline.py` | Reciprocal rank fusion (k=60) |
| LangGraph agent loop | `src/agents/medivision_agent.py` | Tool-use, conditional edges, max-iteration gate |
| FastAPI REST backend | `src/api/main.py` | /analyze, /segment, /search, /health |
| Docker deployment | `Dockerfile`, `docker-compose.yml` | CPU image, optional Ollama sidecar |
| Unit tests | `tests/test_full_suite.py` | Segmentation, RAG, reports, API schema |

### Partially implemented / demo-scale

| Component | Current state | What's missing |
|---|---|---|
| Knowledge base | 5 hardcoded sample docs | Real PubMed ingestion pipeline |
| DICOM loading | `pydicom` in deps | Actual loader not yet written |
| PDF ingestion | API accepts field | Parser not implemented |
| Confidence calibration | Raw softmax | Temperature scaling not applied |
| Model weights | Architecture complete | `.pth` file not in repo — use Git LFS if adding |

### Planned

| Feature | Why |
|---|---|
| Live PubMed ingestion via E-utilities API | Scale RAG to real literature |
| BioMistral / ClinicalBERT local LLM | Remove cloud API dependency |
| 3D volumetric segmentation (DICOM series) | True volume measurements |
| Temperature scaling for calibration | Calibrated uncertainty estimates |
| FHIR R4 schema validation | Real EHR integration |
| Web UI with scan viewer | Usable without API client |

---

## Technical Decisions

**Why LangGraph over a sequential chain?** The agent needs to decide what to do next based on what it found — e.g., only retrieve literature if an anomaly exists. A fixed chain can't branch. LangGraph models this as a conditional graph.

**Why hybrid retrieval?** BM25 catches exact clinical terms ("Bosniak IIF", "Child-Pugh B"). Dense BioBERT catches paraphrases ("enlarged liver" → "hepatomegaly"). Neither alone matches both. RRF fusion consistently outperforms either in medical IR literature.

**Why BioBERT over MiniLM?** Pretrained on PubMed abstracts — 85% top-1 retrieval accuracy on test queries vs 60% for general-purpose MiniLM. Trade-off: 3× slower encoding, 440MB vs 90MB.

**Why ResNet-34 encoder?** Best Dice/parameter trade-off in our backbone ablation. Full experiment in [`evaluation/segmentation_metrics.md`](evaluation/segmentation_metrics.md).

Full decision log with references in [`docs/architecture.md`](docs/architecture.md).

---

## Limitations

- **Not clinically validated.** No output has been reviewed by a medical professional.
- **Synthetic testing only.** Sample data is generated, not real CT scans.
- **Knowledge base is 5 documents.** Not representative of real literature scale.
- **Confidence scores are uncalibrated.** Softmax ≠ probability. Treat as relative rankings.
- **2D slices only.** Volume estimates from single slices are approximations.
- **May hallucinate.** LLM backends can produce plausible but wrong clinical reasoning.
- **No artifact rejection.** Metal implants, motion blur → unreliable segmentation.

---

## References

- Ronneberger et al. (2015) — [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- He et al. (2016) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Lee et al. (2020) — [BioBERT: a pre-trained biomedical language representation model](https://arxiv.org/abs/1901.08746)
- Johnson et al. (2019) — [Billion-scale similarity search with FAISS](https://arxiv.org/abs/1702.08734)
- Cormack et al. (2009) — [Reciprocal Rank Fusion outperforms Condorcet](https://dl.acm.org/doi/10.1145/1571941.1572114)
- Guo et al. (2017) — [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>
This is a research prototype. Not for clinical use.<br>
Built by <a href="https://niyatinikunjkapadia.wixsite.com/portfolio">Niyati Kapadia</a>
</sub>
</div>
