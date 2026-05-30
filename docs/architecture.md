# Architecture & Technical Decisions

## Why LangGraph?

**Problem:** Chaining LLM calls sequentially (chain A → B → C) breaks as soon as the reasoning
needs to loop — e.g., segment → find anomaly → retrieve more specific literature → re-measure.

**LangGraph** models the agent as a directed graph with conditional edges. The agent can:
- Call tools in any order it decides
- Loop back (tool result → re-reason → different tool)
- Gate on a stopping condition (no more tool calls, or max iterations hit)

Alternatives considered:
- **LangChain AgentExecutor** — simpler but less control over state between steps; harder to debug
- **AutoGen** — multi-agent conversations; overkill for a single-agent pipeline
- **Raw loop** — would work but lose observability and state management

**Decision:** LangGraph for the agent loop, LangChain abstractions for LLM/tool interfaces.

---

## Why Hybrid Retrieval (BM25 + Dense)?

Medical text has two retrieval challenges that pull in opposite directions:

1. **Exact terminology matters.** "Bosniak IIF", "ALBI score", "Child-Pugh B" — these are
   precise clinical terms where a single wrong word changes meaning entirely. Dense embeddings
   sometimes miss exact matches.

2. **Semantic paraphrase is common.** "enlarged liver" vs "hepatomegaly" vs "increased hepatic
   volume" — same concept, very different surface forms. BM25 keyword search misses these.

**Hybrid BM25 + BioBERT dense with RRF fusion** handles both:
- BM25 catches exact clinical terminology
- Dense retrieval catches semantic paraphrase
- RRF ranks by reciprocal rank sum — documents appearing in both lists rise to the top

Reference: Cormack et al. (2009) — Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.

---

## Why BioBERT over general-purpose embeddings?

Tested `all-MiniLM-L6-v2` (90MB, fast) vs `dmis-lab/biobert-base-cased-v1.2` (440MB) on 20 clinical queries.

BioBERT correctly ranked the relevant document #1 in 85% of cases vs 60% for MiniLM.

Key difference: BioBERT was pretrained on PubMed abstracts + PMC full text. It has seen
"hepatocellular carcinoma" thousands of times in context. MiniLM has not.

Trade-off: BioBERT encoding is ~3× slower. For batch ingestion this is fine. For sub-100ms
real-time queries on large indexes, a distilled BioELECTRA or ClinicalBERT might be better.

---

## Why FAISS over a vector database (Pinecone, Weaviate, Qdrant)?

For the current scale (5–100k documents), FAISS in-process is:
- Zero infrastructure — no server, no API key, no latency over network
- Fast enough (<10ms on CPU for 100k vectors)
- Fully controllable (index type, quantization, search parameters)

When to switch to a vector DB:
- >1M documents (FAISS starts hitting RAM limits)
- Multi-user / concurrent queries (FAISS is not thread-safe without wrapping)
- Need for metadata filtering + vector search combined

The code is already abstracted through `MedicalRAGPipeline.search()` — swapping FAISS
for Qdrant would require changing ~30 lines in `src/rag/pipeline.py`.

---

## Why UNet with ResNet-34 encoder?

UNet is the standard architecture for medical image segmentation since Ronneberger et al. (2015)
because skip connections preserve fine spatial detail — critical for organ boundary precision.

The original UNet used a VGG-style encoder. Replacing it with ResNet-34:
- Adds residual connections → more stable training on deeper networks
- Gives access to ImageNet pretrained weights → faster convergence with less data
- Reduces parameters vs VGG-16 (21M vs 138M) with better Dice score

We tested ResNet-34, VGG-16, and EfficientNet-B4. See `evaluation/segmentation_metrics.md`
for ablation results. ResNet-34 won on Dice/parameter trade-off.

---

## Confidence Scoring — Current Approach and Limitations

**What we do:** confidence = mean softmax probability of predicted-class pixels.

**Problem:** softmax probabilities are not calibrated probabilities. A 0.92 softmax score
does not mean 92% chance of correct segmentation. Neural networks are typically overconfident.

**What should be done for production:** temperature scaling (Guo et al., 2017) on a held-out
calibration set. This is a known limitation of the current implementation.

The agent-level confidence (harmonic mean of segmentation + retrieval scores) inherits
this miscalibration. All confidence values should be treated as relative rankings,
not absolute probabilities.

---

## What Is Not Implemented

Being explicit here prevents overclaim:

| Feature | Status | File |
|---|---|---|
| DICOM native loading | Not implemented | `src/vision/dicom_loader.py` — stub only |
| PDF report parsing | Not implemented | Listed in API schema, no parser written |
| Live PubMed ingestion | Not implemented | KB is 5 hardcoded docs in `src/rag/pipeline.py` |
| Confidence calibration | Not implemented | Raw softmax used |
| FHIR R4 compliance | Prototype only | Output structure resembles FHIR, not schema-validated |
| 3D volumetric segmentation | Not implemented | 2D slices only |
| Model weights (.pth file) | Not in repo | File is ~100MB; would use Git LFS |

---

## References

- Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- He et al. (2016) — Deep Residual Learning for Image Recognition
- Lee et al. (2020) — BioBERT: a pre-trained biomedical language representation model
- Johnson et al. (2019) — Billion-scale similarity search with FAISS
- Cormack et al. (2009) — Reciprocal rank fusion outperforms Condorcet
- Guo et al. (2017) — On Calibration of Modern Neural Networks
