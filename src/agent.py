"""
MediVision Agent — Ollama local LLM reasoning.
Honest about what is and is not measured.
"""
from __future__ import annotations
import requests

SYSTEM_PROMPT = """You are MediVision, a medical imaging AI assistant (research prototype).

You receive:
- CT scan organ segmentation results (detection confidence + slice coverage %)
- Retrieved medical literature
- Clinical notes

IMPORTANT CONSTRAINTS:
- Volume measurements are NOT available (requires full DICOM series)
- You only have one 2D slice — this is a limitation you must acknowledge
- You are a research prototype — always recommend expert radiologist review

Produce a structured report with these sections:
**Key Findings** — what was detected, with confidence levels
**Clinical Observations** — what the coverage patterns suggest
**Differential Considerations** — 2-3 possibilities with reasoning
**Recommended Next Steps** — specific, actionable
**Limitations** — what this single-slice analysis cannot tell us

End with: DISCLAIMER: Research prototype. Not for clinical use. Requires expert radiologist review."""

class MediVisionAgent:
    def __init__(self, rag, ollama_model: str = "llama3",
                 ollama_url: str = "http://localhost:11434"):
        self.rag        = rag
        self.model      = ollama_model
        self.ollama_url = ollama_url
        self._check_ollama()

    def _check_ollama(self):
        try:
            r      = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"  Ollama connected. Models: {models}")
            if not any(self.model in m for m in models):
                if models:
                    self.model = models[0].split(":")[0]
                    print(f"  Using {self.model}")
        except Exception as e:
            print(f"  Ollama not reachable: {e}")

    def reason(self, seg_output, measurements, normals, rag_results, clinical_notes) -> dict:
        prompt = self._build_prompt(seg_output, measurements, normals, rag_results, clinical_notes)
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model":   self.model,
                    "prompt":  prompt,
                    "system":  SYSTEM_PROMPT,
                    "stream":  False,
                    "options": {"num_predict": 400, "temperature": 0.2, "top_p": 0.9}
                },
                timeout=180,
            )
            text = resp.json().get("response", "No response.")
        except Exception as e:
            text = self._fallback(normals, seg_output, str(e))
        return {"report_text": text, "model": self.model}

    def _build_prompt(self, seg_output, measurements, normals, rag_results, clinical_notes):
        organs    = seg_output.get("organ_masks", {})
        anomalies = seg_output.get("anomalies", [])

        organ_lines = "\n".join(
            f"  {name}: confidence={d['confidence']:.2f}, "
            f"slice_coverage={measurements.get(name,{}).get('coverage_pct',0):.1f}%, "
            f"size_assessment={normals.get(name,{}).get('status','unknown')}"
            for name, d in organs.items()
        ) or "  No organs detected"

        evidence = "\n".join(
            f"  [{i+1}] {r['title']}: {r['abstract'][:100]}..."
            for i, r in enumerate(rag_results[:2])
        )

        anomaly_lines = "\n".join(
            f"  {a['type']}: confidence={a['confidence']:.2f}"
            for a in anomalies
        ) or "  None flagged"

        return f"""CLINICAL NOTES: {clinical_notes}

DETECTED ORGANS ({len(organs)} organs on this slice):
{organ_lines}

FLAGGED ANOMALIES:
{anomaly_lines}

RETRIEVED EVIDENCE:
{evidence}

NOTE: This is a single 2D axial CT slice. Volume measurements require full DICOM series.
Coverage percentages reflect the organ's footprint on this one slice only.

Write a structured diagnostic report."""

    def _fallback(self, normals, seg_output, error):
        lines = [f"**Agent unavailable ({error[:50]})**\n\n**Automated Findings:**\n"]
        for organ, data in normals.items():
            lines.append(f"- {organ}: {data.get('status','unknown')}")
        lines.append("\nDISCLAIMER: Research prototype. Not for clinical use.")
        return "\n".join(lines)
