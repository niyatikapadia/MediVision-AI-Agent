"""
MediVision Agent — calls Ollama locally (llama3 or mistral).
No API key needed. Runs fully offline.
"""
from __future__ import annotations
import json, requests

SYSTEM_PROMPT = """You are MediVision, an expert AI medical imaging analyst.
You will receive:
- Organ segmentation results from a CT scan
- Measurements and normal range comparisons
- Retrieved medical literature

Your task: produce a structured diagnostic report with:
1. Summary of findings
2. Differential diagnosis (ranked by likelihood) with confidence 0-1
3. Recommended follow-up actions

Be concise. Always include this disclaimer at the end:
"DISCLAIMER: Research prototype only. Not for clinical use. Requires expert radiologist review."
"""

class MediVisionAgent:
    def __init__(self, rag, ollama_model: str = "llama3", ollama_url: str = "http://localhost:11434"):
        self.rag         = rag
        self.model       = ollama_model
        self.ollama_url  = ollama_url
        self._check_ollama()

    def _check_ollama(self):
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"  Ollama connected. Available models: {models}")
            if not any(self.model in m for m in models):
                print(f"  ⚠️  {self.model} not found. Available: {models}")
                if models:
                    self.model = models[0].split(":")[0]
                    print(f"  Using {self.model} instead.")
        except Exception as e:
            print(f"  ⚠️  Ollama not reachable: {e}")
            print("  Make sure Ollama is running: ollama serve")

    def reason(self, seg_output, measurements, normals, rag_results, clinical_notes) -> dict:
        prompt = self._build_prompt(seg_output, measurements, normals, rag_results, clinical_notes)
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "system": SYSTEM_PROMPT},
                timeout=120,
            )
            report_text = response.json().get("response", "No response from model.")
        except Exception as e:
            report_text = f"Agent error: {e}\n\nFallback summary:\n{self._fallback_report(normals, seg_output)}"

        return {"report_text": report_text, "model": self.model}

    def _build_prompt(self, seg_output, measurements, normals, rag_results, clinical_notes):
        organs = seg_output.get("organ_masks", {})
        anomalies = seg_output.get("anomalies", [])

        organ_lines = "\n".join(
            f"  - {name}: detected={d['detected']}, confidence={d['confidence']}, "
            f"volume≈{measurements.get(name,{}).get('estimated_volume_cm3','—')}cm³, "
            f"status={normals.get(name,{}).get('status','unknown')}"
            for name, d in organs.items()
        )
        anomaly_lines = "\n".join(
            f"  - {a['type']}: severity={a['severity']}, confidence={a['confidence']}"
            for a in anomalies
        ) or "  None detected"

        rag_lines = "\n".join(
            f"  [{i+1}] {r['title']}\n      Key: {r['abstract'][:120]}..."
            for i, r in enumerate(rag_results)
        )

        return f"""CLINICAL NOTES: {clinical_notes}

SEGMENTATION FINDINGS:
{organ_lines}

ANOMALIES:
{anomaly_lines}

RETRIEVED MEDICAL EVIDENCE:
{rag_lines}

Please provide a structured diagnostic report."""

    def _fallback_report(self, normals, seg_output):
        lines = ["## Automated Findings Summary\n"]
        for organ, data in normals.items():
            status = data.get("status", "unknown")
            emoji = "✅" if status == "normal" else "⚠️"
            lines.append(f"{emoji} {organ}: {status}")
        if seg_output.get("anomalies"):
            lines.append("\n⚠️ Anomalies detected — radiologist review recommended.")
        lines.append("\nDISCLAIMER: Research prototype only. Not for clinical use.")
        return "\n".join(lines)
