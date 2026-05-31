"""
MediVision Agent — calls Ollama locally.
Uses tinyllama for fast local inference (~5-15 seconds).
"""
from __future__ import annotations
import json, requests

SYSTEM_PROMPT = """You are a medical imaging AI assistant.
Given CT scan segmentation results and clinical notes, produce a concise report with:
1. Key findings (2-3 bullet points)
2. Top 2 differential diagnoses with confidence (0.0-1.0)
3. One recommended follow-up action

Be brief and structured. Always end with:
DISCLAIMER: Research prototype. Not for clinical use. Requires radiologist review."""

class MediVisionAgent:
    def __init__(self, rag, ollama_model: str = "tinyllama",
                 ollama_url: str = "http://localhost:11434"):
        self.rag        = rag
        self.model      = ollama_model
        self.ollama_url = ollama_url
        self._check_ollama()

    def _check_ollama(self):
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"  Ollama connected. Models: {models}")
            if not any(self.model in m for m in models):
                print(f"  {self.model} not found. Available: {models}")
                if models:
                    self.model = models[0].split(":")[0]
                    print(f"  Falling back to {self.model}")
        except Exception as e:
            print(f"  Ollama not reachable: {e}. Run: ollama serve")

    def reason(self, seg_output, measurements, normals,
               rag_results, clinical_notes) -> dict:
        prompt = self._build_prompt(
            seg_output, measurements, normals, rag_results, clinical_notes)
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model":   self.model,
                    "prompt":  prompt,
                    "system":  SYSTEM_PROMPT,
                    "stream":  False,
                    "options": {
                        "num_predict": 300,   # short output = fast
                        "temperature": 0.3,   # focused/deterministic
                        "top_p": 0.9,
                    }
                },
                timeout=120,
            )
            text = response.json().get("response", "No response from model.")
        except Exception as e:
            text = (f"Agent error: {e}\n\n"
                    + self._fallback_report(normals, seg_output))
        return {"report_text": text, "model": self.model}

    def _build_prompt(self, seg_output, measurements, normals,
                      rag_results, clinical_notes):
        organs = seg_output.get("organ_masks", {})
        anomalies = seg_output.get("anomalies", [])

        organ_lines = "\n".join(
            f"  {name}: conf={d['confidence']:.2f}, "
            f"vol={measurements.get(name,{}).get('estimated_volume_cm3','?'):.1f}cm3, "
            f"status={normals.get(name,{}).get('status','unknown')}"
            for name, d in organs.items()
        ) or "  No organs detected"

        anomaly_lines = "\n".join(
            f"  {a['type']}: severity={a['severity']}"
            for a in anomalies
        ) or "  None"

        rag_line = rag_results[0]["abstract"][:150] if rag_results else "No evidence retrieved."

        return f"""Patient: {clinical_notes}

Segmentation findings:
{organ_lines}

Anomalies:
{anomaly_lines}

Relevant evidence: {rag_line}

Write a brief structured diagnostic report."""

    def _fallback_report(self, normals, seg_output):
        lines = ["**Automated Findings**\n"]
        for organ, data in normals.items():
            s = data.get("status","unknown")
            lines.append(f"- {organ}: {s}")
        if seg_output.get("anomalies"):
            lines.append("\n⚠️ Anomalies detected — review recommended.")
        lines.append("\nDISCLAIMER: Research prototype. Not for clinical use.")
        return "\n".join(lines)
