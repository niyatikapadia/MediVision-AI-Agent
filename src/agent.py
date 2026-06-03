"""
MediVision Agent — Ollama local LLM.
Strict prompt: findings and observations only.
No disease diagnoses. No treatment recommendations.
"""
from __future__ import annotations
import requests

SYSTEM_PROMPT = """You are MediVision, a medical imaging AI assistant.
You are a RESEARCH PROTOTYPE. You must never make clinical diagnoses.

Your role is strictly to:
1. Describe what was detected in the segmentation results
2. Note which measurements are within or outside reference ranges
3. Flag any segmentation quality warnings
4. Recommend expert radiologist review

You must NOT:
- Diagnose any disease (e.g. do NOT say "CKD", "hepatitis", "cancer", "cirrhosis")
- Recommend any treatment
- Make prognostic statements
- Interpret findings as definitive pathology

Structure your response as:

**Imaging Observations**
Describe detected organs and their measured cross-sectional areas.
Note any SEGMENTATION_WARNING flags explicitly.

**Measurements vs Reference Ranges**
List each organ: measured area, reference range, status (normal/borderline/above/below).
For SEGMENTATION_WARNING organs: state the measurement cannot be used.

**Segmentation Quality Notes**
Note any organs with low confidence or anatomical warnings.

**Recommended Next Steps**
Always include:
- Expert radiologist review required
- Full DICOM series for volumetric measurements
- Clinical correlation with patient history

**Disclaimer**
DISCLAIMER: Research prototype. Not for clinical use.
Single 2D slice analysis only. Not validated clinically.
Expert radiologist review required before any clinical decision."""


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

    def reason(self, seg_output, measurements, normals,
               rag_results, clinical_notes) -> dict:
        prompt = self._build_prompt(
            seg_output, measurements, normals, rag_results, clinical_notes
        )
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model":   self.model,
                    "prompt":  prompt,
                    "system":  SYSTEM_PROMPT,
                    "stream":  False,
                    "options": {
                        "num_predict": 500,
                        "temperature": 0.1,  # low temp = more conservative output
                        "top_p": 0.9
                    }
                },
                timeout=180,
            )
            text = resp.json().get("response", "No response.")
        except Exception as e:
            text = self._fallback(normals, seg_output, str(e))
        return {"report_text": text, "model": self.model}

    def _build_prompt(self, seg_output, measurements, normals,
                      rag_results, clinical_notes):
        organs    = seg_output.get("organ_masks", {})
        anomalies = seg_output.get("anomalies", [])

        organ_lines = []
        warnings    = []
        for name, data in organs.items():
            norm   = normals.get(name, {})
            status = norm.get("status", "unknown")
            conf   = data["confidence"]

            if status == "SEGMENTATION_WARNING":
                warnings.append(
                    f"  SEGMENTATION WARNING — {name}: {norm.get('warning','')}"
                )
            elif "area_cm2" in norm:
                area    = norm["area_cm2"]
                ref_rng = norm.get("reference_range_cm2", [])
                organ_lines.append(
                    f"  {name}: conf={conf:.2f}, area={area}cm2 "
                    f"(ref {ref_rng[0]}-{ref_rng[1]}cm2), status={status}"
                )
            elif "diameter_mm" in norm:
                diam    = norm["diameter_mm"]
                ref_rng = norm.get("reference_range_mm", [])
                organ_lines.append(
                    f"  {name}: conf={conf:.2f}, diameter={diam}mm "
                    f"(ref {ref_rng[0]}-{ref_rng[1]}mm), status={status}"
                )

        evidence = "\n".join(
            f"  [{i+1}] {r['title']}: {r['abstract'][:100]}..."
            for i, r in enumerate(rag_results[:2])
        )

        sections = [f"CLINICAL NOTES: {clinical_notes}\n"]

        if warnings:
            sections.append("SEGMENTATION QUALITY WARNINGS (do not use for clinical assessment):")
            sections.extend(warnings)
            sections.append("")

        sections.append(f"DETECTED ORGANS ({len(organ_lines)} with usable measurements):")
        sections.extend(organ_lines)

        if anomalies:
            sections.append("\nLOW CONFIDENCE FLAGS:")
            for a in anomalies:
                sections.append(f"  {a['type']}: conf={a['confidence']:.2f}")

        sections.append(f"\nRELEVANT LITERATURE:\n{evidence}")
        sections.append(
            "\nIMPORTANT: This is a single 2D axial slice. "
            "Do not infer diagnoses. Describe observations only."
        )

        return "\n".join(sections)

    def _fallback(self, normals, seg_output, error):
        lines = [
            f"Agent unavailable ({error[:50]})\n",
            "**Automated Observations:**\n"
        ]
        for organ, data in normals.items():
            status = data.get("status", "unknown")
            if status == "SEGMENTATION_WARNING":
                lines.append(f"- {organ}: SEGMENTATION WARNING — measurement unreliable")
            else:
                lines.append(f"- {organ}: {status}")
        lines.append(
            "\nDISCLAIMER: Research prototype. "
            "Not for clinical use. Expert radiologist review required."
        )
        return "\n".join(lines)
