"""
Report Generator — parses agent output into structured AnalysisResult.
Supports FHIR R4-compatible JSON output format.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Converts raw LLM agent output into a structured AnalysisResult.

    Handles:
    - JSON extraction from agent messages
    - Fallback heuristic parsing for non-JSON output
    - FHIR DiagnosticReport structure generation
    """

    def parse_agent_output(
        self,
        agent_output: str,
        segmentation_output: dict | None = None,
        rag_results: list[dict] | None = None,
    ):
        from src.agents.medivision_agent import AnalysisResult

        parsed = self._try_json_parse(agent_output)

        if parsed:
            return AnalysisResult(
                findings=parsed.get("findings", []),
                differential_diagnosis=parsed.get("differential_diagnosis", []),
                confidence_scores=parsed.get("confidence_scores", {}),
                evidence_chain=parsed.get("evidence_chain", []),
                recommended_followup=parsed.get("recommended_followup", []),
                report_text=parsed.get("report_text", agent_output),
                metadata=self._build_metadata(segmentation_output, rag_results),
            )

        # Fallback: build result from segmentation + RAG data
        findings = self._extract_findings_from_seg(segmentation_output)
        return AnalysisResult(
            findings=findings,
            differential_diagnosis=[{"diagnosis": "See report text", "confidence": 0.5}],
            confidence_scores={},
            evidence_chain=[r.get("title", "") for r in (rag_results or [])],
            recommended_followup=["Radiologist review recommended"],
            report_text=agent_output,
            metadata=self._build_metadata(segmentation_output, rag_results),
        )

    def _try_json_parse(self, text: str) -> dict | None:
        """Extract and parse JSON block from agent output."""
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _extract_findings_from_seg(self, seg_output: dict | None) -> list[dict]:
        if not seg_output:
            return []
        findings = []
        for organ, data in (seg_output.get("organ_masks") or {}).items():
            findings.append({
                "organ": organ,
                "detected": data.get("detected", False),
                "confidence": data.get("confidence", 0.0),
            })
        for anomaly in seg_output.get("anomalies", []):
            findings.append({
                "type": "anomaly",
                "description": anomaly.get("type"),
                "severity": anomaly.get("severity"),
                "confidence": anomaly.get("confidence", 0.0),
            })
        return findings

    def _build_metadata(self, seg_output: dict | None, rag_results: list | None) -> dict:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model": "medivision-v1.0",
            "segmentation_available": seg_output is not None,
            "rag_sources_used": len(rag_results or []),
        }

    def to_fhir(self, result, patient_id: str = "unknown") -> dict:
        """Convert AnalysisResult to FHIR R4 DiagnosticReport resource."""
        return {
            "resourceType": "DiagnosticReport",
            "id": f"medivision-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "RAD"}]}],
            "code": {"text": "AI-Assisted Imaging Analysis — MediVision"},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": result.metadata.get("generated_at"),
            "conclusion": result.report_text,
            "conclusionCode": [
                {
                    "coding": [{"display": dx.get("diagnosis", "")}],
                    "text": f"Confidence: {dx.get('confidence', 0):.0%}",
                }
                for dx in result.differential_diagnosis
            ],
        }
