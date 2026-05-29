"""
MediVision AI Agent — FastAPI REST backend.

Exposes the full agentic pipeline as an API for integration
with EHR systems, PACS viewers, and clinical dashboards.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.agents.medivision_agent import MediVisionAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MediVision AI Agent API",
    description=(
        "Multimodal medical imaging analysis powered by UNet-ResNet34 segmentation, "
        "BioBERT-based medical RAG, and LangGraph agentic reasoning. "
        "Generates structured differential diagnoses from CT/MRI scans."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

agent = MediVisionAgent(llm_backend="local")


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    version: str


class AnalysisResponse(BaseModel):
    findings: list[dict]
    differential_diagnosis: list[dict] = Field(description="Ranked diagnoses with confidence scores")
    confidence_scores: dict[str, float]
    evidence_chain: list[str] = Field(description="Supporting evidence from medical literature")
    recommended_followup: list[str]
    report_text: str


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model="unet_resnet34 + langraph_agent",
        device=str(agent.segmentation_model.device),
        version="1.0.0",
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_scan(
    scan: UploadFile = File(..., description="CT or MRI scan image (PNG, JPG, DICOM)"),
    clinical_notes: str = Form(default="", description="Free-text clinical history"),
    llm_backend: str = Form(default="local", description="LLM backend: local, claude, gpt4o"),
):
    """
    Analyze a medical scan with the full MediVision agentic pipeline.

    Runs organ segmentation, medical RAG retrieval, and LLM reasoning
    to produce a structured differential diagnosis report.
    """
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "application/dicom"}
    if scan.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {scan.content_type}. Use PNG, JPEG, or DICOM."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        scan_path = Path(tmpdir) / (scan.filename or "scan.png")
        content = await scan.read()
        scan_path.write_bytes(content)

        try:
            result = agent.analyze(
                scan_path=str(scan_path),
                clinical_notes=clinical_notes or None,
            )
        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return AnalysisResponse(
        findings=result.findings,
        differential_diagnosis=result.differential_diagnosis,
        confidence_scores=result.confidence_scores,
        evidence_chain=result.evidence_chain,
        recommended_followup=result.recommended_followup,
        report_text=result.report_text,
    )


@app.post("/segment", tags=["Vision"])
async def segment_only(
    scan: UploadFile = File(..., description="CT or MRI scan image"),
):
    """
    Run organ segmentation only — without full agentic analysis.
    Faster endpoint for pure computer vision tasks.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_path = Path(tmpdir) / (scan.filename or "scan.png")
        scan_path.write_bytes(await scan.read())

        try:
            seg_output = agent.segmentation_model.run(str(scan_path))
            measurements = agent.segmentation_model.measure_anomalies(seg_output)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    seg_output.pop("organ_masks", None)  # remove large mask arrays from response
    return JSONResponse({"segmentation": seg_output, "measurements": measurements})


@app.post("/search", tags=["RAG"])
async def search_knowledge(
    query: str = Form(..., description="Clinical question or finding"),
    domain: str = Form(default="radiology", description="Medical domain"),
    top_k: int = Form(default=5),
):
    """
    Query the medical knowledge base directly.
    Returns relevant PubMed abstracts and clinical guidelines.
    """
    try:
        results = agent.rag_pipeline.search(query, domain=domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"query": query, "domain": domain, "results": results[:top_k]}
