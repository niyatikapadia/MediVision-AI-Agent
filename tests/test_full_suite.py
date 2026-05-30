"""
Unit tests for MediVision AI Agent.

Tests cover: segmentation model, RAG pipeline, report generator, and API schema.
All tests run without network access or GPU.
"""

import json
import numpy as np
import pytest
import torch


# ─── Segmentation ────────────────────────────────────────────────────────────

class TestUNetArchitecture:

    def test_forward_pass_output_shape(self):
        """UNet must output (B, num_classes, H, W) matching input spatial dims."""
        from src.vision.segmentation import UNetResNet34
        model = UNetResNet34(num_classes=7)
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 512, 512)
            out = model(x)
        assert out.shape == (1, 7, 512, 512), f"Bad output shape: {out.shape}"

    def test_batch_size_2(self):
        """Model must handle batch size > 1."""
        from src.vision.segmentation import UNetResNet34
        model = UNetResNet34(num_classes=7)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 512, 512)
            out = model(x)
        assert out.shape[0] == 2

    def test_output_is_not_all_background(self):
        """Sanity: model should predict at least some non-background pixels."""
        from src.vision.segmentation import UNetResNet34
        model = UNetResNet34(num_classes=7)
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 512, 512)
            out = model(x)
        pred = out.argmax(dim=1).squeeze(0).numpy()
        assert pred.max() > 0, "All pixels predicted as background — something is wrong"


class TestSegmentationMetrics:

    def test_dice_perfect(self):
        """Dice of identical masks = 1.0."""
        from evaluation.segmentation_eval import dice_score
        mask = np.ones((100, 100), dtype=np.uint8)
        assert abs(dice_score(mask, mask) - 1.0) < 1e-5

    def test_dice_no_overlap(self):
        """Dice of non-overlapping masks ≈ 0."""
        from evaluation.segmentation_eval import dice_score
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[:50, :] = 1
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[50:, :] = 1
        assert dice_score(pred, gt) < 0.01

    def test_iou_perfect(self):
        from evaluation.segmentation_eval import iou_score
        mask = np.ones((50, 50), dtype=np.uint8)
        assert abs(iou_score(mask, mask) - 1.0) < 1e-5

    def test_normal_ranges_all_organs_present(self):
        from src.vision.segmentation import NORMAL_RANGES
        required = {"liver", "pancreas", "kidney_left", "kidney_right", "spleen"}
        assert required.issubset(set(NORMAL_RANGES.keys()))

    def test_measure_anomalies_returns_correct_keys(self):
        from src.vision.segmentation import SegmentationModel
        model = SegmentationModel.__new__(SegmentationModel)
        model.PIXEL_SPACING_MM = 0.7
        seg_output = {
            "organ_masks": {
                "liver": {"pixel_count": 50000, "confidence": 0.92, "detected": True}
            },
            "anomalies": [
                {"type": "potential_mass", "pixel_count": 300, "severity": "moderate", "confidence": 0.76}
            ],
        }
        result = model.measure_anomalies(seg_output)
        assert "liver" in result
        assert "anomaly_potential_mass" in result
        assert result["liver"]["estimated_volume_cm3"] > 0

    def test_compare_to_normals_liver_normal(self):
        from src.vision.segmentation import SegmentationModel
        model = SegmentationModel.__new__(SegmentationModel)
        model.PIXEL_SPACING_MM = 0.7
        measurements = {"liver": {"estimated_volume_cm3": 1500.0}}
        result = model.compare_to_normals(measurements)
        assert result["liver"]["status"] == "normal"

    def test_compare_to_normals_liver_above(self):
        from src.vision.segmentation import SegmentationModel
        model = SegmentationModel.__new__(SegmentationModel)
        model.PIXEL_SPACING_MM = 0.7
        measurements = {"liver": {"estimated_volume_cm3": 2500.0}}
        result = model.compare_to_normals(measurements)
        assert result["liver"]["status"] == "above_normal"


# ─── RAG Pipeline ────────────────────────────────────────────────────────────

class TestRAGPipeline:

    def test_keyword_fallback_returns_results(self):
        """BM25 fallback (no rank_bm25) should return results for matching queries."""
        from src.rag.pipeline import MedicalRAGPipeline
        pipeline = MedicalRAGPipeline.__new__(MedicalRAGPipeline)
        pipeline.knowledge_base = [
            {"id": "doc1", "title": "Liver CT", "abstract": "Liver volumetry on CT scans.",
             "domain": "radiology", "year": 2024, "source": "PubMed"},
        ]
        pipeline._bm25 = None
        pipeline._index = None
        results = pipeline._keyword_fallback("liver CT")
        assert len(results) >= 1
        assert results[0]["id"] == "doc1"

    def test_rrn_fusion_ranking(self):
        """RRF should rank docs appearing in both lists higher."""
        from src.rag.pipeline import MedicalRAGPipeline
        pipeline = MedicalRAGPipeline.__new__(MedicalRAGPipeline)
        pipeline.knowledge_base = []
        dense = [
            {"id": "A", "title": "A", "abstract": "", "domain": "radiology", "year": 2024, "source": "PubMed", "dense_score": 0.9},
            {"id": "B", "title": "B", "abstract": "", "domain": "radiology", "year": 2024, "source": "PubMed", "dense_score": 0.7},
        ]
        sparse = [
            {"id": "A", "title": "A", "abstract": "", "domain": "radiology", "year": 2024, "source": "PubMed", "bm25_score": 3.0},
            {"id": "C", "title": "C", "abstract": "", "domain": "radiology", "year": 2024, "source": "PubMed", "bm25_score": 2.5},
        ]
        fused = pipeline._reciprocal_rank_fusion(dense, sparse)
        # Doc A appears in both → should be ranked first
        assert fused[0]["id"] == "A"

    def test_search_returns_list(self):
        """search() must always return a list, even with no installed deps."""
        from src.rag.pipeline import MedicalRAGPipeline
        pipeline = MedicalRAGPipeline(top_k=3)
        results = pipeline.search("liver segmentation", domain="radiology")
        assert isinstance(results, list)

    def test_domain_filter(self):
        """Results should be filtered by domain when specified."""
        from src.rag.pipeline import MedicalRAGPipeline
        pipeline = MedicalRAGPipeline(top_k=5)
        results = pipeline.search("kidney disease", domain="nephrology")
        for r in results:
            assert r["domain"] in ("nephrology", "general")


# ─── Report Generator ────────────────────────────────────────────────────────

class TestReportGenerator:

    def test_json_parse_extracts_fields(self):
        from src.utils.report_generator import ReportGenerator
        gen = ReportGenerator()
        agent_output = json.dumps({
            "findings": [{"organ": "liver", "status": "normal"}],
            "differential_diagnosis": [{"diagnosis": "normal", "confidence": 0.95}],
            "confidence_scores": {"overall": 0.95},
            "evidence_chain": ["PMID 12345"],
            "recommended_followup": ["No action needed"],
            "report_text": "Normal study."
        })
        result = gen.parse_agent_output(agent_output)
        assert result.findings[0]["organ"] == "liver"
        assert result.differential_diagnosis[0]["confidence"] == 0.95

    def test_fallback_on_non_json(self):
        """Non-JSON agent output should not crash — fallback to text report."""
        from src.utils.report_generator import ReportGenerator
        gen = ReportGenerator()
        result = gen.parse_agent_output("The scan shows a normal liver.", segmentation_output=None)
        assert result.report_text == "The scan shows a normal liver."
        assert isinstance(result.findings, list)

    def test_fhir_output_has_required_fields(self):
        from src.utils.report_generator import ReportGenerator
        from src.agents.medivision_agent import AnalysisResult
        gen = ReportGenerator()
        result = AnalysisResult(
            findings=[], differential_diagnosis=[], confidence_scores={},
            evidence_chain=[], recommended_followup=[], report_text="Normal.",
            metadata={"generated_at": "2026-01-01T00:00:00Z"}
        )
        fhir = gen.to_fhir(result, patient_id="test-001")
        assert fhir["resourceType"] == "DiagnosticReport"
        assert fhir["status"] == "final"
        assert "test-001" in fhir["subject"]["reference"]


# ─── API Schema ──────────────────────────────────────────────────────────────

class TestAPISchema:

    def test_analysis_response_model(self):
        from src.api.main import AnalysisResponse
        resp = AnalysisResponse(
            findings=[],
            differential_diagnosis=[],
            confidence_scores={},
            evidence_chain=[],
            recommended_followup=[],
            report_text="Test report."
        )
        assert resp.report_text == "Test report."

    def test_health_endpoint_structure(self):
        from src.api.main import HealthResponse
        h = HealthResponse(status="healthy", model="test", device="cpu", version="1.0.0")
        assert h.status == "healthy"
