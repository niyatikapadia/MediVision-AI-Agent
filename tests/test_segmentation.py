"""Basic tests for the segmentation pipeline."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import torch


def test_unet_forward_pass():
    """Test UNet model produces correct output shape."""
    from src.vision.segmentation import UNetResNet34
    model = UNetResNet34(num_classes=7)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 512, 512)
        out = model(x)
    assert out.shape == (1, 7, 512, 512), f"Unexpected output shape: {out.shape}"


def test_normal_ranges_coverage():
    """Test that normal ranges cover expected organs."""
    from src.vision.segmentation import NORMAL_RANGES
    expected = {"liver", "pancreas", "kidney_left", "kidney_right", "spleen"}
    assert expected.issubset(set(NORMAL_RANGES.keys()))


def test_measure_anomalies_returns_dict():
    """Test anomaly measurement on mock segmentation output."""
    from src.vision.segmentation import SegmentationModel
    model = SegmentationModel.__new__(SegmentationModel)
    model.PIXEL_SPACING_MM = 0.7
    seg_output = {
        "organ_masks": {"liver": {"pixel_count": 50000, "confidence": 0.92, "detected": True}},
        "anomalies": [{"type": "potential_mass", "pixel_count": 300, "severity": "moderate", "confidence": 0.75}],
    }
    result = model.measure_anomalies(seg_output)
    assert "liver" in result
    assert "anomaly_potential_mass" in result
    assert result["liver"]["estimated_volume_cm3"] > 0
