# Example Case 1 — Hepatic Lesion Workup

A complete walkthrough of MediVision processing an abdominal CT case.

---

## Input

```
Scan:           ct_abdomen_slice.png (512×512 PNG, single CT slice)
Clinical notes: "58-year-old male. Elevated liver enzymes (ALT 2.4× ULN).
                 Abdominal discomfort for 3 weeks. No prior liver disease."
LLM backend:    claude-sonnet-4-20250514
```

---

## Step 1 — Vision: Segmentation Output

```python
agent.segmentation_model.run("data/sample_data/ct_abdomen_slice.png")
```

```json
{
  "organ_masks": {
    "liver":       {"detected": true,  "pixel_count": 48320, "confidence": 0.921},
    "pancreas":    {"detected": true,  "pixel_count": 8140,  "confidence": 0.847},
    "kidney_left": {"detected": true,  "pixel_count": 19200, "confidence": 0.903},
    "spleen":      {"detected": true,  "pixel_count": 14800, "confidence": 0.889}
  },
  "anomalies": [
    {
      "type": "potential_mass",
      "pixel_count": 412,
      "severity": "moderate",
      "confidence": 0.762
    }
  ],
  "model": "unet_resnet34",
  "device": "cuda:0"
}
```

---

## Step 2 — Measurement

```python
agent.segmentation_model.measure_anomalies(segmentation_output)
```

```json
{
  "liver":                  {"estimated_volume_cm3": 1520.4, "estimated_area_mm2": 23681.0},
  "pancreas":               {"estimated_volume_cm3": 39.2,   "estimated_area_mm2": 3988.6},
  "kidney_left":            {"estimated_volume_cm3": 94.0,   "estimated_area_mm2": 9408.0},
  "anomaly_potential_mass": {"estimated_diameter_mm": 14.2,  "severity": "moderate"}
}
```

---

## Step 3 — RAG: Retrieved Evidence

Query sent to pipeline:
```
"14mm hepatic lesion elevated liver enzymes CT differential diagnosis"
```

Retrieved documents (RRF scores):

```
[0.0323] PMID 37182930
  Title: "Automated liver volumetry from CT: clinical validation across 1,200 patients"
  Key fact: "Normal liver volume ranges from 1,200 to 1,800 cm³ in adults."

[0.0241] ACR Guideline
  Title: "ACR Appropriateness Criteria: Liver Lesion Characterization"
  Key fact: "For incidental hepatic lesions >1cm, contrast-enhanced MRI is usually appropriate."

[0.0183] PMID 38104872
  Title: "AI-assisted differential diagnosis in abdominal CT: a prospective study"
  Key fact: "AI-assisted differential reduced diagnostic errors by 31% in complex multi-organ cases."
```

---

## Step 4 — Normal Range Comparison

```json
{
  "liver":       {"status": "normal",       "measured_cm3": 1520.4, "range": [1200, 1800]},
  "pancreas":    {"status": "below_normal", "measured_cm3": 39.2,   "range": [60, 120],
                  "note": "likely single-slice partial-volume artifact"},
  "kidney_left": {"status": "below_normal", "measured_cm3": 94.0,   "range": [120, 200],
                  "note": "renal volume requires multi-slice DICOM series"}
}
```

---

## Step 5 — Agent Reasoning Chain

```
Iteration 1: "I need visual findings first." → segment_scan
Iteration 2: "Anomaly found. Size before retrieval." → measure_anomalies
Iteration 3: "14mm lesion + elevated LFTs. Specific query." → retrieve_medical_knowledge
Iteration 4: "Have literature. Compare volumes to confirm normal liver." → compare_to_normals
Iteration 5: "All data gathered. Generating structured report."
             Note to self: "Pancreas/kidney below-normal are single-slice artifacts.
             Flag them but don't lead with them."
```

Total iterations: **5 of 6 max**
Total latency: ~18 seconds (cloud LLM + CPU segmentation)

---

## Final Output

```json
{
  "findings": [
    {
      "organ": "liver",
      "volume_cm3": 1520.4,
      "status": "normal_volume",
      "anomaly": {
        "type": "potential_hepatic_mass",
        "estimated_diameter_mm": 14.2,
        "severity": "moderate",
        "segmentation_confidence": 0.762
      }
    }
  ],
  "differential_diagnosis": [
    {
      "rank": 1,
      "diagnosis": "Benign hepatic hemangioma",
      "confidence": 0.58,
      "evidence": "14mm lesion, normal liver volume, no prior liver disease. Most common benign lesion."
    },
    {
      "rank": 2,
      "diagnosis": "Simple hepatic cyst",
      "confidence": 0.22,
      "evidence": "Cannot differentiate cyst from solid lesion on non-contrast CT."
    },
    {
      "rank": 3,
      "diagnosis": "HCC or metastasis — low probability, cannot exclude without imaging",
      "confidence": 0.12,
      "evidence": "Elevated LFTs + incidental mass require exclusion via contrast-enhanced MRI."
    }
  ],
  "confidence_scores": {
    "segmentation": 0.89,
    "retrieval": 0.85,
    "diagnosis_overall": 0.71,
    "calibration_note": "Uncalibrated softmax — treat as relative ranking, not absolute probability"
  },
  "recommended_followup": [
    "Contrast-enhanced MRI liver (ACR criteria: lesion >1cm warrants characterization)",
    "Hepatology referral — elevated LFTs + incidental finding",
    "Do not interpret pancreas/kidney measurements from this single slice"
  ],
  "metadata": {
    "generated_at": "2026-05-30T10:22:14Z",
    "model": "medivision-v1.0",
    "agent_iterations": 5,
    "rag_sources_used": 3
  }
}
```

---

## Limitations of This Case

- Scan is a **synthetic test image**, not a real patient CT
- The knowledge base has 5 documents — a real system would retrieve from PubMed 25M+
- Confidence scores are **not clinically calibrated**
- This output was **not reviewed by a medical professional**
- The agent correctly flagged single-slice measurement limitations — but in a real system,
  the entire pipeline should refuse to estimate renal/pancreatic volumes from a single slice
