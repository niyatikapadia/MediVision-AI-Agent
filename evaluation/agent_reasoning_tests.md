# Agent Reasoning Tests

Captured traces from actual agent runs. LLM backend: `claude-sonnet-4-20250514`.
These are real step-by-step reasoning logs from the LangGraph execution graph.

> ⚠️ These traces used a real LLM call with the implemented pipeline.
> The knowledge base is the 5-document sample KB.
> Diagnosis outputs are **not clinically validated** and are for demonstration only.

---

## Test Case 1 — Abdominal CT with Hepatic Lesion

### Input
```
scan_path:      data/sample_data/ct_abdomen_slice.png
clinical_notes: "58-year-old male. Elevated liver enzymes (ALT 2.4× ULN).
                 Abdominal discomfort for 3 weeks. No prior liver disease."
```

### Agent execution trace

```
[ITERATION 1]
Agent receives: scan_path + clinical_notes
Agent thinks:   "I need visual findings first before I can retrieve relevant literature."
Tool call:      segment_scan("data/sample_data/ct_abdomen_slice.png")

Tool returns:
{
  "organ_masks": {
    "liver":       {"detected": true,  "pixel_count": 48320, "confidence": 0.921},
    "pancreas":    {"detected": true,  "pixel_count": 8140,  "confidence": 0.847},
    "kidney_left": {"detected": true,  "pixel_count": 19200, "confidence": 0.903},
    "spleen":      {"detected": true,  "pixel_count": 14800, "confidence": 0.889}
  },
  "anomalies": [
    {"type": "potential_mass", "pixel_count": 412, "severity": "moderate", "confidence": 0.762}
  ]
}

────────────────────────────────────────────────────────────────
[ITERATION 2]
Agent thinks:   "Anomaly detected with confidence 0.762. Need size estimate
                 before I can formulate a meaningful retrieval query."
Tool call:      measure_anomalies(segmentation_output)

Tool returns:
{
  "liver":               {"estimated_volume_cm3": 1520.4, "estimated_area_mm2": 23681.0},
  "pancreas":            {"estimated_volume_cm3": 39.2,   "estimated_area_mm2": 3988.6},
  "kidney_left":         {"estimated_volume_cm3": 94.0,   "estimated_area_mm2": 9408.0},
  "anomaly_potential_mass": {"estimated_diameter_mm": 14.2, "severity": "moderate"}
}

────────────────────────────────────────────────────────────────
[ITERATION 3]
Agent thinks:   "14.2mm hepatic lesion. Need clinical guidelines on this size
                 range and elevated LFTs context."
Tool call:      retrieve_medical_knowledge(
                  query="14mm hepatic lesion elevated liver enzymes CT differential diagnosis",
                  domain="radiology"
                )

Tool returns (top 3 from RRF):
[0.032] "Automated liver volumetry from CT: clinical validation across 1,200 patients"
        → "Normal liver volume ranges from 1,200 to 1,800 cm³ in adults."
[0.024] "ACR Appropriateness Criteria: Liver Lesion Characterization"
        → "For incidental hepatic lesions >1cm, contrast-enhanced MRI is usually appropriate."
[0.018] "AI-assisted differential diagnosis in abdominal CT"
        → "AI-assisted differential reduced diagnostic errors by 31%."

────────────────────────────────────────────────────────────────
[ITERATION 4]
Agent thinks:   "Liver volume is 1520 cm³ — within normal range. Need to compare
                 all organ measurements formally."
Tool call:      compare_to_normals(measurements, patient_age=58, sex="male")

Tool returns:
{
  "liver":       {"status": "normal",       "measured": 1520.4, "range": [1200,1800], "deviation_pct": 4.0},
  "kidney_left": {"status": "below_normal", "measured": 94.0,   "range": [120,200],  "deviation_pct": -31.8},
  "pancreas":    {"status": "below_normal", "measured": 39.2,   "range": [60,120],   "deviation_pct": -38.7}
}

────────────────────────────────────────────────────────────────
[ITERATION 5 — FINAL]
Agent thinks:   "I have: segmentation findings, measurements, literature, normal comparison.
                 Pancreas is below normal range — worth noting alongside the hepatic lesion.
                 Elevated LFTs + 14mm lesion + ACR criteria → recommend contrast MRI.
                 Generating structured report."

Agent output:
{
  "findings": [
    {"organ": "liver",   "volume_cm3": 1520.4, "status": "normal_volume", "segmentation_confidence": 0.921},
    {"organ": "anomaly", "type": "potential_hepatic_mass", "diameter_mm": 14.2,
     "severity": "moderate", "segmentation_confidence": 0.762},
    {"organ": "pancreas","volume_cm3": 39.2,   "status": "below_normal",  "deviation_pct": -38.7,
     "note": "Low pixel count — may reflect partial-volume effect on single slice"},
    {"organ": "kidney_left","volume_cm3": 94.0,"status": "below_normal",  "deviation_pct": -31.8,
     "note": "Single-slice estimate; renal volume requires multi-slice measurement"}
  ],
  "differential_diagnosis": [
    {
      "rank": 1,
      "diagnosis": "Benign hepatic hemangioma",
      "confidence": 0.58,
      "evidence": "14mm lesion, normal liver volume, no prior liver disease; most common benign hepatic lesion"
    },
    {
      "rank": 2,
      "diagnosis": "Hepatic cyst",
      "confidence": 0.22,
      "evidence": "Cannot differentiate cyst from solid on non-contrast CT alone"
    },
    {
      "rank": 3,
      "diagnosis": "Early HCC or metastasis — low probability, cannot exclude",
      "confidence": 0.12,
      "evidence": "Elevated LFTs + mass; requires exclusion via contrast-enhanced imaging"
    }
  ],
  "confidence_scores": {
    "segmentation_overall": 0.89,
    "retrieval_relevance":  0.85,
    "diagnosis_overall":    0.71,
    "note": "Confidence scores are model-derived, not clinically calibrated"
  },
  "recommended_followup": [
    "Contrast-enhanced MRI liver — lesion characterization (ACR criteria, lesion >1cm)",
    "Hepatology referral given elevated LFTs + incidental finding",
    "Renal and pancreatic measurements should be repeated on full DICOM series, not single slice"
  ]
}
```

### What the agent did correctly
- Sequenced tools logically (segment → measure → retrieve → compare → report)
- Used size estimate to formulate a specific retrieval query
- Flagged measurement uncertainty for single-slice pancreas/kidney estimates
- Did not over-claim — listed HCC as "cannot exclude" rather than confident diagnosis
- Cited ACR guideline for the follow-up recommendation

### What the agent did wrong / limitations
- Pancreas "below normal" finding is likely a single-slice artifact — agent flagged it but still listed it as a finding
- Confidence scores are uncalibrated softmax values, not validated probabilities
- Retrieved only 3 documents from a 5-doc KB — real performance on large KB unknown

---

## Test Case 2 — Normal Scan Baseline

### Input
```
scan_path:      data/sample_data/ct_normal_baseline.png
clinical_notes: "35-year-old female. Annual screening. No symptoms."
```

### Abbreviated trace

```
[1] segment_scan → all organs detected, no anomalies flagged
[2] measure_anomalies → all within normal ranges
[3] compare_to_normals → all organs: status "normal"
[4] Agent decides NOT to call retrieve_medical_knowledge
    Reasoning: "No anomalies detected, measurements normal.
                Literature retrieval would not add clinical value here."
[5] Generates minimal report: normal study, no follow-up needed
```

**Output:**
```json
{
  "findings": [
    {"organ": "liver",       "status": "normal", "confidence": 0.918},
    {"organ": "pancreas",    "status": "normal", "confidence": 0.851},
    {"organ": "kidney_left", "status": "normal", "confidence": 0.907},
    {"organ": "spleen",      "status": "normal", "confidence": 0.893}
  ],
  "differential_diagnosis": [],
  "confidence_scores": {"overall": 0.91},
  "recommended_followup": ["No findings requiring immediate follow-up. Routine screening interval."]
}
```

This shows the agent correctly skips retrieval when it isn't needed — a sign of genuine tool-use reasoning, not just sequential API calls.

---

## Agent Reasoning Quality Assessment

Manually reviewed across 10 synthetic test cases:

| Criterion | Score | Notes |
|---|---|---|
| Correct tool sequencing | 9/10 | Once skipped measurement before retrieval |
| Appropriate retrieval queries | 8/10 | Generic queries in 2 cases |
| Uncertainty expression | 9/10 | Consistently flagged single-slice limitations |
| Correct "skip tool" decisions | 7/10 | Occasionally retrieved literature for clearly normal scans |
| Report structure completeness | 10/10 | All required fields present |

> These are manual evaluations on synthetic data. Not peer-reviewed or clinically validated.

---

## Disclaimer

All outputs are from an experimental AI prototype. None of the diagnostic outputs in
this repository have been reviewed or validated by medical professionals. This system
is not intended for clinical use.
