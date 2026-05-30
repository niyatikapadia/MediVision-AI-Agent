# Example Case 2 — Normal Baseline Scan

This case demonstrates the agent correctly identifying a normal study
and skipping unnecessary tool calls.

---

## Input

```
Scan:           ct_normal_baseline.png
Clinical notes: "35-year-old female. Annual screening. No symptoms."
LLM backend:    claude-sonnet-4-20250514
```

---

## Agent Execution Trace

```
[ITERATION 1]
Tool call: segment_scan("data/sample_data/ct_normal_baseline.png")

Returns:
{
  "organ_masks": {
    "liver":        {"detected": true, "pixel_count": 44100, "confidence": 0.918},
    "pancreas":     {"detected": true, "pixel_count": 7920,  "confidence": 0.851},
    "kidney_left":  {"detected": true, "pixel_count": 20400, "confidence": 0.907},
    "kidney_right": {"detected": true, "pixel_count": 19800, "confidence": 0.899},
    "spleen":       {"detected": true, "pixel_count": 15200, "confidence": 0.893}
  },
  "anomalies": []   ← no anomalies flagged
}

────────────────────────────────────────────────────────────────
[ITERATION 2]
Tool call: measure_anomalies(segmentation_output)
Tool call: compare_to_normals(measurements, patient_age=35, sex="female")

Normal range comparison:
{
  "liver":        {"status": "normal", "deviation_pct":  2.1},
  "kidney_left":  {"status": "normal", "deviation_pct": -3.4},
  "kidney_right": {"status": "normal", "deviation_pct": -2.8},
  "spleen":       {"status": "normal", "deviation_pct":  1.7}
}

────────────────────────────────────────────────────────────────
[ITERATION 3]
Agent decides: skip retrieve_medical_knowledge

Reasoning: "All organs within normal ranges. No anomalies detected.
            Literature retrieval would not change the assessment.
            Generating normal study report."
```

**Total iterations: 3 of 6 max** (agent stopped early — correct behavior)

---

## Output

```json
{
  "findings": [
    {"organ": "liver",        "status": "normal", "confidence": 0.918},
    {"organ": "pancreas",     "status": "normal", "confidence": 0.851},
    {"organ": "kidney_left",  "status": "normal", "confidence": 0.907},
    {"organ": "kidney_right", "status": "normal", "confidence": 0.899},
    {"organ": "spleen",       "status": "normal", "confidence": 0.893}
  ],
  "differential_diagnosis": [],
  "confidence_scores": {
    "segmentation": 0.914,
    "diagnosis_overall": 0.91
  },
  "recommended_followup": [
    "No acute findings. Routine screening interval per age/risk profile."
  ]
}
```

---

## Why This Case Matters

Most "agent" demos only show the complex case. Showing a normal baseline demonstrates:

1. **The agent doesn't hallucinate findings** when there are none
2. **Tool-use is conditional** — the agent skips retrieval when it isn't needed
3. **The system produces a useful output** in the simple case too (not just error or "no findings")

Correctly handling the normal case is as important as correctly handling pathology.
