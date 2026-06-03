"""
MediVision Demo — Local Web App v4
Full pipeline: segmentation -> RAG -> Ollama -> structured report
Run: python app.py  ->  http://localhost:7860
"""
import json
import time

import gradio as gr
import numpy as np
from PIL import Image

from src.segmentation import SegmentationModel
from src.rag_module import MedicalRAG
from src.agent import MediVisionAgent
from src.visualize import draw_segmentation_overlay

print("Loading segmentation model...")
seg_model = SegmentationModel(
    checkpoint="models/unet_resnet34_v31_best.pth",
    device="cpu"
)
print("Loading RAG pipeline...")
rag = MedicalRAG()
print("Connecting to Ollama...")
agent = MediVisionAgent(rag=rag, ollama_model="llama3")
print("Ready.\n")


def run_pipeline(image, clinical_notes):
    if image is None:
        return None, "Please upload a CT scan image first.", "[]", "No image provided."

    t0 = time.time()

    seg_output   = seg_model.run(image)
    overlay_img  = draw_segmentation_overlay(image, seg_output)
    measurements = seg_model.measure(seg_output)
    normals      = seg_model.compare_to_normals(measurements)

    detected_organs = list(seg_output.get("organ_masks", {}).keys())
    rag_results = rag.search(
        query=" ".join(detected_organs) + " " + (clinical_notes or ""),
        detected_organs=detected_organs,
        top_k=3
    )

    report = agent.reason(
        seg_output=seg_output,
        measurements=measurements,
        normals=normals,
        rag_results=rag_results,
        clinical_notes=clinical_notes or "No clinical notes provided.",
    )

    elapsed = time.time() - t0
    findings_md = format_findings(seg_output, measurements, normals, elapsed)
    rag_json = json.dumps([{
        "title":        r["title"],
        "source":       r.get("source", ""),
        "match_reason": r.get("match_reason", ""),
        "abstract":     r.get("abstract", "")[:120] + "..."
    } for r in rag_results], indent=2)

    return overlay_img, findings_md, rag_json, report.get("report_text", "")


def format_findings(seg_output, measurements, normals, elapsed):
    organs    = seg_output.get("organ_masks", {})
    anomalies = seg_output.get("anomalies", [])

    if not organs:
        lines = [
            "### No organs detected",
            "",
            "Upload a real abdominal CT scan slice (grayscale, axial view).",
            "",
            "Ran in {:.1f}s".format(elapsed)
        ]
        return "\n".join(lines)

    # Separate valid measurements from segmentation warnings
    valid   = {}
    warned  = {}
    for organ, norm in normals.items():
        if norm.get("status") == "SEGMENTATION_WARNING":
            warned[organ] = norm
        else:
            valid[organ] = norm

    lines = [
        "### {} organs detected  |  {:.1f}s".format(len(organs), elapsed),
        "",
        "> Cross-sectional area (cm2) — single-slice CT analysis.",
        "> Whole-organ volumetry requires the full DICOM series.",
        "",
    ]

    # Show segmentation warnings first and prominently
    if warned:
        lines.append("**Segmentation Quality Warnings**")
        lines.append("")
        for organ, norm in warned.items():
            conf = organs.get(organ, {}).get("confidence", 0)
            lines.append(
                "[SEG WARNING] {} | conf: {:.2f} | {}".format(
                    organ, conf,
                    norm.get("warning", "Measurement unreliable — do not use clinically")
                )
            )
        lines.append("")
        lines.append("**Usable Measurements**")
        lines.append("")

    # Show valid measurements
    for organ, data in organs.items():
        if organ in warned:
            continue
        conf   = data.get("confidence", 0)
        norm   = normals.get(organ, {})
        status = norm.get("status", "detected")

        if "area_cm2" in norm:
            area    = norm["area_cm2"]
            rng     = norm.get("area_range_cm2", (area, area))
            ref_rng = norm.get("reference_range_cm2", [])
            level   = norm.get("anatomical_level", "")
            meas_str = "area: {} cm2 (range {}-{})".format(area, rng[0], rng[1])
            ref_str  = "ref: {}-{} cm2".format(ref_rng[0], ref_rng[1]) if ref_rng else ""
        elif "diameter_mm" in norm:
            diam    = norm["diameter_mm"]
            rng     = norm.get("diam_range_mm", (diam, diam))
            ref_rng = norm.get("reference_range_mm", [])
            meas_str = "diameter: {}mm (range {}-{}mm)".format(diam, rng[0], rng[1])
            ref_str  = "ref: {}-{}mm".format(ref_rng[0], ref_rng[1]) if ref_rng else ""
        else:
            meas_str = ""
            ref_str  = ""

        if "below_normal" in status:    label = "BELOW"
        elif "above_normal" in status:  label = "ABOVE"
        elif status == "normal":        label = "OK"
        elif "borderline" in status:    label = "BORDERLINE"
        else:                           label = "DETECTED"

        ref_note = " | {}".format(ref_str) if ref_str else ""
        lines.append(
            "[{}] {} | conf: {:.2f} | {}{}".format(
                label, organ, conf, meas_str, ref_note
            )
        )

    if anomalies:
        lines.append("")
        lines.append("**Low Confidence Flags**")
        for a in anomalies:
            lines.append("- {} | conf: {:.2f}".format(a["type"], a["confidence"]))

    lines.extend([
        "",
        "---",
        "Area estimated: FOV=370mm / 512px = 0.684mm/px, uncertainty +-25%.",
        "Segmentation warnings indicate likely model errors — not clinical findings.",
        "Expert radiologist review required."
    ])

    return "\n".join(lines)


with gr.Blocks(title="MediVision AI Agent") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:20px 0 6px">
      <h1 style="font-size:1.9em;margin:0">MediVision AI Agent</h1>
      <p style="color:#888;margin:6px 0 0">
        UNet-ResNet34 (Dice 0.776) | Medical RAG (12 docs) | Llama3 local
      </p>
    </div>
    """)

    gr.HTML("""
    <div style="background:#fff3cd;border-left:4px solid #ffc107;
                padding:10px 16px;margin:0 0 14px;font-size:0.88em">
      Research prototype. Not validated for clinical use.
      Single 2D slice analysis. Cross-sectional area estimated (+-25%).
      All outputs require expert radiologist review.
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload CT Slice")
            image_input = gr.Image(
                label="Abdominal CT axial slice (PNG/JPG)",
                type="pil",
                height=300,
            )
            clinical_notes = gr.Textbox(
                label="Clinical Notes (optional)",
                placeholder="e.g. 58yo male, elevated LFTs, abdominal discomfort",
                lines=2,
            )
            run_btn = gr.Button("Run Full Pipeline", variant="primary", size="lg")
            gr.Markdown(
                "Upload a real abdominal CT axial slice.\n\n"
                "Agent reasoning takes 30-90s locally (Llama3 7B)."
            )

        with gr.Column(scale=2):
            gr.Markdown("### Segmentation Overlay")
            overlay_output = gr.Image(
                label="Organ masks — colours = organs, numbers = confidence",
                height=340
            )
            with gr.Tabs():
                with gr.Tab("Findings"):
                    findings_output = gr.Markdown(value="Upload a CT scan and click Run.")
                with gr.Tab("Retrieved Evidence"):
                    rag_output = gr.Code(
                        language="json",
                        label="Organ-matched medical literature"
                    )
                with gr.Tab("Agent Report"):
                    report_output = gr.Textbox(
                        label="Llama3 structured diagnostic report",
                        lines=16,
                        placeholder="Report appears after pipeline runs..."
                    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, clinical_notes],
        outputs=[overlay_output, findings_output, rag_output, report_output],
    )

    gr.Markdown("""---
Pipeline: Segmentation (UNet-ResNet34, Dice 0.776) -> RAG (BM25, 12 docs, organ-aware) -> Agent (Llama3 via Ollama, local)

Volume methodology: Pixel spacing estimated from standard abdominal CT FOV (370mm/512px). Uncertainty +-25%. Provide DICOM for definitive measurements.

Built by [Niyati Kapadia](https://niyatinikunjkapadia.wixsite.com/portfolio) |
[GitHub](https://github.com/niyatikapadia/MediVision-AI-Agent) |
[LinkedIn](https://www.linkedin.com/in/niyati-nikunj-k-ab47861a4/)
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
    )
