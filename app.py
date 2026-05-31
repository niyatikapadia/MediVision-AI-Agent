"""
MediVision Demo — Local Web App
Full pipeline: segmentation → RAG → Ollama agent → structured report

Run:
    python app.py

Then open http://localhost:7860
"""

import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.segmentation import SegmentationModel
from src.rag import MedicalRAG
from src.agent import MediVisionAgent
from src.visualize import draw_segmentation_overlay

# ── Load models once at startup ───────────────────────────────
print("Loading segmentation model...")
seg_model = SegmentationModel(
    checkpoint="models/unet_resnet34_synapse_best.pth",
    device="auto"
)
print("Loading RAG pipeline...")
rag = MedicalRAG()
print("Connecting to Ollama...")
agent = MediVisionAgent(rag=rag, ollama_model="mistral")
print("Ready.\n")

# ── Class colors for overlay ──────────────────────────────────
CLASS_COLORS = {
    0: (0,   0,   0,   0),    # background — transparent
    1: (255, 100, 100, 180),  # aorta — red
    2: (100, 255, 100, 180),  # gallbladder — green
    3: (100, 100, 255, 180),  # spleen — blue
    4: (255, 255, 100, 180),  # left kidney — yellow
    5: (255, 165, 0,   180),  # right kidney — orange
    6: (200, 100, 200, 180),  # liver — purple
    7: (100, 220, 220, 180),  # stomach — cyan
    8: (255, 180, 180, 180),  # pancreas — pink
}
CLASS_NAMES = [
    "background","aorta","gallbladder","spleen",
    "left_kidney","right_kidney","liver","stomach","pancreas"
]


def run_pipeline(image, clinical_notes, progress=gr.Progress()):
    """
    Full MediVision pipeline:
    1. Segmentation
    2. Measurement
    3. RAG retrieval
    4. Agent reasoning
    5. Report generation
    """
    if image is None:
        return None, "Please upload a CT scan image.", "{}", "No findings."

    progress(0.1, desc="Running segmentation...")
    t0 = time.time()

    # ── Step 1: Segmentation ──────────────────────────────────
    seg_output = seg_model.run(image)
    overlay_img = draw_segmentation_overlay(image, seg_output, CLASS_COLORS, CLASS_NAMES)

    progress(0.35, desc="Measuring findings...")

    # ── Step 2: Measurements ─────────────────────────────────
    measurements = seg_model.measure(seg_output)
    normals      = seg_model.compare_to_normals(measurements)

    progress(0.5, desc="Retrieving medical literature...")

    # ── Step 3: RAG ───────────────────────────────────────────
    anomalies = seg_output.get("anomalies", [])
    query = _build_rag_query(normals, anomalies, clinical_notes)
    rag_results = rag.search(query, top_k=3)

    progress(0.7, desc="Agent reasoning (Ollama/Mistral)...")

    # ── Step 4: Agent reasoning ───────────────────────────────
    report = agent.reason(
        seg_output    = seg_output,
        measurements  = measurements,
        normals       = normals,
        rag_results   = rag_results,
        clinical_notes= clinical_notes or "No clinical notes provided.",
    )

    elapsed = time.time() - t0
    progress(1.0, desc=f"Done in {elapsed:.1f}s")

    # ── Format outputs ────────────────────────────────────────
    findings_md  = _format_findings(seg_output, measurements, normals)
    report_text  = report.get("report_text", "Agent did not produce a report.")
    rag_json     = json.dumps(
        [{"title": r["title"], "score": r["relevance_score"]} for r in rag_results],
        indent=2
    )

    return overlay_img, findings_md, rag_json, report_text


def _build_rag_query(normals, anomalies, clinical_notes):
    parts = []
    for organ, data in normals.items():
        if data["status"] != "normal":
            parts.append(f"{organ} {data['status']}")
    for a in anomalies:
        parts.append(f"{a['type']} CT differential")
    if clinical_notes:
        parts.append(clinical_notes[:100])
    return " ".join(parts) if parts else "abdominal CT organ segmentation findings"


def _format_findings(seg_output, measurements, normals):
    lines = ["## Segmentation Findings\n"]
    for organ, data in seg_output.get("organ_masks", {}).items():
        conf   = data.get("confidence", 0)
        vol    = measurements.get(organ, {}).get("estimated_volume_cm3", "—")
        status = normals.get(organ, {}).get("status", "—")
        emoji  = "✅" if status == "normal" else "⚠️"
        lines.append(f"{emoji} **{organ}** — conf: {conf:.2f} | vol: {vol} cm³ | {status}")
    if seg_output.get("anomalies"):
        lines.append("\n## Anomalies Detected\n")
        for a in seg_output["anomalies"]:
            diam = measurements.get(f"anomaly_{a['type']}", {}).get("estimated_diameter_mm", "—")
            lines.append(f"🔴 **{a['type']}** — ~{diam}mm | severity: {a['severity']} | conf: {a['confidence']:.2f}")
    lines.append(
        "\n---\n*⚠️ Research prototype. Not for clinical use. "
        "All findings require expert radiologist review.*"
    )
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1200px !important; }
.warning-box { background: #fff3cd; border: 1px solid #ffc107;
               border-radius: 8px; padding: 12px; margin-bottom: 16px; }
"""

with gr.Blocks(title="MediVision AI Agent", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <h1 style="font-size:2em; margin:0;">🧠 MediVision AI Agent</h1>
        <p style="color:#666; margin:6px 0 0;">
            Multimodal medical imaging analysis — UNet-ResNet34 + Medical RAG + Mistral 7B
        </p>
    </div>
    """)

    gr.HTML("""
    <div class="warning-box">
        ⚠️ <strong>Research prototype only.</strong>
        This system is not validated for clinical use and must not be used for medical decisions.
        All outputs require expert radiologist review.
    </div>
    """)

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            image_input = gr.Image(
                label="CT Scan Slice (PNG/JPG)",
                type="pil",
                height=300,
            )
            clinical_notes = gr.Textbox(
                label="Clinical Notes (optional)",
                placeholder="e.g. 58yo male, elevated LFTs, abdominal discomfort",
                lines=3,
            )
            run_btn = gr.Button("▶ Run Full Pipeline", variant="primary", size="lg")

            gr.Markdown("### Sample Inputs")
            gr.Examples(
                examples=[
                    ["data/sample_data/ct_abdomen_slice.png", "58yo male, elevated LFTs"],
                    ["data/sample_data/ct_normal_baseline.png", "35yo female, annual screening"],
                ],
                inputs=[image_input, clinical_notes],
            )

        # ── Right column: outputs ─────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Segmentation Overlay")
            overlay_output = gr.Image(label="Organ Segmentation", height=300)

            with gr.Tabs():
                with gr.Tab("📋 Findings"):
                    findings_output = gr.Markdown()

                with gr.Tab("📚 Retrieved Evidence"):
                    rag_output = gr.Code(language="json", label="RAG Results")

                with gr.Tab("📄 Agent Report"):
                    report_output = gr.Textbox(
                        label="Diagnostic Report",
                        lines=15,
                        show_copy_button=True,
                    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, clinical_notes],
        outputs=[overlay_output, findings_output, rag_output, report_output],
    )

    gr.Markdown("""
    ---
    **How it works:**
    1. UNet-ResNet34 (trained on Synapse CT dataset, Test Dice 0.776) segments organs
    2. Hybrid BM25 + BioBERT RAG retrieves relevant medical literature
    3. Mistral 7B (via Ollama, local) reasons across all findings and generates a structured report

    **Built by [Niyati Kapadia](https://niyatinikunjkapadia.wixsite.com/portfolio)**
    · [GitHub](https://github.com/niyatikapadia/MediVision-AI-Agent)
    · [LinkedIn](https://www.linkedin.com/in/niyati-nikunj-k-ab47861a4/)
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,          # set True to get a public URL
        inbrowser=True,
    )
