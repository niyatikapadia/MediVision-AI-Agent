"""
MediVision Demo — Local Web App v2
Full pipeline: segmentation → RAG → Ollama (llama3) → structured report
Run: python app.py  →  http://localhost:7860
"""

import json, time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.segmentation import SegmentationModel
from src.rag_module import MedicalRAG
from src.agent import MediVisionAgent
from src.visualize import draw_segmentation_overlay

print("Loading segmentation model...")
seg_model = SegmentationModel(
    checkpoint="models/unet_resnet34_synapse_best.pth",
    device="cpu"
)
print("Loading RAG pipeline...")
rag = MedicalRAG()
print("Connecting to Ollama...")
agent = MediVisionAgent(rag=rag, ollama_model="llama3")
print("Ready.\n")


def run_pipeline(image, clinical_notes, progress=gr.Progress()):
    if image is None:
        return None, "⬆️ Please upload a CT scan image first.", "[]", "No image provided."

    progress(0.1, desc="🔬 Running organ segmentation...")
    t0 = time.time()

    seg_output   = seg_model.run(image)
    overlay_img  = draw_segmentation_overlay(image, seg_output)
    measurements = seg_model.measure(seg_output)
    normals      = seg_model.compare_to_normals(measurements)

    progress(0.45, desc="📚 Retrieving medical literature...")
    organs_found = list(seg_output.get("organ_masks", {}).keys())
    anomalies    = seg_output.get("anomalies", [])
    query_parts  = organs_found[:3] + [clinical_notes or ""]
    rag_results  = rag.search(" ".join(query_parts), top_k=3)

    progress(0.65, desc="🤖 Agent reasoning via Llama3 (30-60s)...")
    report = agent.reason(
        seg_output=seg_output, measurements=measurements,
        normals=normals, rag_results=rag_results,
        clinical_notes=clinical_notes or "No clinical notes provided.",
    )

    elapsed = time.time() - t0
    progress(1.0, desc=f"✅ Done in {elapsed:.1f}s")

    findings_md = _format_findings(seg_output, measurements, normals, elapsed)
    rag_json    = json.dumps(
        [{"title": r["title"], "source": r.get("source",""),
          "abstract": r.get("abstract","")[:120]+"..."}
         for r in rag_results], indent=2)
    report_text = report.get("report_text", "No report generated.")

    return overlay_img, findings_md, rag_json, report_text


def _format_findings(seg_output, measurements, normals, elapsed):
    organs    = seg_output.get("organ_masks", {})
    anomalies = seg_output.get("anomalies", [])

    if not organs:
        return (
            "### No organs detected\n\n"
            "The model did not detect any organs in this image.\n\n"
            "**Try uploading a real CT scan** — the sample synthetic images "
            "have very low contrast and may not produce good segmentation.\n\n"
            f"*Pipeline ran in {elapsed:.1f}s*"
        )

    lines = [f"### Detected organs ({len(organs)}) — {elapsed:.1f}s\n"]
    for organ, data in organs.items():
        conf   = data.get("confidence", 0)
        vol    = measurements.get(organ, {}).get("estimated_volume_cm3", "—")
        status = normals.get(organ, {}).get("status", "—")
        emoji  = "✅" if status == "normal" else ("⚠️" if status != "—" else "🔵")
        lines.append(f"{emoji} **{organ}** | conf: {conf:.2f} | vol: {vol} cm³ | {status}")

    if anomalies:
        lines.append("\n### ⚠️ Anomalies flagged\n")
        for a in anomalies:
            lines.append(f"- **{a['type']}** | severity: {a['severity']} | conf: {a['confidence']:.2f}")

    lines.append(
        "\n---\n"
        "*⚠️ Research prototype. Not for clinical use. "
        "Single-slice volume estimates only. Requires expert radiologist review.*"
    )
    return "\n".join(lines)


# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(title="MediVision AI Agent") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px">
      <h1 style="font-size:2em;margin:0">🧠 MediVision AI Agent</h1>
      <p style="color:#888;margin:6px 0 0;font-size:1em">
        UNet-ResNet34 segmentation (Dice 0.776) &nbsp;·&nbsp;
        Medical RAG &nbsp;·&nbsp; Llama3 local reasoning
      </p>
    </div>
    """)

    gr.HTML("""
    <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;
                padding:10px 16px;margin:0 0 16px;font-size:0.9em">
      ⚠️ <strong>Research prototype — not for clinical use.</strong>
      All outputs require expert radiologist review.
      Agent reasoning takes <strong>30–90 seconds</strong> locally — this is normal.
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            image_input = gr.Image(
                label="CT Scan Slice",
                type="pil",
                height=280,
            )
            clinical_notes = gr.Textbox(
                label="Clinical Notes (optional)",
                placeholder="e.g. 58yo male, elevated LFTs, abdominal discomfort",
                lines=2,
            )
            run_btn = gr.Button(
                "▶ Run Full Pipeline",
                variant="primary",
                size="lg"
            )
            gr.Markdown(
                "**Tip:** Use a real CT scan PNG for best results.\n\n"
                "Sample images in `data/sample_data/` work but may show "
                "few organs due to low contrast."
            )

        with gr.Column(scale=2):
            gr.Markdown("### 🖼 Segmentation Overlay")
            overlay_output = gr.Image(
                label="Organ masks overlaid on scan",
                height=280
            )
            with gr.Tabs():
                with gr.Tab("📋 Findings"):
                    findings_output = gr.Markdown(
                        value="*Upload an image and click Run to see findings.*"
                    )
                with gr.Tab("📚 Retrieved Evidence"):
                    rag_output = gr.Code(
                        language="json",
                        label="Top medical literature retrieved"
                    )
                with gr.Tab("📄 Agent Report"):
                    report_output = gr.Textbox(
                        label="Llama3 diagnostic report",
                        lines=14,
                        placeholder="Report will appear here after pipeline runs..."
                    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, clinical_notes],
        outputs=[overlay_output, findings_output, rag_output, report_output],
    )

    gr.Markdown("""---
**How it works:**
1. **Segmentation** — UNet-ResNet34 trained on Synapse CT dataset (Test Dice 0.776, 150 epochs on Kaggle P100)
2. **RAG** — BM25 keyword retrieval over medical literature knowledge base
3. **Agent** — Llama3 7B via Ollama (local, fully offline, no API key)

Built by [Niyati Kapadia](https://niyatinikunjkapadia.wixsite.com/portfolio) · 
[GitHub](https://github.com/niyatikapadia/MediVision-AI-Agent) · 
[LinkedIn](https://www.linkedin.com/in/niyati-nikunj-k-ab47861a4/)
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
    )
