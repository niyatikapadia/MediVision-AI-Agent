"""
MediVision Demo — Local Web App v3
Full pipeline: segmentation → RAG (organ-aware) → Ollama → structured report
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


def run_pipeline(image, clinical_notes):
    if image is None:
        return None, "⬆️ Please upload a CT scan image first.", "[]", "No image provided."

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
        seg_output=seg_output, measurements=measurements,
        normals=normals, rag_results=rag_results,
        clinical_notes=clinical_notes or "No clinical notes provided.",
    )

    elapsed = time.time() - t0

    findings_md = _format_findings(seg_output, measurements, normals, elapsed)
    rag_json    = json.dumps([{
        "title":        r["title"],
        "source":       r.get("source",""),
        "match_reason": r.get("match_reason",""),
        "abstract":     r.get("abstract","")[:120]+"..."
    } for r in rag_results], indent=2)
    report_text = report.get("report_text", "No report generated.")

    return overlay_img, findings_md, rag_json, report_text


def _format_findings(seg_output, measurements, normals, elapsed):
    organs    = seg_output.get("organ_masks", {})
    anomalies = seg_output.get("anomalies", [])

    if not organs:
        return (
            "### No organs detected\n\n"
            "Upload a real abdominal CT scan slice (grayscale, axial view).\n\n"
            f"*Ran in {elapsed:.1f}s*"
        )

    lines = [
        f"### {len(organs)} organs detected &nbsp; · &nbsp; {elapsed:.1f}s\n",
        "> ⚠️ **Single-slice analysis only.** "
        "Coverage % shown — volume requires full DICOM series.\n"
    ]

    for organ, data in organs.items():
        conf     = data.get("confidence", 0)
        coverage = measurements.get(organ, {}).get("coverage_pct", 0)
        status   = normals.get(organ, {}).get("status", "—")
        exp      = normals.get(organ, {}).get("expected_range", "—")

        if status == "within_expected_range":
            emoji = "✅"
        elif status in ("smaller_than_expected","larger_than_expected"):
            emoji = "⚠️"
        else:
            emoji = "🔵"

        lines.append(
            f"{emoji} **{organ}** &nbsp; "
            f"conf: `{conf:.2f}` &nbsp; "
            f"coverage: `{coverage:.1f}%` &nbsp; "
            f"expected: `{exp}` &nbsp; "
            f"*{status}*"
        )

    if anomalies:
        lines.append("\n### ⚠️ Flagged for review\n")
        for a in anomalies:
            lines.append(f"- **{a['type']}** | conf: {a['confidence']:.2f}")

    lines.append(
        "\n---\n"
        "*Research prototype — not for clinical use. "
        "Expert radiologist review required.*"
    )
    return "\n".join(lines)


with gr.Blocks(title="MediVision AI Agent") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:20px 0 6px">
      <h1 style="font-size:1.9em;margin:0">🧠 MediVision AI Agent</h1>
      <p style="color:#888;margin:6px 0 0">
        UNet-ResNet34 · Synapse CT · Test Dice 0.776 &nbsp;|&nbsp;
        Medical RAG · 12 documents &nbsp;|&nbsp;
        Llama3 · fully local
      </p>
    </div>
    """)

    gr.HTML("""
    <div style="background:#fff3cd;border-left:4px solid #ffc107;
                padding:10px 16px;margin:0 0 14px;font-size:0.88em">
      ⚠️ <strong>Research prototype.</strong>
      Not validated for clinical use. Single 2D slice analysis only.
      Volume measurements require full DICOM series.
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
                placeholder="e.g. 58yo male, elevated LFTs, 3 weeks abdominal discomfort",
                lines=2,
            )
            run_btn = gr.Button("▶ Run Full Pipeline", variant="primary", size="lg")
            gr.Markdown(
                "**What to upload:** A real abdominal CT scan axial slice.\n\n"
                "**Not:** Photos, MRI, or chest CT.\n\n"
                "**Agent reasoning takes 30–90s** — this is normal for local Llama3."
            )

        with gr.Column(scale=2):
            gr.Markdown("### Segmentation Overlay")
            overlay_output = gr.Image(
                label="Organ masks — coloured regions = detected organs",
                height=330
            )
            with gr.Tabs():
                with gr.Tab("📋 Findings"):
                    findings_output = gr.Markdown(
                        value="*Upload a CT scan and click Run.*"
                    )
                with gr.Tab("📚 Retrieved Evidence"):
                    rag_output = gr.Code(language="json",
                        label="Organ-matched medical literature")
                with gr.Tab("📄 Agent Report"):
                    report_output = gr.Textbox(
                        label="Llama3 structured diagnostic report",
                        lines=16,
                        placeholder="Report appears here after pipeline runs..."
                    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, clinical_notes],
        outputs=[overlay_output, findings_output, rag_output, report_output],
    )

    gr.Markdown("""---
**Pipeline:** Segmentation (UNet-ResNet34, Dice 0.776 on Synapse benchmark)
→ RAG (BM25, 12 medical documents, organ-aware routing)
→ Agent (Llama3 7B via Ollama, fully local, no API key)

**Honest limitations:** Single 2D slice · Small RAG KB · Volume requires DICOM · Not clinically validated

Built by [Niyati Kapadia](https://niyatinikunjkapadia.wixsite.com/portfolio) ·
[GitHub](https://github.com/niyatikapadia/MediVision-AI-Agent) ·
[LinkedIn](https://www.linkedin.com/in/niyati-nikunj-k-ab47861a4/)
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,
                inbrowser=True, share=False)
