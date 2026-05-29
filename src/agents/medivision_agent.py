"""
MediVision AI Agent — Main agentic reasoning loop using LangGraph.

Orchestrates vision analysis, RAG retrieval, and LLM reasoning
into a multi-step clinical diagnostic pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.rag.pipeline import MedicalRAGPipeline
from src.vision.segmentation import SegmentationModel
from src.utils.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are MediVision, an expert AI medical imaging analyst.

You have access to the following tools:
- segment_scan: Run organ segmentation on a CT/MRI scan image
- retrieve_medical_knowledge: Search PubMed and clinical guidelines
- measure_anomalies: Quantify detected anomalies from segmentation output
- compare_to_normals: Compare measurements against population norms

Your task: analyze the provided medical imaging data, retrieve relevant clinical
knowledge, and produce a structured differential diagnosis with evidence chains.

ALWAYS:
1. Start by segmenting the scan to extract visual findings
2. Retrieve relevant medical literature for each finding
3. Cross-reference with clinical notes
4. Reason step-by-step before producing a diagnosis
5. Cite your evidence sources
6. Express uncertainty clearly with confidence scores

Respond in structured JSON for the final report.
"""


@dataclass
class AnalysisResult:
    """Structured output from the MediVision agent."""
    findings: list[dict]
    differential_diagnosis: list[dict]
    confidence_scores: dict[str, float]
    evidence_chain: list[str]
    recommended_followup: list[str]
    report_text: str
    metadata: dict = field(default_factory=dict)

    def save_report(self, path: str) -> None:
        output = {
            "findings": self.findings,
            "differential_diagnosis": self.differential_diagnosis,
            "confidence_scores": self.confidence_scores,
            "evidence_chain": self.evidence_chain,
            "recommended_followup": self.recommended_followup,
            "report_text": self.report_text,
            "metadata": self.metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Report saved to {path}")


class AgentState(TypedDict):
    """State passed through the LangGraph node graph."""
    messages: list[BaseMessage]
    scan_path: str
    report_pdf: str | None
    clinical_notes: str | None
    segmentation_output: dict | None
    rag_results: list[dict]
    final_result: AnalysisResult | None
    iteration_count: int


class MediVisionAgent:
    """
    End-to-end multimodal medical AI agent.

    Combines:
    - UNet-ResNet34 vision segmentation
    - BioBERT-powered medical RAG
    - LangGraph agentic reasoning loop
    - Structured FHIR-compatible report generation
    """

    MAX_ITERATIONS = 6

    def __init__(
        self,
        llm_backend: str = "local",
        segmentation_model: str = "unet_resnet34",
        rag_top_k: int = 5,
        device: str = "auto",
    ):
        self.llm_backend = llm_backend
        self.rag_top_k = rag_top_k

        logger.info(f"Initializing MediVision agent — LLM: {llm_backend}, device: {device}")

        self.segmentation_model = SegmentationModel(
            model_name=segmentation_model,
            device=device,
        )
        self.rag_pipeline = MedicalRAGPipeline(top_k=rag_top_k)
        self.report_generator = ReportGenerator()
        self.llm = self._init_llm()
        self.graph = self._build_graph()

    def _init_llm(self):
        """Initialize the LLM backbone based on selected backend."""
        if self.llm_backend == "claude":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.1)
        elif self.llm_backend == "gpt4o":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.1)
        else:
            # Local fallback — uses Ollama with BioMistral
            from langchain_ollama import ChatOllama
            return ChatOllama(model="biomistral", temperature=0.1)

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph agentic reasoning graph."""

        tools = self._get_tools()
        llm_with_tools = self.llm.bind_tools(tools)
        tool_node = ToolNode(tools)

        def call_agent(state: AgentState) -> AgentState:
            """Main agent reasoning node."""
            messages = state["messages"]
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

            response = llm_with_tools.invoke(messages)
            state["messages"] = messages + [response]
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            return state

        def should_continue(state: AgentState) -> str:
            """Decide whether to continue reasoning or finalize the report."""
            last_message = state["messages"][-1]
            if state.get("iteration_count", 0) >= self.MAX_ITERATIONS:
                return "finalize"
            if not isinstance(last_message, AIMessage):
                return "finalize"
            if last_message.tool_calls:
                return "tools"
            return "finalize"

        def finalize(state: AgentState) -> AgentState:
            """Extract structured result from agent messages."""
            last_content = state["messages"][-1].content
            result = self.report_generator.parse_agent_output(
                agent_output=last_content,
                segmentation_output=state.get("segmentation_output"),
                rag_results=state.get("rag_results", []),
            )
            state["final_result"] = result
            return state

        graph = StateGraph(AgentState)
        graph.add_node("agent", call_agent)
        graph.add_node("tools", tool_node)
        graph.add_node("finalize", finalize)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue, {
            "tools": "tools",
            "finalize": "finalize",
        })
        graph.add_edge("tools", "agent")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _get_tools(self) -> list:
        """Register all tools available to the agent."""
        seg_model = self.segmentation_model
        rag = self.rag_pipeline

        @tool
        def segment_scan(scan_path: str) -> dict:
            """
            Run organ segmentation on a medical scan image.
            Returns detected organs, bounding boxes, and anomaly flags.

            Args:
                scan_path: Path to the CT/MRI scan (PNG, DICOM, or NIfTI)
            """
            return seg_model.run(scan_path)

        @tool
        def retrieve_medical_knowledge(query: str, domain: str = "radiology") -> list[dict]:
            """
            Search PubMed abstracts and clinical guidelines for relevant evidence.

            Args:
                query: Clinical question or finding description
                domain: One of 'radiology', 'oncology', 'cardiology', 'nephrology'
            """
            return rag.search(query, domain=domain)

        @tool
        def measure_anomalies(segmentation_output: dict) -> dict:
            """
            Quantify anomalies detected in segmentation output.
            Returns size measurements, shape descriptors, and density metrics.

            Args:
                segmentation_output: Output dict from segment_scan tool
            """
            return seg_model.measure_anomalies(segmentation_output)

        @tool
        def compare_to_normals(measurements: dict, patient_age: int = 50, sex: str = "unknown") -> dict:
            """
            Compare organ measurements against population normal ranges.

            Args:
                measurements: Output from measure_anomalies
                patient_age: Patient age for age-adjusted normals
                sex: Patient sex for sex-adjusted normals
            """
            return seg_model.compare_to_normals(measurements, patient_age, sex)

        return [segment_scan, retrieve_medical_knowledge, measure_anomalies, compare_to_normals]

    def analyze(
        self,
        scan_path: str,
        report_pdf: str | None = None,
        clinical_notes: str | None = None,
    ) -> AnalysisResult:
        """
        Run the full MediVision agentic pipeline.

        Args:
            scan_path: Path to CT/MRI scan image
            report_pdf: Optional path to existing radiology report PDF
            clinical_notes: Optional free-text clinical history

        Returns:
            AnalysisResult with findings, differential diagnosis, and recommendations
        """
        user_prompt = self._build_prompt(scan_path, report_pdf, clinical_notes)

        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_prompt)],
            "scan_path": scan_path,
            "report_pdf": report_pdf,
            "clinical_notes": clinical_notes,
            "segmentation_output": None,
            "rag_results": [],
            "final_result": None,
            "iteration_count": 0,
        }

        logger.info(f"Starting MediVision analysis for scan: {scan_path}")
        final_state = self.graph.invoke(initial_state)

        result = final_state.get("final_result")
        if result is None:
            raise RuntimeError("Agent failed to produce a final result.")

        logger.info(f"Analysis complete. Top diagnosis: {result.differential_diagnosis[0] if result.differential_diagnosis else 'None'}")
        return result

    def _build_prompt(self, scan_path: str, report_pdf: str | None, clinical_notes: str | None) -> str:
        parts = [f"Please analyze the following medical case:\n\n**Scan:** {scan_path}"]
        if report_pdf:
            parts.append(f"**Radiology report:** {report_pdf}")
        if clinical_notes:
            parts.append(f"**Clinical notes:** {clinical_notes}")
        parts.append(
            "\nPlease:\n"
            "1. Segment the scan and identify all organs and anomalies\n"
            "2. Measure any detected anomalies\n"
            "3. Retrieve relevant medical literature\n"
            "4. Compare findings to normal ranges\n"
            "5. Generate a differential diagnosis with confidence scores\n"
            "6. Recommend follow-up actions"
        )
        return "\n\n".join(parts)
