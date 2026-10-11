"""Streamlit user interface for evidence-grounded report generation."""

from __future__ import annotations

import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGPipeline, RetrievedChunk

load_dotenv()

st.set_page_config(
    page_title="Evidence Grounded Report Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; }
    .hero { padding: 1.1rem 0 0.7rem; }
    .hero h1 { color: #102a43; font-size: 2.2rem; margin-bottom: 0.15rem; }
    .hero p { color: #52606d; font-size: 1.05rem; }
    .module-card { border: 1px solid #d9e2ec; background: white; border-radius: 12px; padding: 0.8rem 0.9rem; min-height: 112px; }
    .module-title { font-weight: 700; color: #102a43; font-size: 0.93rem; }
    .module-detail { color: #627d98; font-size: 0.82rem; margin-top: 0.35rem; }
    .status-pending { color: #829ab1; font-weight: 600; }
    .status-active { color: #1d4ed8; font-weight: 700; }
    .status-done { color: #16803c; font-weight: 700; }
    .source-card { border-left: 4px solid #2f80ed; background: #f7fbff; padding: 0.65rem 0.8rem; margin: 0.45rem 0; border-radius: 0 8px 8px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODULES = [
    ("Document Upload", "Validate files and create document ID"),
    ("Text Extraction", "Read content and page information"),
    ("Text Preprocessing", "Clean text and identify sections"),
    ("Chunking and Storage", "Create embeddings and vector index"),
    ("Query and Retrieval", "Find relevant document evidence"),
    ("RAG Report Generation", "Generate grounded answer or report"),
    ("Evidence Verification", "Add citations and confidence score"),
    ("Output Ready", "Prepare answer or report for review"),
]


def init_state() -> None:
    if "module_status" not in st.session_state:
        st.session_state.module_status = {i: ("pending", "Waiting to start", 0) for i in range(1, 9)}
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "processing_started" not in st.session_state:
        st.session_state.processing_started = None


def reset_status() -> None:
    st.session_state.module_status = {i: ("pending", "Waiting to start", 0) for i in range(1, 9)}


def status_callback(module: int, state: str, detail: str) -> None:
    progress = 100 if state == "done" else 55 if state == "active" else 0
    st.session_state.module_status[module] = (state, detail, progress)
    render_module_flow(pipeline_slot)


def render_module_flow(target) -> None:
    """Replace one fixed pipeline view instead of appending a new view per update."""
    statuses = st.session_state.module_status
    completed_progress = sum(statuses[index][2] for index in statuses)
    overall = round(completed_progress / len(MODULES))
    active_module = next((index for index, value in statuses.items() if value[0] == "active"), None)
    elapsed = "00:00"
    if st.session_state.processing_started:
        seconds = int(time.time() - st.session_state.processing_started)
        elapsed = f"{seconds // 60:02d}:{seconds % 60:02d}"

    with target.container():
        st.subheader("Live Processing Pipeline")
        st.progress(overall, text=f"Overall processing: {overall}%  •  Elapsed time: {elapsed}")
        if active_module:
            st.caption(f"Currently processing: Module {active_module} — {MODULES[active_module - 1][0]}")
        elif st.session_state.processed:
            st.caption("Document processing complete — ready for questions and reports.")
        else:
            st.caption("Upload documents, then click Process documents to begin.")

        for start in (0, 4):
            cols = st.columns(4)
            for offset, (name, fallback) in enumerate(MODULES[start:start + 4]):
                index = start + offset + 1
                state, detail, progress = statuses[index]
                icon = {"pending": "○", "active": "◉", "done": "✓"}[state]
                color = "#16803c" if state == "done" else "#2563eb" if state == "active" else "#cbd5e1"
                cols[offset].markdown(
                    f'<div class="module-card"><div class="module-title">{icon} {index}. {name}</div>'
                    f'<div class="status-{state}">{state.title()} · {progress}%</div>'
                    f'<div class="module-detail">{detail or fallback}</div>'
                    f'<div style="background:#e7edf3;border-radius:99px;height:7px;margin-top:10px;overflow:hidden;">'
                    f'<div style="background:{color};width:{progress}%;height:100%;"></div></div></div>',
                    unsafe_allow_html=True,
                )
def display_sources(evidence: list[RetrievedChunk]) -> None:
    with st.expander("Retrieved evidence and source citations", expanded=False):
        for index, item in enumerate(evidence, 1):
            st.markdown(
                f'<div class="source-card"><b>{index}. {item.chunk.source} - page {item.chunk.page}</b> '
                f'({item.score:.0%} relevance)<br>{item.chunk.text[:420]}...</div>',
                unsafe_allow_html=True,
            )


init_state()

st.markdown(
    """
    <div class="hero">
      <h1>Evidence Grounded Report Generator</h1>
      <p>Analyze documents, retrieve supporting evidence, and generate traceable answers or structured reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Workspace")
    uploaded_files = st.file_uploader(
        "Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True,
        help="Upload one or more documents. Maximum file size: 25 MB each.",
    )
    st.caption("Supported formats: PDF, DOCX, TXT")
    api_keys = [key.strip() for key in os.getenv("OPENROUTER_API_KEYS", "").replace("\n", ",").split(",") if key.strip()]
    api_ready = bool(api_keys or os.getenv("OPENROUTER_API_KEY"))
    if api_ready:
        st.success(f"OpenRouter API ready ({len(api_keys) or 1} key{'s' if len(api_keys) != 1 else ''})")
    else:
        st.warning("Add OPENROUTER_API_KEY to .env before generation")
    st.caption("After selecting files, click **Process documents** to start the pipeline.")
    process = st.button("Process documents", type="primary", use_container_width=True, disabled=not uploaded_files)
    if st.button("Reset workspace", use_container_width=True):
        st.session_state.pipeline = None
        st.session_state.processed = False
        reset_status()
        st.rerun()

pipeline_slot = st.empty()
render_module_flow(pipeline_slot)

if process:
    reset_status()
    st.session_state.processing_started = time.time()
    with st.spinner("Processing documents through the pipeline..."):
        try:
            pipeline = RAGPipeline(status_callback=status_callback)
            stats = pipeline.ingest(uploaded_files)
            st.session_state.pipeline = pipeline
            st.session_state.processed = True
            status_callback(8, "done", "Knowledge base is ready for questions and reports")
            st.success(f"Knowledge base ready: {stats['pages']} page segment(s), {stats['chunks']} searchable chunks.")
        except Exception as error:
            st.session_state.processed = False
            st.error(str(error))

st.divider()
qa_tab, report_tab = st.tabs(["Ask a Question", "Generate Structured Report"])

with qa_tab:
    st.subheader("Document Question Answering")
    question = st.text_area("Ask a question about the uploaded documents", placeholder="Example: What are the main risks identified in the document?", height=100)
    if st.button("Generate grounded answer", type="primary", disabled=not st.session_state.processed):
        try:
            with st.spinner("Retrieving evidence and generating an answer..."):
                answer, evidence, confidence = st.session_state.pipeline.answer(question)
            st.markdown("### Answer")
            st.markdown(answer)
            st.metric("Evidence confidence", f"{confidence}%")
            display_sources(evidence)
        except Exception as error:
            st.error(str(error))

with report_tab:
    st.subheader("Structured Report Generation")
    report_request = st.text_area(
        "Describe the report you need",
        placeholder="Example: Generate a management report highlighting performance, risks, metrics, and recommendations.",
        height=100,
    )
    if st.button("Generate structured report", type="primary", disabled=not st.session_state.processed):
        try:
            with st.spinner("Retrieving evidence and generating a structured report..."):
                report, evidence, confidence = st.session_state.pipeline.report(report_request)
            st.markdown("### Structured Report")
            st.markdown(report)
            st.metric("Evidence confidence", f"{confidence}%")
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button("Download report as Markdown", report, f"grounded_report_{stamp}.md", "text/markdown")
            display_sources(evidence)
        except Exception as error:
            st.error(str(error))
