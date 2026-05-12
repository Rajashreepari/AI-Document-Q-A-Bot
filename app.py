import streamlit as st
import os
import time
from pathlib import Path
from rag_pipeline import RAGPipeline

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* Root Variables */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --bg-card-hover: #1c1c28;
    --accent-1: #7c6af7;
    --accent-2: #f76a8c;
    --accent-3: #6af7c8;
    --text-primary: #e8e8f0;
    --text-secondary: #8888aa;
    --text-muted: #55556a;
    --border: #222235;
    --border-accent: #3a3a55;
}

/* Global Reset */
html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div {
    padding: 2rem 1.5rem !important;
}

/* Main content area */
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1100px !important;
}

/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}

.app-logo {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}

.app-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
    margin: 0 !important;
    line-height: 1 !important;
}

.app-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Cards */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.stat-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.3rem;
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}

.stat-accent { color: var(--accent-1); }

/* Upload zone */
.upload-zone {
    background: var(--bg-card);
    border: 2px dashed var(--border-accent);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    transition: all 0.2s;
}

/* File item */
.file-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.85rem;
}

.file-icon {
    font-size: 1.1rem;
}

.file-name {
    color: var(--text-primary);
    font-weight: 500;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.file-badge {
    background: rgba(124, 106, 247, 0.15);
    color: var(--accent-1);
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    white-space: nowrap;
}

/* Query input */
.query-container {
    background: var(--bg-card);
    border: 1px solid var(--border-accent);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Answer card */
.answer-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.answer-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2), var(--accent-3));
}

.answer-label {
    font-size: 0.7rem;
    color: var(--accent-1);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.answer-text {
    font-size: 1rem;
    line-height: 1.8;
    color: var(--text-primary);
    font-weight: 300;
}

/* Source chunks */
.source-section {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}

.source-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.8rem;
}

.source-chunk {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-3);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.6;
    font-family: 'DM Mono', monospace;
}

.chunk-meta {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
    display: flex;
    gap: 1rem;
}

/* History item */
.history-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.15s;
}

.history-item:hover {
    border-color: var(--accent-1);
    background: var(--bg-card-hover);
}

.history-q {
    font-size: 0.85rem;
    color: var(--text-primary);
    font-weight: 500;
    margin-bottom: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-time {
    font-size: 0.7rem;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace;
}

/* Status badges */
.status-ready {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(106, 247, 200, 0.1);
    color: var(--accent-3);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    border: 1px solid rgba(106, 247, 200, 0.2);
}

.status-idle {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(136, 136, 170, 0.1);
    color: var(--text-secondary);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    border: 1px solid rgba(136, 136, 170, 0.2);
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}

/* Section title */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* Streamlit widgets overrides */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-accent) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 2px rgba(124, 106, 247, 0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-accent) !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-accent) !important;
    border-radius: 14px !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-1) !important;
}

.stSpinner > div {
    border-top-color: var(--accent-1) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
}

/* Labels */
.stTextInput label, .stTextArea label {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    font-family: 'DM Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* Alerts */
.stAlert {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* Sidebar title */
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}

.sidebar-subtitle {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace;
    margin-bottom: 1.5rem;
}

/* Metric override */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
    font-family: 'DM Mono', monospace !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent-1) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Select box */
.stSelectbox > div > div {
    background: var(--bg-secondary) !important;
    border-color: var(--border-accent) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 DocMind AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">RAG · GEMINI · CHROMADB</div>', unsafe_allow_html=True)

    # API Key
    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
    gemini_key = st.text_input(
        "GEMINI API KEY",
        type="password",
        placeholder="AIza...",
        help="Get your key at https://aistudio.google.com",
        value=os.environ.get("GEMINI_API_KEY", "")
    )

    # Model selection
    model_choice = st.selectbox(
        "MODEL",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0
    )

    st.markdown('<div class="section-title">RAG Settings</div>', unsafe_allow_html=True)
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512, 64)
    chunk_overlap = st.slider("Chunk Overlap", 0, 256, 64, 32)
    top_k = st.slider("Top-K Retrieval", 1, 10, 4)

    st.markdown("---")

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Files", len(st.session_state.indexed_files))
    with col2:
        st.metric("🧩 Chunks", st.session_state.total_chunks)

    st.metric("💬 Queries", len(st.session_state.qa_history))

    # System status
    st.markdown("---")
    if st.session_state.rag and st.session_state.indexed_files:
        st.markdown('<span class="status-ready"><span class="status-dot"></span> System Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-idle"><span class="status-dot"></span> Awaiting Documents</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear All Data", use_container_width=True):
        st.session_state.rag = None
        st.session_state.indexed_files = []
        st.session_state.qa_history = []
        st.session_state.current_answer = None
        st.session_state.total_chunks = 0
        st.rerun()

# ─── Main Area ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-logo">🧠</div>
    <div>
        <div class="app-title">DocMind AI</div>
        <div class="app-subtitle">Retrieval-Augmented Generation · Semantic Search · Gemini LLM</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Two Columns Layout ──────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1.9], gap="large")

# ════ LEFT: Upload ════
with left_col:
    st.markdown('<div class="section-title">📁 Document Ingestion</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Index Documents", use_container_width=True):
            if not gemini_key:
                st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
            else:
                with st.spinner("Initializing RAG pipeline..."):
                    try:
                        rag = RAGPipeline(
                            gemini_api_key=gemini_key,
                            model_name=model_choice,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        file_paths = []
                        for uf in uploaded_files:
                            save_path = Path(f"/tmp/{uf.name}")
                            save_path.write_bytes(uf.read())
                            file_paths.append(str(save_path))

                        total = rag.ingest_documents(file_paths)
                        st.session_state.rag = rag
                        st.session_state.indexed_files = [f.name for f in uploaded_files]
                        st.session_state.total_chunks = total
                        st.success(f"✅ Indexed {len(uploaded_files)} file(s) → {total} chunks")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Indexed files list
    if st.session_state.indexed_files:
        st.markdown('<div class="section-title">Indexed Files</div>', unsafe_allow_html=True)
        for fname in st.session_state.indexed_files:
            st.markdown(f"""
            <div class="file-item">
                <span class="file-icon">📄</span>
                <span class="file-name">{fname}</span>
                <span class="file-badge">indexed</span>
            </div>""", unsafe_allow_html=True)

    # Query history
    if st.session_state.qa_history:
        st.markdown('<div class="section-title">Query History</div>', unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.qa_history[-6:])):
            ts = item.get("timestamp", "")
            q = item.get("question", "")[:60] + ("…" if len(item.get("question","")) > 60 else "")
            st.markdown(f"""
            <div class="history-item">
                <div class="history-q">💬 {q}</div>
                <div class="history-time">{ts}</div>
            </div>""", unsafe_allow_html=True)

# ════ RIGHT: Q&A ════
with right_col:
    st.markdown('<div class="section-title">🔍 Ask Your Documents</div>', unsafe_allow_html=True)

    query = st.text_area(
        "QUERY",
        placeholder="Ask anything about your uploaded documents...",
        height=100,
        label_visibility="visible"
    )

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        ask_btn = st.button("🔍 Ask DocMind", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.current_answer = None
        st.rerun()

    if ask_btn:
        if not query.strip():
            st.warning("Please enter a question.")
        elif not st.session_state.rag:
            st.warning("Please upload and index documents first.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    result = st.session_state.rag.query(query.strip(), top_k=top_k)
                    st.session_state.current_answer = result

                    # Save to history
                    from datetime import datetime
                    st.session_state.qa_history.append({
                        "question": query.strip(),
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Query failed: {e}")

    # ── Display Answer ──
    if st.session_state.current_answer:
        res = st.session_state.current_answer

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">
                <span>◆</span> AI Response
            </div>
            <div class="answer-text">{res['answer']}</div>
        </div>
        """, unsafe_allow_html=True)

        if res.get("sources"):
            st.markdown(f"""
            <div class="source-section">
                <div class="source-label">📎 Retrieved Context Chunks ({len(res['sources'])})</div>
            """, unsafe_allow_html=True)

            for i, src in enumerate(res["sources"]):
                doc_name = src.get("source", "Unknown")
                chunk_id = src.get("chunk_id", i)
                score = src.get("score", 0)
                text_preview = src.get("text", "")[:300]

                st.markdown(f"""
                <div class="source-chunk">
                    <div class="chunk-meta">
                        <span>📄 {doc_name}</span>
                        <span>Chunk #{chunk_id}</span>
                        <span>Score: {score:.3f}</span>
                    </div>
                    {text_preview}{"…" if len(src.get("text","")) > 300 else ""}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div style="
            background: #16161f;
            border: 1px dashed #222235;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            margin-top: 1rem;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔍</div>
            <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #e8e8f0; margin-bottom: 0.5rem;">
                Ready to answer
            </div>
            <div style="font-size: 0.85rem; color: #55556a; line-height: 1.6;">
                Upload PDFs → Index them → Ask questions.<br>
                DocMind retrieves relevant chunks and generates<br>
                grounded answers using Gemini.
            </div>
        </div>
        """, unsafe_allow_html=True)
