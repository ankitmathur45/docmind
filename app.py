import streamlit as st
import tempfile
import os
from src.rag import RAGPipeline

st.set_page_config(
    page_title="PDF Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Document Q&A")
st.markdown("Upload a PDF and ask questions about it. Powered by local Llama 3.2 — no internet, no API costs.")

# ── Initialize RAG pipeline in session state ───────────────────────────────
# Session state persists across reruns — the index stays in memory
if "rag" not in st.session_state:
    st.session_state.rag      = RAGPipeline()
    st.session_state.messages = []
    st.session_state.doc_info = None

# ── Sidebar — PDF upload ───────────────────────────────────────────────────
st.sidebar.header("📂 Document")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Only reindex if a new file is uploaded
    if st.session_state.doc_info is None or \
       st.session_state.doc_info["filename"] != uploaded_file.name:

        with st.sidebar:
            with st.spinner("Reading and indexing PDF..."):
                # Save uploaded file to a temp location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load into RAG pipeline
                info = st.session_state.rag.load_pdf(tmp_path)
                os.unlink(tmp_path)  # clean up temp file

                st.session_state.doc_info = {**info, "filename": uploaded_file.name}
                st.session_state.messages = []  # reset chat for new doc

        st.sidebar.success("Document indexed successfully")

# ── Show document info ─────────────────────────────────────────────────────
if st.session_state.doc_info:
    info = st.session_state.doc_info
    st.sidebar.divider()
    st.sidebar.subheader("📊 Document Info")
    st.sidebar.metric("Pages",      info["pages"])
    st.sidebar.metric("Chunks",     info["chunks"])
    st.sidebar.metric("Characters", f"{info['characters']:,}")

# ── Settings ───────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")
top_k       = st.sidebar.slider("Chunks to retrieve", 1, 6, 3,
                                 help="More chunks = more context but slower")
show_sources = st.sidebar.checkbox("Show source chunks", value=True)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ── Main chat interface ────────────────────────────────────────────────────
if not st.session_state.rag.is_loaded():
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and "sources" in msg:
            with st.expander("📚 Source chunks used"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}** (distance: {msg['distances'][i-1]})")
                    st.markdown(f"> {source}")
                    st.divider()

# Chat input
if question := st.chat_input("Ask a question about the document..."):

    # Add user message
    st.session_state.messages.append({
        "role":    "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            result = st.session_state.rag.query(question, k=top_k)

        st.markdown(result["answer"])

        if show_sources:
            with st.expander("📚 Source chunks used"):
                for i, source in enumerate(result["sources"], 1):
                    st.markdown(f"**Chunk {i}** (distance: {result['distances'][i-1]})")
                    st.markdown(f"> {source}")
                    st.divider()

    # Save to history
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   result["answer"],
        "sources":   result["sources"],
        "distances": result["distances"]
    })