import streamlit as st
import tempfile
import os
import pandas as pd
from src.rag import RAGPipeline
from src.evaluator import RAGEvaluator
from src.test_set import TEST_SET
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PDF Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Document Q&A")
st.markdown("Upload a PDF and ask questions. Powered by local Llama 3.2 — no internet, no API costs.")

# ── Initialize session state ───────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag      = RAGPipeline()
    st.session_state.messages = []
    st.session_state.doc_info = None
    st.session_state.eval_report = None

# ── Sidebar — PDF upload ───────────────────────────────────────────────────
st.sidebar.header("📂 Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if st.session_state.doc_info is None or \
       st.session_state.doc_info["filename"] != uploaded_file.name:
        with st.sidebar:
            with st.spinner("Reading and indexing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                info = st.session_state.rag.load_pdf(tmp_path)
                os.unlink(tmp_path)
                st.session_state.doc_info    = {**info, "filename": uploaded_file.name}
                st.session_state.messages    = []
                st.session_state.eval_report = None
        st.sidebar.success("Document indexed successfully")

if st.session_state.doc_info:
    info = st.session_state.doc_info
    st.sidebar.divider()
    st.sidebar.subheader("📊 Document Info")
    st.sidebar.metric("Pages",      info["pages"])
    st.sidebar.metric("Chunks",     info["chunks"])
    st.sidebar.metric("Characters", f"{info['characters']:,}")

st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")
top_k        = st.sidebar.slider("Chunks to retrieve", 1, 6, 3)
show_sources = st.sidebar.checkbox("Show source chunks", value=True)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chat", "📊 Evaluation"])

# ── TAB 1: Chat ────────────────────────────────────────────────────────────
with tab1:
    if not st.session_state.rag.is_loaded():
        st.info("👈 Upload a PDF from the sidebar to get started.")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and "sources" in msg:
                with st.expander("📚 Source chunks used"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}** (distance: {msg['distances'][i-1]})")
                        st.markdown(f"> {source}")
                        st.divider()

    if question := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                result = st.session_state.rag.query(question, k=top_k)
            st.markdown(result["answer"])
            if show_sources:
                with st.expander("📚 Source chunks used"):
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"**Chunk {i}** (distance: {result['distances'][i-1]})")
                        st.markdown(f"> {source}")
                        st.divider()

        st.session_state.messages.append({
            "role":      "assistant",
            "content":   result["answer"],
            "sources":   result["sources"],
            "distances": result["distances"]
        })

# ── TAB 2: Evaluation ──────────────────────────────────────────────────────
with tab2:
    st.header("📊 RAG Evaluation Dashboard")
    st.markdown("Measure retrieval quality and answer faithfulness on a hand-crafted test set.")

    if not st.session_state.rag.is_loaded():
        st.info("👈 Upload a PDF first to run evaluation.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Test Set")
            st.markdown(f"**{len(TEST_SET)} questions** with ground truth relevant chunks")
            for i, tc in enumerate(TEST_SET, 1):
                st.markdown(f"**Q{i}:** {tc['question']}")

        with col2:
            k_eval = st.selectbox("Evaluate at K", [1, 2, 3, 4, 5], index=2)
            run_eval = st.button("▶️ Run Evaluation", type="primary")

        if run_eval:
            with st.spinner("Running evaluation — this takes a few minutes..."):
                evaluator = RAGEvaluator(
                    rag_pipeline=st.session_state.rag, k=k_eval
                )
                st.session_state.eval_report = evaluator.evaluate_all(TEST_SET)

        if st.session_state.eval_report:
            report = st.session_state.eval_report
            agg    = report["aggregates"]

            st.divider()
            st.subheader("Aggregate Metrics")

            # Retrieval metrics
            st.markdown("**Retrieval Quality**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Precision@K", f"{agg['mean_precision_at_k']:.2f}")
            c2.metric("Recall@K",    f"{agg['mean_recall_at_k']:.2f}")
            c3.metric("Hit Rate@K",  f"{agg['mean_hit_rate_at_k']:.2f}")
            c4.metric("MRR",         f"{agg['mean_mrr']:.2f}")
            c5.metric("NDCG@K",      f"{agg['mean_ndcg_at_k']:.2f}")

            # Answer quality metrics
            st.markdown("**Answer Quality**")
            c6, c7, c8, c9 = st.columns(4)
            c6.metric("Faithfulness",     f"{agg['mean_faithfulness']:.2f}")
            c7.metric("Answer Relevance", f"{agg['mean_answer_relevance']:.2f}")
            c8.metric("Retrieval Latency",f"{agg['mean_retrieval_latency_s']:.3f}s")
            c9.metric("RAG Latency",      f"{agg['mean_full_rag_latency_s']:.2f}s")

            st.divider()
            st.subheader("Per Question Results")

            rows = []
            for r in report["results"]:
                rows.append({
                    "Question":        r["question"],
                    "Precision@K":     round(r["precision_at_k"], 2),
                    "Recall@K":        round(r["recall_at_k"], 2),
                    "Hit Rate@K":      round(r["hit_rate_at_k"], 2),
                    "MRR":             round(r["mrr"], 2),
                    "NDCG@K":          round(r["ndcg_at_k"], 2),
                    "Faithfulness":    round(r["faithfulness"], 2),
                    "Answer Relevance":round(r["answer_relevance"], 2),
                    "Latency (s)":     round(r["full_rag_latency_s"], 2),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Answers and Sources")
            for i, r in enumerate(report["results"], 1):
                with st.expander(f"Q{i}: {r['question']}"):
                    st.markdown(f"**Answer:** {r['answer']}")
                    st.markdown("**Retrieved chunks:**")
                    for j, chunk in enumerate(r["retrieved_chunks"], 1):
                        st.markdown(f"> **Chunk {j}:** {chunk[:200]}...")