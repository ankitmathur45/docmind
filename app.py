import streamlit as st
import tempfile
import os
import pandas as pd
from dotenv import load_dotenv
from src.rag import RAGPipeline
from src.corrective_rag import CorrectiveRAGPipeline
from src.agent import DocMindAgent
from src.evaluator import RAGEvaluator
from src.test_set import TEST_SET

load_dotenv()

st.set_page_config(
    page_title="DocMind",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocMind")
st.markdown("Local document Q&A powered by Llama 3.2 — no internet, no API costs.")

# ── Session state ──────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag         = RAGPipeline()
    st.session_state.messages    = []
    st.session_state.doc_info    = None
    st.session_state.eval_report = None

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("📂 Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if st.session_state.doc_info is None or \
       st.session_state.doc_info["filename"] != uploaded_file.name:
        with st.sidebar:
            with st.spinner("Reading and indexing PDF..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                info = st.session_state.rag.load_pdf(tmp_path)
                os.unlink(tmp_path)
                st.session_state.doc_info    = {
                    **info, "filename": uploaded_file.name
                }
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
mode         = st.sidebar.radio("RAG Mode", ["Standard", "Corrective", "Agent"])
top_k        = st.sidebar.slider("Chunks to retrieve", 1, 6, 3)
show_sources = st.sidebar.checkbox("Show source chunks", value=True)

if mode == "Corrective":
    st.sidebar.caption("Grades chunks for relevance before generating. Slower but more accurate.")
elif mode == "Agent":
    st.sidebar.caption("Selects the best tool for each question — search, summarise, calculate, or metadata.")

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

    # Mode banner
    if mode == "Corrective":
        st.info("🔄 Corrective RAG — chunks graded before generation.")
    elif mode == "Agent":
        st.info("🤖 Agent mode — selects tools based on your question.")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("tools_used"):
                    st.caption("🔧 Tools used: " +
                               ", ".join(t["tool"] for t in msg["tools_used"]))
                if show_sources and msg.get("sources"):
                    with st.expander("📚 Source chunks"):
                        for i, source in enumerate(msg["sources"], 1):
                            score = msg["relevance_scores"][i-1] \
                                    if msg.get("relevance_scores") else None
                            color = "🟢" if score == "relevant" \
                                    else "🔴" if score == "irrelevant" else "⚪"
                            label = f"{color} **Chunk {i}**" + \
                                    (f" ({score})" if score else "")
                            st.markdown(label)
                            st.markdown(f"> {source[:200]}...")
                            st.divider()

    # Chat input
    if question := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({
            "role": "user", "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if mode == "Corrective":
                with st.spinner("Grading chunks and generating answer..."):
                    crag   = CorrectiveRAGPipeline(st.session_state.rag)
                    result = crag.query(question)
                result["tools_used"] = []

            elif mode == "Agent":
                with st.spinner("Agent thinking — selecting tools..."):
                    agent  = DocMindAgent(st.session_state.rag)
                    result = agent.query(question)
                result["sources"]            = []
                result["relevance_scores"]   = []
                result["rewritten_question"] = ""

            else:
                with st.spinner("Searching and generating..."):
                    result = st.session_state.rag.query(question, k=top_k)
                result["relevance_scores"]   = []
                result["rewritten_question"] = ""
                result["tools_used"]         = []

            st.markdown(result["answer"])

            if result.get("tools_used"):
                st.caption("🔧 Tools used: " +
                           ", ".join(t["tool"] for t in result["tools_used"]))

            if result.get("rewritten_question"):
                st.caption(f"🔄 Rewritten to: {result['rewritten_question']}")

            if show_sources and result.get("sources"):
                with st.expander("📚 Source chunks"):
                    for i, source in enumerate(result["sources"], 1):
                        score = result["relevance_scores"][i-1] \
                                if result.get("relevance_scores") else None
                        color = "🟢" if score == "relevant" \
                                else "🔴" if score == "irrelevant" else "⚪"
                        label = f"{color} **Chunk {i}**" + \
                                (f" ({score})" if score else "")
                        st.markdown(label)
                        st.markdown(f"> {source[:200]}...")
                        st.divider()

        st.session_state.messages.append({
            "role":               "assistant",
            "content":            result["answer"],
            "sources":            result.get("sources", []),
            "relevance_scores":   result.get("relevance_scores", []),
            "rewritten_question": result.get("rewritten_question", ""),
            "tools_used":         result.get("tools_used", []),
        })

# ── TAB 2: Evaluation ──────────────────────────────────────────────────────
with tab2:
    st.header("📊 RAG Evaluation Dashboard")
    st.markdown("Measure retrieval quality and answer faithfulness.")

    if not st.session_state.rag.is_loaded():
        st.info("👈 Upload a PDF first.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Test Set")
            for i, tc in enumerate(TEST_SET, 1):
                st.markdown(f"**Q{i}:** {tc['question']}")
        with col2:
            k_eval   = st.selectbox("Evaluate at K", [1, 2, 3, 4, 5], index=2)
            run_eval = st.button("▶️ Run Evaluation", type="primary")

        if run_eval:
            with st.spinner("Running evaluation..."):
                evaluator = RAGEvaluator(
                    rag_pipeline=st.session_state.rag, k=k_eval
                )
                st.session_state.eval_report = evaluator.evaluate_all(TEST_SET)

        if st.session_state.eval_report:
            report = st.session_state.eval_report
            agg    = report["aggregates"]

            st.divider()
            st.subheader("Aggregate Metrics")

            st.markdown("**Retrieval Quality**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Precision@K", f"{agg['mean_precision_at_k']:.2f}")
            c2.metric("Recall@K",    f"{agg['mean_recall_at_k']:.2f}")
            c3.metric("Hit Rate@K",  f"{agg['mean_hit_rate_at_k']:.2f}")
            c4.metric("MRR",         f"{agg['mean_mrr']:.2f}")
            c5.metric("NDCG@K",      f"{agg['mean_ndcg_at_k']:.2f}")

            st.markdown("**Answer Quality**")
            c6, c7, c8, c9 = st.columns(4)
            c6.metric("Faithfulness",      f"{agg['mean_faithfulness']:.2f}")
            c7.metric("Answer Relevance",  f"{agg['mean_answer_relevance']:.2f}")
            c8.metric("Retrieval Latency", f"{agg['mean_retrieval_latency_s']:.3f}s")
            c9.metric("RAG Latency",       f"{agg['mean_full_rag_latency_s']:.2f}s")

            st.divider()
            st.subheader("Per Question Results")
            rows = []
            for r in report["results"]:
                rows.append({
                    "Question":         r["question"],
                    "Precision@K":      round(r["precision_at_k"], 2),
                    "Recall@K":         round(r["recall_at_k"], 2),
                    "Hit Rate@K":       round(r["hit_rate_at_k"], 2),
                    "MRR":              round(r["mrr"], 2),
                    "NDCG@K":           round(r["ndcg_at_k"], 2),
                    "Faithfulness":     round(r["faithfulness"], 2),
                    "Answer Relevance": round(r["answer_relevance"], 2),
                    "Latency (s)":      round(r["full_rag_latency_s"], 2),
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )