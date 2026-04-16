# 📄 DocMind

A production-quality RAG system for PDF document Q&A.
Fully local — no internet, no API costs, no data leaving your machine.

## Features

- 💬 Chat with any PDF document
- 🔄 Two RAG modes — Standard and Corrective
- 📊 Retrieval evaluation dashboard — 7 metrics
- 🔍 LangSmith observability — every query traced
- 🟢🔴 Chunk relevance indicators in UI

## Architecture

### Standard RAG

```
PDF → Chunk → Embed → FAISS Index
Question → Embed → Retrieve → Generate → Answer
```

### Corrective RAG (LangGraph)

```
Question → Retrieve → Grade chunks
                    ↓ relevant    ↓ irrelevant
                Generate      Rewrite question → Retrieve again
                    ↓
                  Answer
```

## DocMind Roadmap

| Step | Feature | Status |
|------|---------|--------|
| 1 | PDF Q&A with FAISS + Ollama | ✅ Done |
| 2 | Retrieval evaluation — Precision, Recall, MRR, NDCG | ✅ Done |
| 3 | LangSmith observability — tracing and prompt versioning | ✅ Done |
| 4 | Corrective RAG with LangGraph — chunk grading, query rewriting | ✅ Done |
| 5 | Tool-using agent | ⬜ Next |
| 6 | LangServe deployment | ⬜ Planned |
| 7 | Multi-agent supervisor | ⬜ Planned |

## Evaluation Results

| Metric | Score |
|--------|-------|
| Mean Recall@3 | 1.00 |
| Mean NDCG@3 | 0.92 |
| Mean Faithfulness | 0.90 |
| Mean Answer Relevance | 1.00 |
| Mean Retrieval Latency | 0.011s |

## Setup

```bash
git clone git@github.com:ankitmathur45/docmind.git
cd docmind
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=docmind
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

Pull the model and run:

```bash
ollama pull llama3.2
python -m streamlit run app.py
```

## Tech Stack

- Python 3.12
- LangChain + LangGraph
- FAISS (local vector store)
- Sentence Transformers (local embeddings)
- Llama 3.2 via Ollama
- LangSmith (observability)
- RapidFuzz (fuzzy chunk matching)
- Streamlit

## Note on Answer Quality

Optimised for local inference on consumer hardware (GTX 1660 Ti, 6GB VRAM).
Answer quality scales significantly with larger models — production deployments
would use Llama 3.1 70B or a cloud LLM for the generation and grading steps.