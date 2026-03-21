# 📄 RAG PDF Q&A with Retrieval Evaluation

Ask questions about any PDF document using a fully local RAG pipeline.
Includes a comprehensive evaluation dashboard to measure retrieval quality
and answer faithfulness. No API costs. No data leaving your machine.

## How It Works

### Chat Pipeline
1. Upload a PDF — text extracted and split into chunks
2. Chunks embedded and stored in a local FAISS vector index
3. Ask a question — embedded and matched against chunks
4. Top matching chunks + question sent to Llama 3.2
5. Answer grounded in document with source references shown

### Evaluation Pipeline
1. Hand-crafted test set with ground truth relevant chunks
2. Retrieval metrics — how well does FAISS find the right chunks?
3. Answer quality metrics — is the LLM faithful to the context?
4. Latency metrics — where is the bottleneck?

## Architecture
```
PDF → PyPDF → RecursiveCharacterTextSplitter → 
SentenceTransformers → FAISS Index

Question → Embed → FAISS Search → Top K Chunks → 
ChatOllama → Grounded Answer

Test Set → RAGEvaluator → Metrics Dashboard
```

## Evaluation Metrics

### Retrieval Quality
| Metric | What it measures |
|--------|-----------------|
| Precision@K | Of K chunks retrieved, what fraction are relevant? |
| Recall@K | Of all relevant chunks, what fraction did we find? |
| Hit Rate@K | Did at least one relevant chunk appear in top K? |
| MRR | How highly ranked is the first relevant chunk? |
| NDCG@K | Are the most relevant chunks ranked highest? |

### Answer Quality
| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Does the answer stay within retrieved context? |
| Answer Relevance | Does the answer address the question asked? |

### System
| Metric | What it measures |
|--------|-----------------|
| Retrieval Latency | Time for FAISS similarity search |
| Full RAG Latency | End-to-end time including LLM generation |

## Setup
```bash
git clone git@github.com:ankitmathur45/rag-pdf-qa.git
cd rag-pdf-qa
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt

# Ollama must be running with llama3.2
ollama pull llama3.2

streamlit run app.py
```

## Results on Sample Test Set
- Mean Recall@3: 1.00 — finds relevant chunks every time
- Mean NDCG@3: 0.92 — ranks best chunks near the top
- Mean Faithfulness: 0.90 — answers grounded in context
- Mean Answer Relevance: 1.00 — always addresses the question
- Mean Retrieval Latency: 0.011s — FAISS is near-instant
- Mean Full RAG Latency: 2.56s — LLM dominates total latency

## Tech Stack
- Python 3.12
- LangChain + LangChain Core
- FAISS (local vector store)
- Sentence Transformers (local embeddings)
- Llama 3.2 via Ollama
- RapidFuzz (fuzzy chunk matching for evaluation)
- Streamlit