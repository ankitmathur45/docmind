# 📄 RAG PDF Q&A

Ask questions about any PDF document using a fully local RAG pipeline.
No API costs. No data leaving your machine.

## How It Works
1. Upload a PDF — text is extracted and split into chunks
2. Chunks are embedded and stored in a local FAISS vector index
3. Ask a question — it gets embedded and matched against chunks
4. Top matching chunks + your question go to Llama 3.2
5. You get an answer grounded in your document with source references

## Architecture
```
PDF → Extract → Chunk → Embed → FAISS Index
Question → Embed → Similarity Search → Relevant Chunks → Llama → Answer
```

## Setup
```bash
git clone git@github.com:ankitmathur45/rag-pdf-qa.git
cd rag-pdf-qa
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt

# Ollama must be running with llama3.2 pulled
ollama pull llama3.2

streamlit run app.py
```

## Tech Stack
- Python 3.12
- LangChain
- FAISS (local vector store)
- Sentence Transformers (local embeddings)
- Llama 3.2 via Ollama
- Streamlit
```