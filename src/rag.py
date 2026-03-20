from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import faiss
import numpy as np
import os
import pickle

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K           = 3


class RAGPipeline:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.llm             = ChatOllama(model="llama3.2")
        self.parser          = StrOutputParser()
        self.chunks          = []
        self.index           = None
        self.current_pdf     = None

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions 
based strictly on the provided context from a document.

Rules:
- Only use information from the context below
- If the answer is not in the context, say "I don't find this in the document"
- Be concise and precise
- Quote relevant parts when helpful

Context:
{context}"""),
            ("human", "{question}")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def load_pdf(self, pdf_path: str) -> dict:
        """Extract text, chunk it, embed it, build FAISS index."""
        # Extract text
        reader    = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        # Chunk
        splitter     = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.chunks  = splitter.split_text(full_text)

        # Embed
        embeddings = self.embedding_model.encode(
            self.chunks, show_progress_bar=False
        )

        # Build FAISS index
        dimension   = embeddings.shape[1]
        self.index  = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.current_pdf = os.path.basename(pdf_path)

        return {
            "filename":   self.current_pdf,
            "pages":      len(reader.pages),
            "chunks":     len(self.chunks),
            "characters": len(full_text),
        }

    def query(self, question: str, k: int = TOP_K) -> dict:
        """Retrieve relevant chunks and generate a grounded answer."""
        if self.index is None:
            raise ValueError("No PDF loaded. Call load_pdf() first.")

        # Retrieve
        query_embedding    = self.embedding_model.encode(
            [question]
        ).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k)
        retrieved_chunks   = [self.chunks[i] for i in indices[0]]

        # Generate
        context = "\n\n---\n\n".join(retrieved_chunks)
        answer  = self.chain.invoke({
            "context":  context,
            "question": question
        })

        return {
            "question": question,
            "answer":   answer,
            "sources":  retrieved_chunks,
            "distances": [round(float(d), 4) for d in distances[0]],
        }

    def is_loaded(self) -> bool:
        return self.index is not None