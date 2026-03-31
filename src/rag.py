from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
import faiss
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

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
            ("system", """You are DocMind, a helpful document analyst.
Answer questions based strictly on the provided context.
If the answer is not in the context, say 'I don't find this in the document.'

Context:
{context}"""),
            ("human", "{question}")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def load_pdf(self, pdf_path: str) -> dict:
        """Extract text, chunk it, embed it, build FAISS index."""
        reader    = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        splitter     = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.chunks  = splitter.split_text(full_text)

        embeddings   = self.embedding_model.encode(
            self.chunks, show_progress_bar=False
        )
        dimension    = embeddings.shape[1]
        self.index   = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.current_pdf = os.path.basename(pdf_path)

        return {
            "filename":   self.current_pdf,
            "pages":      len(reader.pages),
            "chunks":     len(self.chunks),
            "characters": len(full_text),
        }

    @traceable(name="docmind-retrieval")
    def _retrieve(self, question: str, k: int) -> dict:
        """FAISS similarity search — traced as separate span."""
        query_embedding    = self.embedding_model.encode(
            [question]
        ).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k)
        chunks             = [self.chunks[i] for i in indices[0]]
        return {
            "question":  question,
            "chunks":    chunks,
            "distances": [round(float(d), 4) for d in distances[0]],
        }

    @traceable(name="docmind-generation")
    def _generate(self, question: str, context: str) -> str:
        """LLM generation — traced as separate span."""
        return self.chain.invoke({
            "context":  context,
            "question": question
        })

    @traceable(name="docmind-query")
    def query(self, question: str, k: int = TOP_K) -> dict:
        """Full RAG pipeline — parent trace."""
        if self.index is None:
            raise ValueError("No PDF loaded. Call load_pdf() first.")

        retrieval = self._retrieve(question, k)
        context   = "\n\n---\n\n".join(retrieval["chunks"])
        answer    = self._generate(question, context)

        return {
            "question": question,
            "answer":   answer,
            "sources":  retrieval["chunks"],
            "distances": retrieval["distances"],
        }

    def is_loaded(self) -> bool:
        return self.index is not None