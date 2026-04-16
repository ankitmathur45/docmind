import os
import numpy as np
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

load_dotenv()

TOP_K           = 5
RELEVANCE_THRESHOLD = 2
MAX_ATTEMPTS    = 2


class RAGState(TypedDict):
    question:           str
    retrieved_chunks:   list[str]
    distances:          list[float]
    relevance_scores:   list[str]
    rewritten_question: str
    context:            str
    answer:             str
    attempts:           int
    decision:           str


class CorrectiveRAGPipeline:
    def __init__(self, rag_pipeline):
        self.rag    = rag_pipeline
        self.llm    = ChatOllama(model="llama3.2")
        self.parser = StrOutputParser()
        self.graph  = self._build_graph()

    def _build_graph(self):
        # ── Prompts ────────────────────────────────────────────────────────
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict relevance grader for a RAG system.
Decide if a document chunk DIRECTLY helps answer the question.
Be strict — loosely related chunks are irrelevant.
If the question is nonsense or gibberish, mark all chunks irrelevant.
Respond with ONLY one word: 'relevant' or 'irrelevant'"""),
            ("human", "Question: {question}\n\nChunk: {chunk}\n\nDirectly relevant?")
        ])

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query rewriter for a RAG system.
Rewrite the question to be more specific and retrieve better chunks.
Return ONLY the rewritten question."""),
            ("human", "Original: {question}\n\nRewrite to be more specific:")
        ])

        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are DocMind, a helpful document analyst.
Answer based strictly on the provided context.
If not found, say 'I don't find this in the document.'

Context:
{context}"""),
            ("human", "{question}")
        ])

        grade_chain    = grade_prompt    | self.llm | self.parser
        rewrite_chain  = rewrite_prompt  | self.llm | self.parser
        generate_chain = generate_prompt | self.llm | self.parser

        # ── Nodes ──────────────────────────────────────────────────────────
        def retrieve(state: RAGState) -> dict:
            question  = state.get("rewritten_question") or state["question"]
            query_emb = self.rag.embedding_model.encode(
                [question]
            ).astype(np.float32)
            dists, idxs = self.rag.index.search(query_emb, TOP_K)
            chunks      = [self.rag.chunks[i] for i in idxs[0]]
            distances   = [round(float(d), 4) for d in dists[0]]
            return {
                "retrieved_chunks": chunks,
                "distances":        distances,
                "relevance_scores": [],
            }

        def grade_chunks(state: RAGState) -> dict:
            question = state.get("rewritten_question") or state["question"]
            scores   = []
            for chunk in state["retrieved_chunks"]:
                score = grade_chain.invoke({
                    "question": question,
                    "chunk":    chunk[:300]
                }).strip().lower()
                score = "relevant" if "relevant" in score \
                        and "irrelevant" not in score else "irrelevant"
                scores.append(score)
            relevant_count = scores.count("relevant")
            decision       = "generate" if relevant_count >= RELEVANCE_THRESHOLD \
                             else "rewrite"
            return {"relevance_scores": scores, "decision": decision}

        def rewrite_question(state: RAGState) -> dict:
            rewritten = rewrite_chain.invoke(
                {"question": state["question"]}
            ).strip()
            return {
                "rewritten_question": rewritten,
                "attempts":           state["attempts"] + 1
            }

        def generate_answer(state: RAGState) -> dict:
            question = state.get("rewritten_question") or state["question"]
            good_chunks = [
                chunk for chunk, score
                in zip(state["retrieved_chunks"], state["relevance_scores"])
                if score == "relevant"
            ] if state["relevance_scores"] else state["retrieved_chunks"]

            chunks  = good_chunks if good_chunks else state["retrieved_chunks"]
            context = "\n\n---\n\n".join(chunks)
            answer  = generate_chain.invoke({
                "context":  context,
                "question": question
            })
            return {"context": context, "answer": answer}

        def route_after_grading(state: RAGState) -> str:
            relevant_count = state["relevance_scores"].count("relevant")
            if relevant_count >= RELEVANCE_THRESHOLD:
                return "generate"
            elif state["attempts"] >= MAX_ATTEMPTS:
                return "generate"
            else:
                return "rewrite"

        # ── Build graph ────────────────────────────────────────────────────
        graph = StateGraph(RAGState)
        graph.add_node("retrieve",  retrieve)
        graph.add_node("grade",     grade_chunks)
        graph.add_node("rewrite",   rewrite_question)
        graph.add_node("generate",  generate_answer)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve",  "grade")
        graph.add_conditional_edges(
            "grade",
            route_after_grading,
            {"generate": "generate", "rewrite": "rewrite"}
        )
        graph.add_edge("rewrite",  "retrieve")
        graph.add_edge("generate", END)

        return graph.compile()

    @traceable(name="corrective-rag-query")
    def query(self, question: str) -> dict:
        """Run corrective RAG pipeline — traced in LangSmith."""
        result = self.graph.invoke({
            "question":           question,
            "retrieved_chunks":   [],
            "distances":          [],
            "relevance_scores":   [],
            "rewritten_question": "",
            "context":            "",
            "answer":             "",
            "attempts":           0,
            "decision":           "",
        })
        return {
            "question":           question,
            "answer":             result["answer"],
            "sources":            result["retrieved_chunks"],
            "relevance_scores":   result["relevance_scores"],
            "rewritten_question": result.get("rewritten_question", ""),
            "attempts":           result["attempts"],
        }

    def is_loaded(self) -> bool:
        return self.rag.is_loaded()