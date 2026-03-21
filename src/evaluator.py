import time
import json
import re
import statistics
import numpy as np

from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RETRIEVAL METRICS
from rapidfuzz import fuzz

def is_relevant(chunk: str, relevant_chunks: list, threshold: int = 80) -> bool:
    """
    Check if a retrieved chunk matches any relevant chunk
    using fuzzy string matching instead of exact matching.
    threshold=80 means 80% similarity is enough to count as a match.
    """
    for relevant in relevant_chunks:
        score = fuzz.partial_ratio(
            chunk[:200].lower().strip(),
            relevant[:200].lower().strip()
        )
        if score >= threshold:
            return True
    return False


def get_relevance_score(chunk: str, relevance_scores: dict, threshold: int = 80) -> float:
    """Get graded relevance score for a chunk using fuzzy matching."""
    best_score = 0
    for ref_chunk, score in relevance_scores.items():
        similarity = fuzz.partial_ratio(
            chunk[:200].lower().strip(),
            ref_chunk[:200].lower().strip()
        )
        if similarity >= threshold:
            best_score = max(best_score, score)
    return best_score


def precision_at_k(retrieved: list, relevant_chunks: list, k: int) -> float:
    hits = sum(1 for chunk in retrieved[:k] if is_relevant(chunk, relevant_chunks))
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list, relevant_chunks: list, k: int) -> float:
    if not relevant_chunks:
        return 0.0
    hits = sum(1 for chunk in retrieved[:k] if is_relevant(chunk, relevant_chunks))
    return hits / len(relevant_chunks)


def hit_rate_at_k(retrieved: list, relevant_chunks: list, k: int) -> float:
    return 1.0 if any(is_relevant(c, relevant_chunks) for c in retrieved[:k]) else 0.0


def mean_reciprocal_rank(retrieved: list, relevant_chunks: list) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, relevant_chunks):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list, relevance_scores: dict, k: int) -> float:
    def get_score(chunk):
        return get_relevance_score(chunk, relevance_scores)

    def dcg(chunks):
        return sum(
            get_score(chunk) / np.log2(rank + 1)
            for rank, chunk in enumerate(chunks[:k], start=1)
        )

    actual_dcg = dcg(retrieved)
    ideal_order = sorted(
        relevance_scores.keys(),
        key=lambda x: relevance_scores[x],
        reverse=True
    )
    ideal_dcg = dcg(ideal_order)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
# ANSWER QUALITY METRICS

class AnswerEvaluator:
    def __init__(self, model: str = "llama3.2"):
            llm = ChatOllama(model=model)
            parser = StrOutputParser()

            self.faithfulness_chain = ChatPromptTemplate.from_messages([
            ("system", """You are an evaluation assistant checking if an answer is 
faithful to the provided context — using only information from the context.

Respond with ONLY a JSON object:
{{"score": 0.9, "reason": "one sentence explanation"}}

1.0 = completely faithful
0.5 = partially faithful  
0.0 = unfaithful or hallucinated"""),
            ("human", "Context:\n{context}\n\nAnswer:\n{answer}\n\nIs this answer faithful?")
        ]) | llm | parser

            self.relevance_chain = ChatPromptTemplate.from_messages([
            ("system", """You are an evaluation assistant checking if an answer
addresses the question asked.

Respond with ONLY a JSON object:
{{"score": 0.9, "reason": "one sentence explanation"}}

1.0 = directly and completely addresses the question
0.5 = partially addresses the question
0.0 = does not address the question"""),
            ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nDoes this answer address the question?")
        ]) | llm | parser

    def _parse_score(self, response: str) -> dict:
        try:
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"score": 0.0, "reason": "Could not parse response"}
    
    def evaluate(self, question: str, context: str, answer: str) -> dict:
        faith = self._parse_score(
            self.faithfulness_chain.invoke({"context": context, "answer": answer})
        )
        rel = self._parse_score(
            self.relevance_chain.invoke({"question": question, "answer": answer})
        )
        return {
            "faithfulness":         round(faith.get("score", 0.0), 3),
            "faithfulness_reason":  faith.get("reason", ""),
            "answer_relevance":     round(rel.get("score", 0.0), 3),
            "relevance_reason":     rel.get("reason", ""),
        }
    

# ── Latency Metrics ────────────────────────────────────────────────────────────

def measure_latency(func, *args, n_runs: int = 2, **kwargs) -> dict:
    """Measure execution time of a function over n_runs."""
    latencies = []
    result    = None
    for _ in range(n_runs):
        start  = time.time()
        result = func(*args, **kwargs)
        latencies.append(time.time() - start)
    return {
        "result": result,
        "mean":   round(statistics.mean(latencies), 3),
        "min":    round(min(latencies), 3),
        "max":    round(max(latencies), 3),
        "std":    round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0.0,
    }


# Full Evaluation Runner

class RAGEvaluator: 
    """
    Runs all retrieval and answer quality metrics on a test set.

    Test set format:
    [
        {
            "question":        "What is supervised learning?",
            "relevant_chunks": ["chunk text 1", "chunk text 2"],
            "relevance_scores": {"chunk text 1": 2, "chunk text 2": 1},
        },
        ...
    ]
    """

    def __init__(self, rag_pipeline, k: int = 3):
        self.rag = rag_pipeline
        self.k = k
        self.evaluator = AnswerEvaluator()

    def _get_retrieved_texts(self, question: str) -> list[str]:
        query_emb = self.rag.embedding_model.encode([question]).astype(np.float32)
        distances, indices = self.rag.index.search(query_emb, self.k)
        return [self.rag.chunks[i] for i in indices[0]]
    
    def evaluate_single(self, test_case: dict) -> dict:
        """Evaluate one question from test set"""
        question        = test_case["question"]
        relevant_chunks = test_case["relevant_chunks"]  # keep as list for fuzzy matching
        relevance_scores = test_case.get("relevance_scores", {
            c: 1 for c in relevant_chunks
        })

        # Measure retrieval latency
        r_stats    = measure_latency(self._get_retrieved_texts, question, n_runs=2)
        retrieved  = r_stats["result"]

        # Measure full RAG latency
        f_stats    = measure_latency(self.rag.query, question, n_runs=1)
        rag_result = f_stats["result"]

        # Retrieval metrics
        retrieval_metrics = {
            "precision_at_k":  precision_at_k(retrieved, relevant_chunks, self.k),
            "recall_at_k":     recall_at_k(retrieved, relevant_chunks, self.k),
            "hit_rate_at_k":   hit_rate_at_k(retrieved, relevant_chunks, self.k),
            "mrr":             mean_reciprocal_rank(retrieved, relevant_chunks),
            "ndcg_at_k":       ndcg_at_k(retrieved, relevance_scores, self.k),
        }

        # Answer quality metrics
        context      = "\n\n---\n\n".join(retrieved)
        answer_metrics = self.evaluator.evaluate(
            question, context, rag_result["answer"]
        )

        # Latency metrics
        latency_metrics = {
            "retrieval_latency_s": r_stats["mean"],
            "full_rag_latency_s":  f_stats["mean"],
        }

        return {
            "question":         question,
            "answer":           rag_result["answer"],
            "retrieved_chunks": retrieved,
            **retrieval_metrics,
            **answer_metrics,
            **latency_metrics,
        }

    def evaluate_all(self, test_set: list[dict]) -> dict:
        """Evaluate all test cases and return aggregate metrics."""
        results = []
        for i, test_case in enumerate(test_set, 1):
            print(f"Evaluating {i}/{len(test_set)}: {test_case['question'][:50]}...")
            results.append(self.evaluate_single(test_case))

        # Aggregate
        metric_keys = [
            "precision_at_k", "recall_at_k", "hit_rate_at_k",
            "mrr", "ndcg_at_k", "faithfulness", "answer_relevance",
            "retrieval_latency_s", "full_rag_latency_s"
        ]
        aggregates = {
            f"mean_{k}": round(
                statistics.mean(r[k] for r in results), 3
            )
            for k in metric_keys
        }

        return {
            "results":    results,
            "aggregates": aggregates,
            "n_questions": len(results),
            "k":           self.k,
        }
