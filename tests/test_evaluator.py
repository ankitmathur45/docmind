"""Unit tests for retrieval and answer-quality metrics in src/evaluator.py."""
import pytest
from src.evaluator import (
    is_relevant,
    get_relevance_score,
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    measure_latency,
    AnswerEvaluator,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

RELEVANT = [
    "The capital of France is Paris, a major European city.",
    "Python is a high-level programming language known for readability.",
]

RETRIEVED_GOOD = [
    "The capital of France is Paris, a major European city.",   # exact match → relevant
    "Python is a high-level programming language known for readability.",  # exact match
    "Unrelated chunk about database indexing strategies.",
]

RETRIEVED_NONE = [
    "Unrelated chunk about database indexing strategies.",
    "Another unrelated chunk about cloud computing.",
    "Something about machine learning pipelines.",
]


# ── is_relevant ─────────────────────────────────────────────────────────────

class TestIsRelevant:
    def test_exact_match(self):
        assert is_relevant(RELEVANT[0], RELEVANT) is True

    def test_no_match(self):
        assert is_relevant("completely unrelated text about dinosaurs", RELEVANT) is False

    def test_fuzzy_near_match(self):
        slightly_different = "The capital of France is Paris, a major European city!"
        assert is_relevant(slightly_different, RELEVANT) is True

    def test_empty_relevant_list(self):
        assert is_relevant("some text", []) is False


# ── get_relevance_score ──────────────────────────────────────────────────────

class TestGetRelevanceScore:
    def test_exact_key_returns_score(self):
        scores = {RELEVANT[0]: 2, RELEVANT[1]: 1}
        assert get_relevance_score(RELEVANT[0], scores) == 2

    def test_no_match_returns_zero(self):
        scores = {RELEVANT[0]: 2}
        assert get_relevance_score("completely different text", scores) == 0

    def test_returns_best_score(self):
        scores = {RELEVANT[0]: 1, RELEVANT[1]: 3}
        # chunk matches RELEVANT[1] best
        assert get_relevance_score(RELEVANT[1], scores) == 3


# ── precision_at_k ──────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = RELEVANT[:2]
        assert precision_at_k(retrieved, RELEVANT, k=2) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(RETRIEVED_NONE, RELEVANT, k=3) == 0.0

    def test_partial_precision(self):
        mixed = [RELEVANT[0], "unrelated", "also unrelated"]
        result = precision_at_k(mixed, RELEVANT, k=3)
        assert abs(result - 1/3) < 0.01

    def test_k_larger_than_retrieved(self):
        result = precision_at_k([RELEVANT[0]], RELEVANT, k=3)
        assert abs(result - 1/3) < 0.01

    def test_k_zero_returns_zero(self):
        assert precision_at_k(RETRIEVED_GOOD, RELEVANT, k=0) == 0.0


# ── recall_at_k ─────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(RETRIEVED_GOOD, RELEVANT, k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(RETRIEVED_NONE, RELEVANT, k=3) == 0.0

    def test_partial_recall(self):
        one_hit = [RELEVANT[0], "unrelated", "also unrelated"]
        result = recall_at_k(one_hit, RELEVANT, k=3)
        assert abs(result - 0.5) < 0.01

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(RETRIEVED_GOOD, [], k=3) == 0.0


# ── hit_rate_at_k ───────────────────────────────────────────────────────────

class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k([RELEVANT[0], "unrelated"], RELEVANT, k=2) == 1.0

    def test_no_hit(self):
        assert hit_rate_at_k(RETRIEVED_NONE, RELEVANT, k=3) == 0.0

    def test_hit_beyond_k_not_counted(self):
        # relevant chunk is at position 3 (index 2), k=2 → should miss
        retrieved = ["unrelated1", "unrelated2", RELEVANT[0]]
        assert hit_rate_at_k(retrieved, RELEVANT, k=2) == 0.0


# ── mean_reciprocal_rank ─────────────────────────────────────────────────────

class TestMeanReciprocalRank:
    def test_first_position(self):
        retrieved = [RELEVANT[0], "unrelated"]
        assert mean_reciprocal_rank(retrieved, RELEVANT) == 1.0

    def test_second_position(self):
        retrieved = ["unrelated", RELEVANT[0]]
        assert abs(mean_reciprocal_rank(retrieved, RELEVANT) - 0.5) < 0.01

    def test_no_relevant(self):
        assert mean_reciprocal_rank(RETRIEVED_NONE, RELEVANT) == 0.0

    def test_third_position(self):
        retrieved = ["u1", "u2", RELEVANT[0]]
        assert abs(mean_reciprocal_rank(retrieved, RELEVANT) - 1/3) < 0.01


# ── ndcg_at_k ───────────────────────────────────────────────────────────────

class TestNdcgAtK:
    def test_perfect_order(self):
        scores = {RELEVANT[0]: 2, RELEVANT[1]: 1}
        retrieved = [RELEVANT[0], RELEVANT[1], "unrelated"]
        result = ndcg_at_k(retrieved, scores, k=3)
        assert abs(result - 1.0) < 0.01

    def test_no_relevant_returns_zero(self):
        scores = {RELEVANT[0]: 2}
        assert ndcg_at_k(RETRIEVED_NONE, scores, k=3) == 0.0

    def test_score_between_zero_and_one(self):
        scores = {RELEVANT[0]: 2, RELEVANT[1]: 1}
        retrieved = [RELEVANT[1], RELEVANT[0], "unrelated"]  # reversed order
        result = ndcg_at_k(retrieved, scores, k=3)
        assert 0.0 <= result <= 1.0


# ── measure_latency ──────────────────────────────────────────────────────────

class TestMeasureLatency:
    def test_returns_result_and_timing(self):
        def identity(x):
            return x * 2

        stats = measure_latency(identity, 5, n_runs=2)
        assert stats["result"] == 10
        assert stats["mean"] >= 0
        assert stats["min"] >= 0
        assert stats["max"] >= stats["min"]
        assert "std" in stats

    def test_single_run(self):
        stats = measure_latency(lambda: 42, n_runs=1)
        assert stats["result"] == 42
        assert stats["std"] == 0.0


# ── AnswerEvaluator — init-only (no Ollama calls) ───────────────────────────

class TestAnswerEvaluatorInit:
    def test_instantiates_without_error(self):
        evaluator = AnswerEvaluator()
        assert hasattr(evaluator, "faithfulness_chain")
        assert hasattr(evaluator, "relevance_chain")

    def test_parse_score_valid_json(self):
        evaluator = AnswerEvaluator()
        result = evaluator._parse_score('{"score": 0.8, "reason": "good"}')
        assert result["score"] == 0.8
        assert result["reason"] == "good"

    def test_parse_score_embedded_in_text(self):
        evaluator = AnswerEvaluator()
        result = evaluator._parse_score('Here is the result: {"score": 0.5, "reason": "ok"} done.')
        assert result["score"] == 0.5

    def test_parse_score_invalid_returns_zero(self):
        evaluator = AnswerEvaluator()
        result = evaluator._parse_score("not json at all")
        assert result["score"] == 0.0
