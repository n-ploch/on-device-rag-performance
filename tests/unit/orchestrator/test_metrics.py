"""Tests defining all metric functions in orchestrator/metrics.py"""

import pytest

from orchestrator.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    detect_abstention,
    abstention_rate,
)


class TestRecallAtK:
    """Tests for recall@k metric."""

    def test_perfect_retrieval(self):
        """All relevant docs retrieved -> recall@k = 1.0"""
        assert recall_at_k(["d1", "d2", "d3"], {"d1", "d2", "d3"}, k=3) == 1.0

    def test_partial_retrieval(self):
        """2 of 4 relevant in top 3 -> recall@3 = 0.5"""
        assert recall_at_k(["d1", "d2", "d3"], {"d1", "d3", "d5", "d6"}, k=3) == 0.5

    def test_no_relevant(self):
        """No relevant docs retrieved -> recall = 0.0"""
        assert recall_at_k(["d1", "d2"], {"d3", "d4"}, k=2) == 0.0

    def test_empty_relevant_set(self):
        """Empty relevant set -> recall = 0.0"""
        assert recall_at_k(["d1"], set(), k=1) == 0.0

    def test_k_zero_raises(self):
        """k=0 should raise ValueError."""
        with pytest.raises(ValueError):
            recall_at_k(["d1"], {"d1"}, k=0)

    def test_k_negative_raises(self):
        """k<0 should raise ValueError."""
        with pytest.raises(ValueError):
            recall_at_k(["d1"], {"d1"}, k=-1)

    def test_k_larger_than_retrieved(self):
        """k > len(retrieved) uses all retrieved."""
        assert recall_at_k(["d1"], {"d1", "d2"}, k=10) == 0.5

    def test_empty_retrieved(self):
        """Empty retrieved -> recall = 0.0"""
        assert recall_at_k([], {"d1"}, k=3) == 0.0


class TestPrecisionAtK:
    """Tests for precision@k metric."""

    def test_all_relevant(self):
        """All top-k relevant -> precision@k = 1.0"""
        assert precision_at_k(["d1", "d2"], {"d1", "d2", "d3"}, k=2) == 1.0

    def test_half_relevant(self):
        """2 relevant in top 4 -> precision@4 = 0.5"""
        assert precision_at_k(["d1", "d2", "d3", "d4"], {"d1", "d3"}, k=4) == 0.5

    def test_none_relevant(self):
        """No relevant in top k -> precision@k = 0.0"""
        assert precision_at_k(["d1", "d2"], {"d3"}, k=2) == 0.0

    def test_k_larger_than_retrieved(self):
        """k > len(retrieved) uses actual count."""
        assert precision_at_k(["d1"], {"d1"}, k=10) == 1.0

    def test_empty_retrieved(self):
        """Empty retrieved -> precision = 0.0"""
        assert precision_at_k([], {"d1"}, k=3) == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank metric."""

    def test_first_result_relevant(self):
        """First result relevant -> MRR = 1.0"""
        assert mrr(["d1", "d2"], {"d1"}) == 1.0

    def test_second_result_relevant(self):
        """First relevant at position 2 -> MRR = 0.5"""
        assert mrr(["d1", "d2"], {"d2"}) == 0.5

    def test_third_result_relevant(self):
        """First relevant at position 3 -> MRR = 1/3"""
        assert mrr(["d1", "d2", "d3"], {"d3"}) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        """No relevant in results -> MRR = 0"""
        assert mrr(["d1", "d2"], {"d3"}) == 0.0

    def test_multiple_relevant(self):
        """Multiple relevant -> uses first one."""
        assert mrr(["d1", "d2", "d3"], {"d2", "d3"}) == 0.5

    def test_empty_retrieved(self):
        """Empty retrieved -> MRR = 0.0"""
        assert mrr([], {"d1"}) == 0.0

    def test_empty_relevant(self):
        """Empty relevant -> MRR = 0.0"""
        assert mrr(["d1", "d2"], set()) == 0.0


class TestDetectAbstention:
    """Tests for abstention detection."""

    def test_i_dont_know_pattern(self):
        """Detect 'I don't know' pattern."""
        assert detect_abstention("I don't know the answer.") is True

    def test_insufficient_info_pattern(self):
        """Detect 'insufficient information' pattern."""
        assert detect_abstention("There is insufficient information.") is True

    def test_cannot_determine_pattern(self):
        """Detect 'cannot determine' pattern."""
        assert detect_abstention("I cannot determine the answer.") is True

    def test_normal_answer(self):
        """Normal answer is not abstention."""
        assert detect_abstention("Aspirin reduces pain.") is False

    def test_empty_output(self):
        """Empty output is abstention."""
        assert detect_abstention("") is True

    def test_case_insensitive(self):
        """Pattern matching is case-insensitive."""
        assert detect_abstention("I DON'T KNOW") is True


class TestAbstentionRate:
    """Tests for abstention rate calculation."""

    def test_no_abstentions(self):
        """All answers provided -> rate = 0.0"""
        outputs = ["Answer 1", "Answer 2"]
        assert abstention_rate(outputs) == 0.0

    def test_all_abstentions(self):
        """All abstentions -> rate = 1.0"""
        outputs = ["I don't know", ""]
        assert abstention_rate(outputs) == 1.0

    def test_mixed(self):
        """Half abstentions -> rate = 0.5"""
        outputs = ["Answer", "I don't know", "Another answer", "I cannot determine the answer."]
        assert abstention_rate(outputs) == 0.5

    def test_empty_list(self):
        """Empty list -> rate = 0.0"""
        assert abstention_rate([]) == 0.0
