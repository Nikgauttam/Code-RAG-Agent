from evaluation.metrics import mean, percentile, recall_at_k, reciprocal_rank


def test_recall_at_k_basic():
    assert recall_at_k(["a", "b", "c"], ["b"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["d"], k=3) == 0.0
    # Two expected, only one in top-2 => 0.5
    assert recall_at_k(["a", "b", "c"], ["b", "d"], k=2) == 0.5


def test_recall_at_k_dedups():
    # Duplicates should not artificially inflate the top-k window.
    assert recall_at_k(["a", "a", "a", "b"], ["b"], k=3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["d"]) == 0.0


def test_mean_and_percentile():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([], 0.5) == 0.0
