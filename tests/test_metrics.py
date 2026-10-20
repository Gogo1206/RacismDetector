import pytest

from racism_detector.metrics import best_threshold, evaluate


def test_best_threshold_separates_perfectly():
    y = [0, 0, 0, 1, 1]
    probs = [0.1, 0.2, 0.3, 0.7, 0.9]
    threshold = best_threshold(y, probs)
    assert 0.3 < threshold <= 0.7
    assert evaluate(y, probs, threshold, [True, False, False, False, False])["f1"] == 1.0


def test_fpr_race_supportive_counts_only_masked_rows():
    y = [0, 0, 0, 1]
    probs = [0.9, 0.9, 0.1, 0.9]
    supportive = [True, False, True, False]
    m = evaluate(y, probs, 0.5, supportive)
    assert m["fpr_race_supportive"] == pytest.approx(0.5)  # 1 of 2 supportive flagged
    assert m["fpr"] == pytest.approx(2 / 3)
