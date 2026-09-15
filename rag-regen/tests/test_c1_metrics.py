import math

import pytest

from ragregen import c1


def test_confusion_counts_each_cell():
    y_true = [True, True, False, False]
    y_pred = [True, False, True, False]
    m = c1.confusion(y_true, y_pred)
    assert (m.tp, m.fn, m.fp, m.tn) == (1, 1, 1, 1)
    assert m.n == 4


def test_confusion_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        c1.confusion([True], [True, False])


def test_perfect_prediction():
    m = c1.confusion([True, False], [True, False])
    assert m.precision == 1.0 and m.recall == 1.0
    assert m.balanced_accuracy == 1.0
    assert m.mcc == 1.0


def test_always_fail_on_the_real_base_rate_is_exposed_by_balanced_accuracy():
    """15 fail / 7 pass. An always-FAIL verifier must look good on F1 and
    useless on balanced accuracy and MCC -- that contrast is the whole reason
    those two are the headline metrics."""
    y_true = [True] * 15 + [False] * 7
    m = c1.confusion(y_true, [True] * 22)
    assert m.recall == 1.0
    assert m.precision == pytest.approx(15 / 22)
    assert m.f1 == pytest.approx(2 * (15 / 22) / ((15 / 22) + 1))
    assert m.f1 > 0.8                      # flattering
    assert m.balanced_accuracy == 0.5      # chance
    assert m.mcc == 0.0                    # no information


def test_always_pass_is_also_at_chance():
    y_true = [True] * 15 + [False] * 7
    m = c1.confusion(y_true, [False] * 22)
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.balanced_accuracy == 0.5
    assert m.mcc == 0.0


def test_metrics_are_zero_not_nan_when_a_denominator_vanishes():
    m = c1.confusion([False, False], [False, False])
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0
    assert m.mcc == 0.0
    assert not any(math.isnan(v) for v in (m.precision, m.recall, m.f1, m.mcc))


def test_mcc_is_negative_when_prediction_is_inverted():
    y_true = [True, True, False, False]
    m = c1.confusion(y_true, [False, False, True, True])
    assert m.mcc == -1.0


def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = c1.wilson_ci(15, 22)
    assert lo < 15 / 22 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_ci_matches_a_known_value():
    """k=15, n=22, z=1.96 -> (0.4732, 0.8364), computed independently."""
    lo, hi = c1.wilson_ci(15, 22)
    assert lo == pytest.approx(0.4732, abs=1e-3)
    assert hi == pytest.approx(0.8364, abs=1e-3)


def test_wilson_ci_is_defined_at_the_boundaries():
    assert c1.wilson_ci(0, 10)[0] == 0.0
    assert c1.wilson_ci(10, 10)[1] == 1.0


def test_wilson_ci_of_an_empty_sample_is_the_whole_interval():
    assert c1.wilson_ci(0, 0) == (0.0, 1.0)
