import pytest

import gmnps.validation.reference_targets as reference_target_module
from gmnps.validation.reference_targets import ReferenceTarget, evaluate_target, reference_targets


def test_reference_targets_include_required_benchmarks():
    targets = reference_targets()
    assert targets["fcs2_population_spearman"].threshold == 0.95
    assert targets["gmwi2_external_balanced_accuracy"].threshold == 0.72
    assert targets["kg_binary_balanced_accuracy"].threshold == 0.65


def test_evaluate_target_reports_pass_and_margin():
    passed = evaluate_target("fcs2_population_spearman", 0.98)
    failed = evaluate_target("kg_binary_balanced_accuracy", 0.58)
    assert passed["passes"] is True
    assert passed["margin"] == 0.03
    assert failed["passes"] is False
    assert failed["direction"] == ">="


def test_evaluate_target_covers_every_registered_target():
    targets = reference_targets()

    for name, target in targets.items():
        observed = target.threshold + 0.01 if target.direction == ">=" else target.threshold - 0.01
        result = evaluate_target(name, observed)
        assert result["name"] == name
        assert result["passes"] is True
        assert result["direction"] == target.direction


def test_evaluate_target_supports_less_than_or_equal_target(monkeypatch):
    monkeypatch.setattr(
        reference_target_module,
        "reference_targets",
        lambda: {
            "lower_is_better": ReferenceTarget(
                "lower_is_better", 0.20, "<=", "Synthetic test target."
            )
        },
    )

    passed = evaluate_target("lower_is_better", 0.18)
    failed = evaluate_target("lower_is_better", 0.22)
    assert passed["passes"] is True
    assert passed["margin"] == 0.02
    assert failed["passes"] is False
    assert failed["margin"] == -0.02


def test_evaluate_target_rejects_unknown_target():
    with pytest.raises(KeyError, match="unknown reference target"):
        evaluate_target("missing_target", 0.5)
