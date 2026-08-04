from gmnps.validation.reference_targets import evaluate_target, reference_targets


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
