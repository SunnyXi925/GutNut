from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmnps.scoring.attribute_calibration import ATTRIBUTE_POINT_FRACTION_MODES
from gmnps.scoring.attribute_recomposition import FINAL_DEVIATION_CAP_MODES
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES
from gmnps.validation.attribute_synthetic_twin import (
    AttributeSyntheticExperimentConfig,
    AttributeSyntheticTwinConfig,
    benchmark_attribute_synthetic_twin,
    freeze_attribute_synthetic_experiment,
    run_attribute_synthetic_experiment,
    sattolo_derangement,
    simulate_attribute_synthetic_twin,
)
from gmnps.validation import attribute_synthetic_twin as twin_module


def _config(**changes: object) -> AttributeSyntheticTwinConfig:
    values: dict[str, object] = {
        "n_individuals": 8,
        "n_foods": 6,
        "seed": 1701,
        "noise_sd": 0.35,
    }
    values.update(changes)
    return AttributeSyntheticTwinConfig(**values)


@pytest.fixture(scope="module")
def default_bundle():
    return simulate_attribute_synthetic_twin(_config())


@pytest.fixture(scope="module")
def default_benchmark(default_bundle):
    return benchmark_attribute_synthetic_twin(
        default_bundle, bootstrap_replicates=40
    )


def test_truth_is_programmed_at_attribute_level_and_raw_noise_has_an_independent_stream():
    noisy = simulate_attribute_synthetic_twin(
        _config(n_individuals=20, n_foods=10)
    )
    noiseless = simulate_attribute_synthetic_twin(
        _config(n_individuals=20, n_foods=10, noise_sd=0.0)
    )

    pd.testing.assert_series_equal(noisy.universal_component, noiseless.universal_component)
    pd.testing.assert_frame_equal(noisy.attribute_effects, noiseless.attribute_effects)
    pd.testing.assert_frame_equal(noisy.score_beta, noiseless.score_beta)
    expected = noisy.universal_component.add(
        noisy.attribute_effects.groupby(level=["individual_id", "food_id"])[
            "true_effect"
        ].sum(),
        fill_value=0.0,
    ).clip(1.0, 100.0)
    pd.testing.assert_series_equal(
        noisy.noiseless_truth["response"], expected.rename("response")
    )
    observed_noise = noisy.observed_response["response"] - noisy.noiseless_truth["response"]
    assert np.array_equal(
        observed_noise.to_numpy(), noisy.effective_noise["effective_noise"].to_numpy()
    )
    unclipped = ~noisy.clipping_indicator["was_clipped"]
    np.testing.assert_allclose(
        noisy.raw_gaussian_noise.loc[unclipped, "raw_gaussian_noise"].to_numpy(),
        noisy.effective_noise.loc[unclipped, "effective_noise"].to_numpy(),
        rtol=0.0,
        atol=1e-14,
    )
    raw = noisy.raw_gaussian_noise["raw_gaussian_noise"]
    assert abs(np.corrcoef(raw, noisy.pair_design["capacity_MAC"])[0, 1]) < 0.25
    assert abs(np.corrcoef(raw, noisy.pair_design["capacity_LIPID"])[0, 1]) < 0.25
    assert noisy.rng_streams["noise"] != noisy.rng_streams["capacities"]
    assert noisy.truth_definition_sha256 == noiseless.truth_definition_sha256


def test_zero_effect_recovers_exact_fcs_without_centering():
    bundle = simulate_attribute_synthetic_twin(
        _config(effect_scale=0.0, noise_sd=0.0)
    )
    result = benchmark_attribute_synthetic_twin(bundle, bootstrap_replicates=20)
    locked = result.predictions.query("comparator == 'locked_attribute_gmnps'")
    fcs = result.predictions.query("comparator == 'fcs_baseline'")
    assert np.array_equal(locked["prediction"].to_numpy(), locked["FCS2"].to_numpy())
    assert np.array_equal(fcs["prediction"].to_numpy(), fcs["FCS2"].to_numpy())
    assert result.metrics.set_index("comparator").loc[
        "locked_attribute_gmnps", "residual_rmse"
    ] == 0.0


def test_locked_attribute_and_final_caps_are_numerically_respected():
    bundle = simulate_attribute_synthetic_twin(_config(noise_sd=0.0))
    result = benchmark_attribute_synthetic_twin(
        bundle, bootstrap_replicates=20, include_sensitivities=True
    )

    diagnostics = result.attribute_cap_diagnostics
    for mode, fraction in ATTRIBUTE_POINT_FRACTION_MODES.items():
        subset = diagnostics.loc[diagnostics["attribute_point_mode"].eq(mode)]
        for row in subset.itertuples(index=False):
            rule = FCS2_RULES[row.attribute]
            allowed = fraction * abs(float(rule.high_points) - float(rule.low_points))
            assert abs(row.attribute_point_delta) <= allowed + 1e-12

    final = result.final_cap_diagnostics
    for mode, cap in FINAL_DEVIATION_CAP_MODES.items():
        subset = final.loc[final["final_cap_mode"].eq(mode)]
        assert subset["GMNPS_delta"].abs().max() <= cap + 1e-12


def test_all_comparators_have_identical_person_food_opportunities_and_no_centering(
    default_bundle, default_benchmark
):
    expected_pairs = set(default_bundle.observed_response.index)
    counts = default_benchmark.predictions.groupby("comparator").size()
    assert counts.nunique() == 1
    assert counts.iloc[0] == len(expected_pairs)
    for _, frame in default_benchmark.predictions.groupby("comparator"):
        assert set(zip(frame["individual_id"], frame["food_id"])) == expected_pairs
    assert not default_benchmark.metrics["uses_population_or_per_food_centering"].any()
    assert {
        "fcs_baseline",
        "locked_attribute_gmnps",
        "legacy_final_score_offset",
        "unanchored_microbiome_score",
        "random_microbiome",
        "sattolo_deranged_microbiome",
        "original_mask_legacy_offset",
        "expert_revised_mask_legacy_offset",
    } == set(default_benchmark.metrics["comparator"])


def test_sattolo_derangement_has_no_fixed_points_and_is_deterministic():
    first = sattolo_derangement(31, seed=8128)
    second = sattolo_derangement(31, seed=8128)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(31))
    assert not np.any(first == np.arange(31))
    with pytest.raises(ValueError, match="at least two"):
        sattolo_derangement(1, seed=1)


def test_random_and_deranged_controls_lose_recovery_and_expert_mask_excludes_proxies(
    default_benchmark,
):
    metrics = default_benchmark.metrics.set_index("comparator")
    locked = metrics.loc["locked_attribute_gmnps"]
    assert locked["residual_rmse"] < metrics.loc["random_microbiome", "residual_rmse"]
    assert locked["residual_rmse"] < metrics.loc[
        "sattolo_deranged_microbiome", "residual_rmse"
    ]
    assert locked["residual_spearman"] > metrics.loc[
        "sattolo_deranged_microbiome", "residual_spearman"
    ]
    assert metrics.loc[
        "expert_revised_mask_legacy_offset", "residual_rmse"
    ] < metrics.loc["original_mask_legacy_offset", "residual_rmse"]
    assert metrics.loc[
        "expert_revised_mask_legacy_offset", "excluded_proxy_contribution_fraction"
    ] == 0.0
    assert metrics.loc[
        "original_mask_legacy_offset", "excluded_proxy_contribution_fraction"
    ] > 0.0


def test_multiseed_experiment_reports_ci_valid_n_and_code_fixed_checks(tmp_path):
    experiment = run_attribute_synthetic_experiment(
        AttributeSyntheticExperimentConfig(
            seeds=(1701, 1702, 1703),
            n_individuals=8,
            n_foods=6,
            noise_sd=0.35,
            bootstrap_replicates=30,
        )
    )
    assert set(experiment.replicate_metrics["seed"]) == {1701, 1702, 1703}
    assert experiment.summary["replicates_requested"].eq(3).all()
    assert experiment.summary["replicates_valid"].eq(3).all()
    assert experiment.summary[["ci_lower", "ci_upper"]].notna().all().all()
    assert (
        experiment.manifest["evidence_role"]
        == "correctly_specified_synthetic_positive_control"
    )
    assert experiment.manifest["data_class"] == "synthetic"
    assert experiment.manifest["checks_code_fixed_in_same_release"] is True
    assert experiment.manifest["independently_preregistered"] is False
    assert experiment.manifest["formal_run_cap_modes"] == {
        "attribute_point_fraction": "primary_20_percent",
        "final_deviation": "primary_plus_or_minus_12",
    }
    assert experiment.manifest["cap_sensitivities"] == (
        "10/30 percent attribute and +/-8/15 final caps are unit boundary tests only; "
        "they are not part of the frozen formal run"
    )
    assert set(experiment.manifest["source_sha256"]) >= {
        "simulator",
        "attribute_calibration",
        "attribute_gmnps",
        "attribute_recomposition",
    }
    assert experiment.manifest["source_sha256"]["simulator"] == sha256(
        Path(twin_module.__file__).read_bytes()
    ).hexdigest()
    assert experiment.manifest["environment"]["python_version"]
    assert experiment.manifest["environment"]["packages"]["numpy"]
    assert set(experiment.success_checks["criterion_id"]) == {
        "programmed_mapping_recovery_vs_fcs",
        "random_assignment_loses_recovery",
        "sattolo_assignment_loses_recovery",
        "locked_preserves_universal_rank",
        "expert_mask_excludes_programmed_proxies",
    }
    first = freeze_attribute_synthetic_experiment(experiment, tmp_path / "first")
    second = freeze_attribute_synthetic_experiment(experiment, tmp_path / "second")
    assert set(first) == set(second)
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()
        text = first[name].read_text()
        assert "correctly_specified_synthetic_positive_control" in text
        assert "synthetic" in text
        assert "sha256" in text
    assert '"seeds"' in first["manifest"].read_text()
    manifest_text = first["manifest"].read_text()
    assert "identifiability" not in manifest_text
    assert (
        "Locked implementation recovered the programmed mapping in a correctly "
        "specified synthetic positive-control and lost recovery after random or "
        "Sattolo-deranged assignment."
    ) in manifest_text
