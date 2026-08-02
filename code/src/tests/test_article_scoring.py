import numpy as np
import pandas as pd
from pathlib import Path

from gmnps.scoring import (
    AnchoredScoringConfig,
    audit_primary_mask,
    bounded_centered_delta,
    score_individual_foods,
)
from gmnps.scoring.masks import EXPERT_REVISED_LIPID, EXPERT_REVISED_MAC
from gmnps.scoring.masks import EXPERT_REVISED_V4_MASK_VERSION, LEGACY_EXPERT_REVISED_MASK_VERSION, get_mask_definition
from gmnps.validation.preservation import nps_preservation_metrics
from gmnps.validation.synthetic_twin import run_synthetic_benchmark, simulate_synthetic_twin


def _toy_article_inputs():
    nutrients = sorted(EXPERT_REVISED_MAC | EXPERT_REVISED_LIPID | {"Carbohydrate (g)", "Zinc (mg)", "Copper (mg)", "Vitamin A, RAE (mcg_RAE)"})
    people = [f"p{i}" for i in range(8)]
    foods = [f"f{i}" for i in range(6)]
    rng = np.random.default_rng(4)
    weights = pd.DataFrame(rng.normal(size=(len(people), len(nutrients))), index=people, columns=nutrients)
    food_nutrients = pd.DataFrame(rng.gamma(2.0, 1.0, size=(len(foods), len(nutrients))), index=foods, columns=nutrients)
    food_meta = pd.DataFrame(
        {
            "food_name": [f"food {i}" for i in range(len(foods))],
            "food_group": ["plant", "plant", "mixed", "animal", "animal", "mixed"],
            "FCS2": [85, 78, 62, 42, 35, 55],
        },
        index=foods,
    )
    return weights, food_nutrients, food_meta


def test_expert_revised_mask_boundaries():
    weights, _, _ = _toy_article_inputs()
    audit = audit_primary_mask(weights.columns)
    assert audit["excluded_in_mac"] == []
    assert audit["excluded_in_lipid"] == []
    assert set(["Carbohydrate (g)", "Copper (mg)", "Zinc (mg)", "Vitamin A, RAE (mcg_RAE)"]).issubset(
        audit["excluded_present_as_columns"]
    )
    assert audit["retinol_present"] == ["Retinol (mcg)"]


def test_expert_revised_v4_alias_matches_legacy_mask():
    v4 = get_mask_definition(EXPERT_REVISED_V4_MASK_VERSION)
    legacy = get_mask_definition(LEGACY_EXPERT_REVISED_MASK_VERSION)
    assert v4.mac == legacy.mac
    assert v4.lipid == legacy.lipid
    assert "Carbohydrate (g)" not in v4.mac
    assert "Zinc (mg)" not in v4.mac
    assert "Copper (mg)" not in v4.mac
    assert "Vitamin A, RAE (mcg_RAE)" not in v4.lipid
    assert "Retinol (mcg)" in v4.lipid


def test_bounded_delta_is_centered_and_capped():
    rng = np.random.default_rng(1)
    raw = pd.DataFrame(rng.normal(size=(30, 5)), index=[f"p{i}" for i in range(30)], columns=[f"f{j}" for j in range(5)])
    delta = bounded_centered_delta(raw, AnchoredScoringConfig(delta_cap=12))
    assert np.all(delta.max(axis=0) <= 12 + 1e-9)
    assert np.all(delta.min(axis=0) >= -12 - 1e-9)
    assert np.allclose(delta.mean(axis=0), 0.0, atol=1e-8)


def test_score_outputs_article_schema_and_range():
    weights, nutrients, food_meta = _toy_article_inputs()
    individual_food, food_summary, manifest = score_individual_foods(weights, nutrients, food_meta)
    expected_columns = {
        "individual_id",
        "food_id",
        "food_name",
        "food_group",
        "FCS2",
        "GMNPS_delta",
        "GMNPS_score",
        "MAC_delta",
        "LIPID_delta",
        "top_positive_drivers",
        "top_negative_drivers",
        "mask_version",
        "scoring_version",
    }
    assert expected_columns.issubset(individual_food.columns)
    assert individual_food["GMNPS_score"].between(1, 100).all()
    assert np.allclose(individual_food.groupby("food_id")["GMNPS_delta"].mean(), 0.0, atol=1e-8)
    assert {"GMNPS_mean", "MAC_variance", "LIPID_variance", "dominant_channel"}.issubset(food_summary.columns)
    assert "Vitamin A, RAE (mcg_RAE)" not in manifest["lipid_nutrients"]
    assert "Retinol (mcg)" in manifest["lipid_nutrients"]
    channel_abs = individual_food["MAC_delta"].abs() + individual_food["LIPID_delta"].abs()
    assert np.all(channel_abs <= individual_food["GMNPS_delta"].abs() + 1e-9)


def test_nps_preservation_metric_passes_on_midrange_scores():
    weights, nutrients, food_meta = _toy_article_inputs()
    individual_food, _, _ = score_individual_foods(weights, nutrients, food_meta, AnchoredScoringConfig(delta_cap=3))
    metrics = nps_preservation_metrics(individual_food, min_rho=0.80)
    assert metrics["passes_min_rho"]
    assert metrics["spearman_fcs2_gmnps_mean"] >= 0.80


def test_synthetic_twin_benchmark_expected_tradeoff():
    bundle = simulate_synthetic_twin(n_individuals=60, n_foods=36, seed=10)
    bench = run_synthetic_benchmark(bundle=bundle, seed=10).set_index("model")
    assert bench.loc["anchored GMNPS", "individual_response_spearman"] > bench.loc["FCS2 only", "individual_response_spearman"]
    assert not bench.loc["FCS2 only", "personalized_residual_defined"]
    assert pd.isna(bench.loc["FCS2 only", "personalized_residual_spearman"])
    assert bench.loc["anchored GMNPS", "personalized_residual_defined"]
    assert bench.loc["anchored GMNPS", "personalized_residual_spearman"] > bench.loc["shuffled microbiome", "personalized_residual_spearman"]
    assert bench.loc["anchored GMNPS", "personalized_residual_spearman"] > bench.loc["random microbiome", "personalized_residual_spearman"]
    assert bench.loc["anchored GMNPS", "nps_preservation_spearman"] > bench.loc["unanchored microbiome score", "nps_preservation_spearman"]
    assert bench.loc["anchored GMNPS", "individual_response_spearman"] > bench.loc["shuffled microbiome", "individual_response_spearman"]


def test_article_bundle_export(tmp_path):
    from gmnps.manuscript import export_article_bundle
    from gmnps.validation.preservation import heterogeneity_metrics, nps_preservation_metrics

    weights, nutrients, food_meta = _toy_article_inputs()
    individual_food, food_summary, manifest = score_individual_foods(weights, nutrients, food_meta)
    paths = export_article_bundle(
        tmp_path,
        individual_food,
        food_summary,
        manifest,
        preservation_metrics=nps_preservation_metrics(individual_food, min_rho=0.80),
        heterogeneity=heterogeneity_metrics(individual_food),
    )
    assert {"individual_food_csv", "food_summary_csv", "manifest_json", "top_food_summary_tex"}.issubset(paths)
    for path in paths.values():
        assert Path(path).exists()


def test_run_report_json_uses_null_for_undefined_residual(tmp_path):
    import json
    from scripts.run_nature_food_article import run_synthetic

    class Args:
        output_dir = tmp_path
        n_individuals = 20
        n_foods = 12
        seed = 10
        delta_cap = 12.0
        min_preservation_rho = 0.80

    run_synthetic(Args())
    report = json.loads((tmp_path / "run_synthetic_report.json").read_text())
    fcs2 = next(row for row in report["benchmark"] if row["model"] == "FCS2 only")
    assert fcs2["personalized_residual_spearman"] is None
    assert fcs2["personalized_residual_defined"] is False
