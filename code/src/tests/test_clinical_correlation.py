import pandas as pd

from gmnps.validation.clinical_correlation import (
    audit_clinical_completeness,
    choose_clinical_cohort,
    evaluate_gmwi2_retention,
    spearman_clinical_correlations,
)


def test_audit_clinical_completeness_and_choose_cra_when_gmrepo_missing():
    variables = ["age", "bmi", "fbg", "tg"]
    gmrepo = pd.DataFrame(
        {
            "sample_id": ["g1", "g2"],
            "age": [50, None],
            "bmi": [None, None],
            "fbg": [None, None],
            "tg": [1.2, None],
        }
    )
    cra = pd.DataFrame(
        {
            "sample_id": ["c1", "c2", "c3"],
            "age": [40, 50, 60],
            "bmi": [22, 24, 28],
            "fbg": [4.8, 5.5, 6.2],
            "tg": [1.0, 1.4, 2.1],
        }
    )

    gmrepo_audit = audit_clinical_completeness(gmrepo, variables, max_missing_fraction=0.40, min_n=3)
    cra_audit = audit_clinical_completeness(cra, variables, max_missing_fraction=0.40, min_n=3)

    assert not gmrepo_audit["passes"].all()
    assert cra_audit["passes"].all()
    assert choose_clinical_cohort(gmrepo_audit, cra_audit) == "CRA013939"


def test_spearman_correlations_and_retention_report():
    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "age": [30, 40, 50, 60],
            "bmi": [20, 22, 28, 31],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "gmwi2": [0.9, 0.7, 0.4, 0.2],
            "new_health": [0.85, 0.72, 0.45, 0.21],
        }
    )
    corr = spearman_clinical_correlations(features, metadata, ["age", "bmi"], ["gmwi2", "new_health"])
    assert set(corr["clinical_variable"]) == {"age", "bmi"}
    assert set(corr["feature"]) == {"gmwi2", "new_health"}

    baseline = corr.loc[corr["feature"].eq("gmwi2")]
    candidate = corr.loc[corr["feature"].eq("new_health")]
    retained = evaluate_gmwi2_retention(
        baseline,
        candidate,
        min_retained_fraction=0.80,
        max_abs_loss=0.05,
    )
    assert retained["passes_retention"].all()
