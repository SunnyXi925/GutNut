from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
APPROVED_TITLE = "Personalized calibration transforms nutrient profiling systems for precision nutrition"
OLD_TITLE = "Gut microbiome-informed nutrient profiling reconciles universal dietary guidelines with personalized metabolic traits"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_config_uses_approved_title():
    config = read("code/src/configs/nature_food_article.yaml")
    assert f"title: {APPROVED_TITLE}" in config
    assert OLD_TITLE not in config


def test_public_docs_do_not_use_old_title():
    for path in ["README.md", "code/README.md"]:
        text = read(path)
        assert APPROVED_TITLE in text
        assert OLD_TITLE not in text


def test_baseline_rationale_names_attribute_coverage():
    phrase = "broad and up-to-date NPS attribute coverage"
    for path in ["README.md", "code/README.md", "manuscript/narrative_blueprint.md"]:
        assert phrase in read(path)


def test_narrative_blueprint_covers_results_and_discussion():
    text = read("manuscript/narrative_blueprint.md")
    required_phrases = [
        APPROVED_TITLE,
        "Static NPS establishes the universal prior",
        "Personalized calibration is bounded and interpretable",
        "Population consensus is preserved",
        "Individual heterogeneity is revealed",
        "Gut microbiome provides a proof-of-concept calibration layer",
        "Digital-gut-twin and retrospective validation support feasibility",
        "Discussion returns to NPS transformation",
        "not clinical intervention efficacy",
    ]
    for phrase in required_phrases:
        assert phrase in text


def test_manuscript_style_guard_rules():
    text = read("manuscript/narrative_blueprint.md")
    assert "—" not in text
    assert text.count('"') <= 2
    assert text.count("'") <= 12
    discouraged_terms = [
        "utilization",
        "implementation of",
        "facilitation of",
        "optimization of",
        "robustness of the framework",
    ]
    lowered = text.lower()
    for term in discouraged_terms:
        assert term not in lowered
