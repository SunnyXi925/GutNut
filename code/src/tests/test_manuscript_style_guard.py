from gmnps.manuscript import check_academic_style


def test_style_guard_flags_ai_like_punctuation_and_zombie_nouns():
    text = (
        'This "novel" and "transformative" framework — through the utilization of optimization of '
        "dietary calibration — enables the facilitation of precision nutrition."
    )
    issues = check_academic_style(text)
    codes = {issue.code for issue in issues}
    assert "excessive_quotes" in codes
    assert "em_dash" in codes
    assert "zombie_noun" in codes


def test_style_guard_accepts_restrained_academic_sentence():
    text = (
        "Personalized calibration adds bounded individual adaptation to a "
        "universal nutrition prior while preserving population-level guidance."
    )
    assert check_academic_style(text) == []
