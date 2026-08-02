# Title Narrative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the GMNPS repository so the approved title and precision-ready NPS storyline guide the config, documentation, manuscript narrative outline and writing-style checks.

**Architecture:** Treat the title narrative as a repository-level contract. The config stores the canonical title, README files communicate the public framing, `manuscript/narrative_blueprint.md` defines the paper-wide Results and Discussion order, and a small style guard prevents obvious AI-like writing artifacts before manuscript drafting expands.

**Tech Stack:** Markdown, YAML, Python standard library, pytest-compatible tests run by direct function execution in the existing bundled Python runtime.

## Global Constraints

- Approved title: `Personalized calibration transforms nutrient profiling systems for precision nutrition`.
- Main storyline: NPS methodological transformation from static population scores to anchored personalized calibration.
- Significance framing: precision nutrition needs NPS that preserve population nutrition consensus while representing individual metabolic heterogeneity.
- GMNPS framing: gut microbiome-informed nutrient responses are the pivotal proof-of-concept calibration layer, not the only possible personalization layer.
- Baseline framing: Food Compass 2.0 is the currently selected anchored baseline because it offers broad and up-to-date NPS attribute coverage, but it should be described as the baseline prior rather than the manuscript's main subject.
- Evidence boundary: retrospective and synthetic validation support feasibility and plausibility, not causal clinical intervention efficacy.
- Writing style: avoid excessive quotation marks, avoid em dashes, avoid zombie nouns, prefer concrete nouns and active verbs, keep academic claims precise and restrained.
- Repository hygiene: do not modify or commit the untracked `.joycode/` directory.

---

### Task 1: Add Narrative Contract Tests

**Files:**
- Create: `code/src/tests/test_title_narrative.py`

**Interfaces:**
- Consumes: repository text files at `README.md`, `code/README.md`, `code/src/configs/nature_food_article.yaml`, `docs/superpowers/specs/2026-08-02-title-positioning-design.md`, and later `manuscript/narrative_blueprint.md`.
- Produces: executable tests named `test_config_uses_approved_title`, `test_public_docs_do_not_use_old_title`, `test_narrative_blueprint_covers_results_and_discussion`, and `test_manuscript_style_guard_rules`.

- [ ] **Step 1: Write the failing tests**

Create `code/src/tests/test_title_narrative.py` with exactly:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/code/src"
/Users/fengxi.25/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from tests import test_title_narrative as t
for name in sorted(dir(t)):
    if name.startswith("test_"):
        print("RUN", name)
        getattr(t, name)()
PY
```

Expected: FAIL because `nature_food_article.yaml` still contains the old title and `manuscript/narrative_blueprint.md` does not exist.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add code/src/tests/test_title_narrative.py
git commit -m "test: add title narrative contract"
```

Expected: commit succeeds and `.joycode/` remains untracked.

---

### Task 2: Update Canonical Title in Config and Public Documentation

**Files:**
- Modify: `code/src/configs/nature_food_article.yaml`
- Modify: `README.md`
- Modify: `code/README.md`

**Interfaces:**
- Consumes: `APPROVED_TITLE` from Task 1 tests.
- Produces: repository-level public framing that uses the approved title and removes old title text.

- [ ] **Step 1: Update config title**

In `code/src/configs/nature_food_article.yaml`, replace:

```yaml
  title: Gut microbiome-informed nutrient profiling reconciles universal dietary guidelines with personalized metabolic traits
```

with:

```yaml
  title: Personalized calibration transforms nutrient profiling systems for precision nutrition
```

- [ ] **Step 2: Update root README opening**

In `README.md`, replace the opening article description:

```markdown
The source code of the article framework "Harnessing Gut Microbiome with
Machine Learning to Advance Nutrient Profiling".
```

with:

```markdown
The source code of the article framework **Personalized calibration transforms
nutrient profiling systems for precision nutrition**.
```

Then replace:

```markdown
Gut Microbiome-informed Nutrient Profiling System (GMNPS) is implemented here
as a Food Compass 2.0-anchored Nature Food article framework:
```

with:

```markdown
Gut Microbiome-informed Nutrient Profiling System (GMNPS) is implemented here
as a proof-of-concept for anchored personalized calibration in nutrient
profiling:
```

- [ ] **Step 3: Update code README opening**

In `code/README.md`, after the `# GMNPS Reproducible Pipeline` heading, insert:

```markdown
Article title: **Personalized calibration transforms nutrient profiling systems
for precision nutrition**.
```

Then keep the existing explanation of the selected baseline NPS as the universal prior and bounded microbiome-defined personalized deviation. Food Compass 2.0 may be named once as the current baseline implementation because of its broad attribute coverage, but do not make it the paragraph's subject.

- [ ] **Step 4: Run title tests**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/code/src"
/Users/fengxi.25/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from tests import test_title_narrative as t
for name in ["test_config_uses_approved_title", "test_public_docs_do_not_use_old_title"]:
    print("RUN", name)
    getattr(t, name)()
print("TITLE DOC TESTS OK")
PY
```

Expected: PASS with `TITLE DOC TESTS OK`.

- [ ] **Step 5: Commit title documentation changes**

Run:

```bash
git add code/src/configs/nature_food_article.yaml README.md code/README.md
git commit -m "docs: update GMNPS article title"
```

Expected: commit succeeds.

---

### Task 3: Add Manuscript Narrative Blueprint

**Files:**
- Create: `manuscript/narrative_blueprint.md`

**Interfaces:**
- Consumes: title positioning spec at `docs/superpowers/specs/2026-08-02-title-positioning-design.md`.
- Produces: a section-level writing blueprint that future manuscript drafting must follow.

- [ ] **Step 1: Create the blueprint**

Create `manuscript/narrative_blueprint.md` with exactly:

```markdown
# GMNPS Manuscript Narrative Blueprint

## Title

Personalized calibration transforms nutrient profiling systems for precision nutrition

## Central Claim

Nutrient profiling systems can move from static population scores to precision-ready scoring by adding bounded personalized calibration on top of a universal nutrition prior. GMNPS tests this idea by using gut microbiome-informed nutrient responses as a proof-of-concept calibration layer.

## Abstract Opening

Nutrient profiling systems rank foods for population-level dietary guidance, yet static scores cannot represent interindividual metabolic heterogeneity required for precision nutrition. We propose an anchored personalized calibration framework that preserves population nutrition consensus while introducing bounded individual deviations. Using gut microbiome-informed nutrient responses as a proof of concept, GMNPS shows that nutrient profiling can extend toward precision nutrition without abandoning universal dietary guidance.

## Results Architecture

### 1. Static NPS establishes the universal prior

Open Results by defining the selected static NPS baseline as the population-level nutrition consensus. Food Compass 2.0 is the current baseline implementation because of its broad attribute coverage, but the purpose is to show what GMNPS preserves before introducing personalization.

### 2. Personalized calibration is bounded and interpretable

Present the score architecture, the personalized deviation term and the expert-reviewed MAC and lipid channels. Explain that calibration adjusts the baseline score within a bounded range rather than replacing the baseline score.

### 3. Population consensus is preserved

Show that population-mean GMNPS remains highly concordant with the baseline NPS and does not create implausible food-group reversals. This result protects the paper from being read as an unanchored recommendation system.

### 4. Individual heterogeneity is revealed

Show that static NPS has no individual variance while GMNPS creates non-random individual-food variation. Organize the evidence by food group and by mechanistic channel so heterogeneity remains interpretable.

### 5. Gut microbiome provides a proof-of-concept calibration layer

Present microbiome-informed nutrient responses as the biological layer that gives the deviation term meaning. Keep the claim focused on feasibility and interpretability rather than clinical efficacy.

### 6. Digital-gut-twin and retrospective validation support feasibility

Benchmark anchored GMNPS against FCS2-only, unanchored microbiome scoring, random microbiome and shuffled controls. The target pattern is high NPS preservation plus improved personalized signal.

## Discussion Architecture

### Discussion returns to NPS transformation

The Discussion should start from the title claim: personalized calibration transforms NPS by adding bounded individual adaptation to a universal nutrition prior.

### Conceptual implication

The central advance is not a replacement for the baseline NPS. It is a general architecture for precision-ready NPS.

### Biological implication

Gut microbiome is a pivotal proof-of-concept because it links diet, microbial metabolism and host metabolic heterogeneity.

### Methodological implication

Anchoring keeps personalized models from drifting into unstructured recommendation systems. Bounded deviations preserve public-health interpretability.

### Evidence boundary

Retrospective and synthetic validation support feasibility and plausibility, not clinical intervention efficacy.

### Future direction

Future NPS can incorporate genetics, metabolomics, continuous glucose response or lifestyle context if each layer remains anchored, bounded and interpretable.

## Writing Rules

- Use `personalized calibration` in the title, abstract, introduction and discussion.
- Use `anchored personalized deviation` when explaining the formula.
- Avoid excessive quotation marks.
- Avoid em dashes.
- Avoid zombie nouns such as utilization, facilitation and optimization when a direct verb works.
- Prefer concrete nouns, active verbs and restrained causal language.
- Do not claim clinical intervention efficacy.
- Do not frame GMNPS as replacing Food Compass 2.0.
```

- [ ] **Step 2: Run blueprint tests**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/code/src"
/Users/fengxi.25/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from tests import test_title_narrative as t
for name in ["test_narrative_blueprint_covers_results_and_discussion", "test_manuscript_style_guard_rules"]:
    print("RUN", name)
    getattr(t, name)()
print("BLUEPRINT TESTS OK")
PY
```

Expected: PASS with `BLUEPRINT TESTS OK`.

- [ ] **Step 3: Commit blueprint**

Run:

```bash
git add manuscript/narrative_blueprint.md
git commit -m "docs: add manuscript narrative blueprint"
```

Expected: commit succeeds.

---

### Task 4: Add Reusable Manuscript Style Guard

**Files:**
- Create: `code/src/gmnps/manuscript/style_guard.py`
- Modify: `code/src/gmnps/manuscript/__init__.py`
- Create: `code/src/tests/test_manuscript_style_guard.py`

**Interfaces:**
- Produces: `StyleIssue` dataclass and `check_academic_style(text: str) -> list[StyleIssue]`.
- Produces: importable public API `from gmnps.manuscript import check_academic_style`.

- [ ] **Step 1: Write style guard tests**

Create `code/src/tests/test_manuscript_style_guard.py` with exactly:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/code/src"
/Users/fengxi.25/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from tests import test_manuscript_style_guard as t
for name in sorted(dir(t)):
    if name.startswith("test_"):
        print("RUN", name)
        getattr(t, name)()
PY
```

Expected: FAIL because `check_academic_style` is not exported yet.

- [ ] **Step 3: Implement style guard**

Create `code/src/gmnps/manuscript/style_guard.py` with exactly:

```python
"""Lightweight academic writing checks for GMNPS manuscript drafts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleIssue:
    """A manuscript style issue that should be reviewed before submission."""

    code: str
    message: str
    count: int


ZOMBIE_NOUNS = (
    "utilization",
    "implementation of",
    "facilitation of",
    "optimization of",
    "robustness of the framework",
)


def check_academic_style(text: str) -> list[StyleIssue]:
    """Return style issues that often make scientific writing sound automated."""

    issues: list[StyleIssue] = []
    quote_count = text.count('"')
    if quote_count > 2:
        issues.append(
            StyleIssue(
                code="excessive_quotes",
                message="Use quotation marks sparingly in manuscript prose.",
                count=quote_count,
            )
        )

    dash_count = text.count("—")
    if dash_count:
        issues.append(
            StyleIssue(
                code="em_dash",
                message="Prefer commas, parentheses or sentence breaks over em dashes.",
                count=dash_count,
            )
        )

    lowered = text.lower()
    zombie_count = sum(lowered.count(term) for term in ZOMBIE_NOUNS)
    if zombie_count:
        issues.append(
            StyleIssue(
                code="zombie_noun",
                message="Replace abstract noun phrases with direct verbs or concrete nouns.",
                count=zombie_count,
            )
        )

    return issues
```

- [ ] **Step 4: Export style guard**

Modify `code/src/gmnps/manuscript/__init__.py` so it is exactly:

```python
"""Manuscript-ready exports for the GMNPS Nature Food article framework."""

from gmnps.manuscript.export import (
    export_article_bundle,
    latex_table_from_frame,
    manuscript_table_summaries,
)
from gmnps.manuscript.style_guard import StyleIssue, check_academic_style

__all__ = [
    "StyleIssue",
    "check_academic_style",
    "export_article_bundle",
    "latex_table_from_frame",
    "manuscript_table_summaries",
]
```

- [ ] **Step 5: Run style guard tests**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/code/src"
/Users/fengxi.25/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from tests import test_manuscript_style_guard as t
for name in sorted(dir(t)):
    if name.startswith("test_"):
        print("RUN", name)
        getattr(t, name)()
print("STYLE GUARD TESTS OK")
PY
```

Expected: PASS with `STYLE GUARD TESTS OK`.

- [ ] **Step 6: Commit style guard**

Run:

```bash
git add code/src/gmnps/manuscript/style_guard.py code/src/gmnps/manuscript/__init__.py code/src/tests/test_manuscript_style_guard.py
git commit -m "feat: add manuscript style guard"
```

Expected: commit succeeds.

---

### Task 5: Run Full Verification and Push

**Files:**
- Verify: all files modified in Tasks 1 to 4.

**Interfaces:**
- Consumes: tests from `test_article_scoring.py`, `test_title_narrative.py`, and `test_manuscript_style_guard.py`.
- Produces: verified local branch pushed to `origin/main`.

- [ ] **Step 1: Run compile checks**

Run:

```bash
cd "$(git rev-parse --show-toplevel)"
/Users/fengxi.25/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m py_compile \
  code/src/gmnps/scoring/masks.py \
  code/src/gmnps/scoring/anchored.py \
  code/src/gmnps/validation/preservation.py \
  code/src/gmnps/validation/synthetic_twin.py \
  code/src/gmnps/manuscript/export.py \
  code/src/gmnps/manuscript/style_guard.py \
  code/src/scripts/run_nature_food_article.py \
  code/src/tests/test_article_scoring.py \
  code/src/tests/test_title_narrative.py \
  code/src/tests/test_manuscript_style_guard.py
```

Expected: exit code 0.

- [ ] **Step 2: Run all direct tests**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/code/src"
/Users/fengxi.25/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from tests import test_article_scoring, test_title_narrative, test_manuscript_style_guard

modules = [test_article_scoring, test_title_narrative, test_manuscript_style_guard]
for module in modules:
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        fn = getattr(module, name)
        print("RUN", module.__name__, name)
        if name == "test_article_bundle_export":
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
print("ALL DIRECT TESTS OK")
PY
```

Expected: PASS with `ALL DIRECT TESTS OK`.

- [ ] **Step 3: Run synthetic CLI smoke test**

Run:

```bash
cd "$(git rev-parse --show-toplevel)"
OUT=/tmp/gmnps_title_narrative_smoke_$$
/Users/fengxi.25/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 code/src/scripts/run_nature_food_article.py synthetic \
  --output-dir "$OUT" \
  --n-individuals 30 \
  --n-foods 18 \
  --seed 11
test -f "$OUT/figure_source_data/synthetic_benchmark.csv"
test -f "$OUT/latex/synthetic_benchmark.tex"
cat "$OUT/tables/nps_preservation_metrics.json"
```

Expected: exit code 0 and JSON containing `"passes_min_rho": true`.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: only `.joycode/` may appear as untracked. No modified tracked files remain.

- [ ] **Step 5: Push**

Run:

```bash
git push origin main
```

Expected: remote `main` updates successfully.
