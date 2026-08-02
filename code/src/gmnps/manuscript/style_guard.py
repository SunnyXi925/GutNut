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
