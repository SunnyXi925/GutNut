"""Testing-only synthetic checks for leakage-safe cohort splitting.

The fixtures in this module are generated in memory and contain no observed,
controlled, aggregate, or repository outcome values.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gmnps.validation.cohort_split import make_nested_group_splits


def _testing_only_participant_meals() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant_number in range(1, 17):
        participant_id = f"testing-p{participant_number:02d}"
        pair_number = (participant_number - 1) // 2
        family_id = f"testing-family-{pair_number:02d}"
        twin_id = (
            f"testing-twin-{pair_number:02d}"
            if participant_number <= 4
            else None
        )
        cohort_id = f"testing-cohort-{pair_number % 4}"
        for meal_number in range(4):
            rows.append(
                {
                    "participant_id": participant_id,
                    "family_id": family_id,
                    "twin_id": twin_id,
                    "meal_id": f"testing-meal-{meal_number}",
                    "food_id": f"testing-food-{meal_number % 3}",
                    "cohort_id": cohort_id,
                }
            )
    return pd.DataFrame(rows)


def _values(frame: pd.DataFrame, positions: tuple[int, ...], column: str) -> set[str]:
    return set(frame.iloc[list(positions)][column].dropna().astype(str))


def test_outer_split_is_subject_held_out_and_keeps_family_twin_components_intact():
    frame = _testing_only_participant_meals()
    plan = make_nested_group_splits(
        frame,
        outer_folds=4,
        inner_folds=3,
        outer_seed=1729,
        inner_seed=2718,
    )

    assert len(plan) == 4
    all_test_rows: set[int] = set()
    for nested in plan:
        train = set(nested.outer.train_positions)
        test = set(nested.outer.test_positions)
        assert train
        assert test
        assert train.isdisjoint(test)
        assert _values(frame, nested.outer.train_positions, "participant_id").isdisjoint(
            _values(frame, nested.outer.test_positions, "participant_id")
        )
        assert _values(frame, nested.outer.train_positions, "family_id").isdisjoint(
            _values(frame, nested.outer.test_positions, "family_id")
        )
        assert _values(frame, nested.outer.train_positions, "twin_id").isdisjoint(
            _values(frame, nested.outer.test_positions, "twin_id")
        )
        all_test_rows.update(test)

    assert all_test_rows == set(range(len(frame)))


def test_nested_inner_splits_are_fit_isolated_inside_outer_development_rows():
    frame = _testing_only_participant_meals()
    plan = make_nested_group_splits(
        frame,
        outer_folds=4,
        inner_folds=3,
        outer_seed=1729,
        inner_seed=2718,
    )

    for nested in plan:
        outer_train = set(nested.outer.train_positions)
        outer_test = set(nested.outer.test_positions)
        inner_validation_union: set[int] = set()
        assert len(nested.inner) == 3
        for inner in nested.inner:
            inner_train = set(inner.train_positions)
            inner_validation = set(inner.test_positions)
            assert inner_train
            assert inner_validation
            assert inner_train.isdisjoint(inner_validation)
            assert (inner_train | inner_validation).issubset(outer_train)
            assert (inner_train | inner_validation).isdisjoint(outer_test)
            assert _values(frame, inner.train_positions, "participant_id").isdisjoint(
                _values(frame, inner.test_positions, "participant_id")
            )
            assert _values(frame, inner.train_positions, "family_id").isdisjoint(
                _values(frame, inner.test_positions, "family_id")
            )
            assert _values(frame, inner.train_positions, "twin_id").isdisjoint(
                _values(frame, inner.test_positions, "twin_id")
            )
            inner_validation_union.update(inner_validation)
        assert inner_validation_union == outer_train


@pytest.mark.parametrize("secondary_unit", ["meal_id", "food_id", "cohort_id"])
def test_optional_secondary_holdout_keeps_subjects_and_requested_unit_disjoint(
    secondary_unit: str,
):
    frame = _testing_only_participant_meals()
    plan = make_nested_group_splits(
        frame,
        outer_folds=3,
        inner_folds=2,
        outer_seed=1729,
        inner_seed=2718,
        secondary_holdout=secondary_unit,
    )

    for nested in plan:
        assert _values(
            frame, nested.outer.train_positions, "participant_id"
        ).isdisjoint(_values(frame, nested.outer.test_positions, "participant_id"))
        assert _values(
            frame, nested.outer.train_positions, secondary_unit
        ).isdisjoint(_values(frame, nested.outer.test_positions, secondary_unit))
        assert set(nested.outer.dropped_positions).isdisjoint(
            set(nested.outer.train_positions) | set(nested.outer.test_positions)
        )
        for inner in nested.inner:
            assert _values(
                frame, inner.train_positions, secondary_unit
            ).isdisjoint(_values(frame, inner.test_positions, secondary_unit))


def test_cohort_holdout_uses_whole_cohorts_and_connects_cross_cohort_relatives():
    frame = _testing_only_participant_meals()
    frame.loc[frame["participant_id"].eq("testing-p01"), "cohort_id"] = "testing-cross-a"
    frame.loc[frame["participant_id"].eq("testing-p02"), "cohort_id"] = "testing-cross-b"
    plan = make_nested_group_splits(
        frame,
        outer_folds=3,
        inner_folds=2,
        outer_seed=1729,
        inner_seed=2718,
        secondary_holdout="cohort_id",
    )

    all_cohorts = set(frame["cohort_id"].astype(str))
    for nested in plan:
        train_cohorts = _values(frame, nested.outer.train_positions, "cohort_id")
        test_cohorts = _values(frame, nested.outer.test_positions, "cohort_id")
        assert train_cohorts.isdisjoint(test_cohorts)
        assert train_cohorts | test_cohorts == all_cohorts
        assert not nested.outer.dropped_positions
        linked = {"testing-cross-a", "testing-cross-b"}
        assert linked.issubset(train_cohorts) or linked.issubset(test_cohorts)


def test_family_and_twin_links_are_closed_transitively():
    frame = _testing_only_participant_meals()
    frame.loc[frame["participant_id"].eq("testing-p02"), "twin_id"] = "testing-bridge"
    frame.loc[frame["participant_id"].eq("testing-p03"), "twin_id"] = "testing-bridge"
    plan = make_nested_group_splits(
        frame,
        outer_folds=3,
        inner_folds=2,
        outer_seed=1729,
        inner_seed=2718,
    )
    linked = {"testing-p01", "testing-p02", "testing-p03", "testing-p04"}
    for nested in plan:
        test = _values(frame, nested.outer.test_positions, "participant_id")
        assert linked.isdisjoint(test) or linked.issubset(test)


def test_split_plan_is_deterministic_and_rejects_inconsistent_participant_metadata():
    frame = _testing_only_participant_meals()
    first = make_nested_group_splits(
        frame,
        outer_folds=4,
        inner_folds=3,
        outer_seed=1729,
        inner_seed=2718,
    )
    second = make_nested_group_splits(
        frame.sample(frac=1.0, random_state=99).reset_index(drop=True),
        outer_folds=4,
        inner_folds=3,
        outer_seed=1729,
        inner_seed=2718,
    )

    first_test_participants = [
        _values(frame, fold.outer.test_positions, "participant_id") for fold in first
    ]
    shuffled = frame.sample(frac=1.0, random_state=99).reset_index(drop=True)
    second_test_participants = [
        _values(shuffled, fold.outer.test_positions, "participant_id") for fold in second
    ]
    assert first_test_participants == second_test_participants

    inconsistent = frame.copy()
    participant_rows = inconsistent["participant_id"].eq("testing-p01")
    inconsistent.loc[participant_rows.idxmax(), "family_id"] = "testing-family-conflict"
    with pytest.raises(ValueError, match="participant.*family_id|family_id.*participant"):
        make_nested_group_splits(
            inconsistent,
            outer_folds=4,
            inner_folds=3,
            outer_seed=1729,
            inner_seed=2718,
        )
