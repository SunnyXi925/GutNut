"""Deterministic family/twin-aware splits for person-by-meal validation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Mapping

import numpy as np
import pandas as pd


_PARTICIPANT_COLUMN = "participant_id"
_GROUPING_COLUMNS = ("family_id", "twin_id")
_ALLOWED_SECONDARY_HOLDOUTS = frozenset({"meal_id", "food_id", "cohort_id"})


@dataclass(frozen=True)
class GroupSplit:
    """One train/test split expressed as immutable positional row indices."""

    fold_id: int
    train_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    dropped_positions: tuple[int, ...] = ()
    held_out_group_ids: tuple[str, ...] = ()
    held_out_secondary_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class NestedGroupSplit:
    """An outer held-out split and its development-only inner splits."""

    outer: GroupSplit
    inner: tuple[GroupSplit, ...]


class _UnionFind:
    def __init__(self, values: tuple[str, ...]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _nonempty_string(value: object, label: str) -> str:
    if pd.isna(value):
        raise ValueError(f"{label} contains missing values")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} contains empty values")
    return normalized


def _participant_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    required = {_PARTICIPANT_COLUMN, *_GROUPING_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"split data missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("split data must contain rows")

    working = frame.loc[:, [_PARTICIPANT_COLUMN, *_GROUPING_COLUMNS]].copy()
    working[_PARTICIPANT_COLUMN] = working[_PARTICIPANT_COLUMN].map(
        lambda value: _nonempty_string(value, _PARTICIPANT_COLUMN)
    )
    for column in _GROUPING_COLUMNS:
        for participant_id, values in working.groupby(_PARTICIPANT_COLUMN)[column]:
            observed = {
                str(value).strip()
                for value in values
                if not pd.isna(value) and str(value).strip()
            }
            if len(observed) > 1:
                raise ValueError(
                    f"participant {participant_id} has inconsistent {column} metadata"
                )

    records = []
    for participant_id, participant_rows in working.groupby(
        _PARTICIPANT_COLUMN, sort=True
    ):
        record: dict[str, object] = {_PARTICIPANT_COLUMN: participant_id}
        for column in _GROUPING_COLUMNS:
            observed = sorted(
                {
                    str(value).strip()
                    for value in participant_rows[column]
                    if not pd.isna(value) and str(value).strip()
                }
            )
            record[column] = observed[0] if observed else None
        records.append(record)
    return pd.DataFrame.from_records(records).set_index(_PARTICIPANT_COLUMN)


def _component_by_participant(frame: pd.DataFrame) -> dict[str, str]:
    metadata = _participant_metadata(frame)
    participants = tuple(sorted(metadata.index.astype(str)))
    union_find = _UnionFind(participants)
    for column in _GROUPING_COLUMNS:
        grouped: dict[str, list[str]] = {}
        for participant_id, value in metadata[column].items():
            if value is not None:
                grouped.setdefault(str(value), []).append(str(participant_id))
        for members in grouped.values():
            anchor = min(members)
            for member in members:
                union_find.union(anchor, member)

    components: dict[str, list[str]] = {}
    for participant_id in participants:
        components.setdefault(union_find.find(participant_id), []).append(participant_id)
    canonical = {
        participant_id: min(members)
        for members in components.values()
        for participant_id in members
    }
    return canonical


def _balanced_assignment(
    weights: Mapping[str, int],
    *,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("number of folds must be an integer of at least two")
    if len(weights) < n_folds:
        raise ValueError(
            f"number of independent groups ({len(weights)}) is smaller than folds ({n_folds})"
        )
    rng = np.random.default_rng(seed)
    keys = sorted(weights)
    randomized_rank = {
        key: rank for rank, key in enumerate(np.asarray(keys)[rng.permutation(len(keys))])
    }
    ordered = sorted(keys, key=lambda key: (-int(weights[key]), randomized_rank[key], key))
    fold_priority = {
        int(fold): rank
        for rank, fold in enumerate(rng.permutation(n_folds).tolist())
    }
    fold_sizes = [0] * n_folds
    assignment: dict[str, int] = {}
    for key in ordered:
        fold = min(range(n_folds), key=lambda value: (fold_sizes[value], fold_priority[value]))
        assignment[key] = fold
        fold_sizes[fold] += int(weights[key])
    return assignment


def _split_whole_cohorts(
    frame: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
) -> tuple[GroupSplit, ...]:
    """Hold out complete cohort components linked by family/twin relationships."""

    if "cohort_id" not in frame.columns:
        raise ValueError("split data missing secondary column cohort_id")
    participant_components = _component_by_participant(frame)
    participant_values = frame[_PARTICIPANT_COLUMN].map(
        lambda value: _nonempty_string(value, _PARTICIPANT_COLUMN)
    )
    cohort_values = frame["cohort_id"].map(
        lambda value: _nonempty_string(value, "cohort_id")
    )
    participant_cohorts = pd.DataFrame(
        {"participant_id": participant_values, "cohort_id": cohort_values}
    ).drop_duplicates()
    inconsistent = participant_cohorts["participant_id"].duplicated(keep=False)
    if inconsistent.any():
        raise ValueError("participant has inconsistent cohort_id metadata")

    cohorts = tuple(sorted(cohort_values.unique()))
    cohort_union = _UnionFind(cohorts)
    component_to_cohorts: dict[str, list[str]] = {}
    for row in participant_cohorts.itertuples(index=False):
        component_to_cohorts.setdefault(
            participant_components[str(row.participant_id)], []
        ).append(str(row.cohort_id))
    for linked_cohorts in component_to_cohorts.values():
        anchor = min(linked_cohorts)
        for cohort_id in linked_cohorts:
            cohort_union.union(anchor, cohort_id)
    cohort_members: dict[str, list[str]] = {}
    for cohort_id in cohorts:
        cohort_members.setdefault(cohort_union.find(cohort_id), []).append(cohort_id)
    cohort_component = {
        cohort_id: min(members)
        for members in cohort_members.values()
        for cohort_id in members
    }
    row_components = cohort_values.map(cohort_component)
    component_weights = row_components.value_counts().astype(int).to_dict()
    component_folds = _balanced_assignment(
        component_weights,
        n_folds=n_folds,
        seed=seed,
    )
    row_folds = row_components.map(component_folds).to_numpy(dtype=int)
    global_positions = frame["__gmnps_global_position__"].to_numpy(dtype=int)
    participant_component_values = participant_values.map(participant_components)
    splits = []
    for fold_id in range(n_folds):
        test_mask = row_folds == fold_id
        train_mask = ~test_mask
        splits.append(
            GroupSplit(
                fold_id=fold_id,
                train_positions=tuple(sorted(global_positions[train_mask].tolist())),
                test_positions=tuple(sorted(global_positions[test_mask].tolist())),
                dropped_positions=(),
                held_out_group_ids=tuple(
                    sorted(participant_component_values[test_mask].unique())
                ),
                held_out_secondary_ids=tuple(
                    sorted(cohort_values[test_mask].unique())
                ),
            )
        )
    return tuple(splits)


def _split_frame(
    frame: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
    secondary_holdout: str | None,
) -> tuple[GroupSplit, ...]:
    if secondary_holdout == "cohort_id":
        return _split_whole_cohorts(frame, n_folds=n_folds, seed=seed)
    components = _component_by_participant(frame)
    participant_values = frame[_PARTICIPANT_COLUMN].map(
        lambda value: _nonempty_string(value, _PARTICIPANT_COLUMN)
    )
    component_values = participant_values.map(components)
    component_weights = component_values.value_counts().astype(int).to_dict()
    component_folds = _balanced_assignment(
        component_weights,
        n_folds=n_folds,
        seed=seed,
    )
    row_group_folds = component_values.map(component_folds).to_numpy(dtype=int)

    secondary_values: pd.Series | None = None
    row_secondary_folds: np.ndarray | None = None
    secondary_folds: dict[str, int] = {}
    if secondary_holdout is not None:
        if secondary_holdout not in _ALLOWED_SECONDARY_HOLDOUTS:
            raise ValueError(
                "secondary_holdout must be one of meal_id, food_id, cohort_id"
            )
        if secondary_holdout not in frame.columns:
            raise ValueError(f"split data missing secondary column {secondary_holdout}")
        secondary_values = frame[secondary_holdout].map(
            lambda value: _nonempty_string(value, secondary_holdout)
        )
        secondary_weights = secondary_values.value_counts().astype(int).to_dict()
        secondary_folds = _balanced_assignment(
            secondary_weights,
            n_folds=n_folds,
            seed=seed + 104_729,
        )
        row_secondary_folds = secondary_values.map(secondary_folds).to_numpy(dtype=int)
        best_permutation: tuple[int, ...] | None = None
        best_score: tuple[int, int, int] | None = None
        for permutation in permutations(range(n_folds)):
            aligned = np.asarray(permutation, dtype=int)[row_secondary_folds]
            test_sizes = [
                int(np.sum((row_group_folds == fold) & (aligned == fold)))
                for fold in range(n_folds)
            ]
            train_sizes = [
                int(np.sum((row_group_folds != fold) & (aligned != fold)))
                for fold in range(n_folds)
            ]
            score = (min(test_sizes), min(train_sizes), sum(test_sizes))
            if best_score is None or score > best_score:
                best_score = score
                best_permutation = tuple(int(value) for value in permutation)
        assert best_permutation is not None
        row_secondary_folds = np.asarray(best_permutation, dtype=int)[
            row_secondary_folds
        ]
        secondary_folds = {
            key: best_permutation[value] for key, value in secondary_folds.items()
        }

    global_positions = frame["__gmnps_global_position__"].to_numpy(dtype=int)
    splits = []
    for fold_id in range(n_folds):
        group_test = row_group_folds == fold_id
        if row_secondary_folds is None:
            test_mask = group_test
            train_mask = ~group_test
        else:
            secondary_test = row_secondary_folds == fold_id
            test_mask = group_test & secondary_test
            train_mask = (~group_test) & (~secondary_test)
        dropped_mask = ~(train_mask | test_mask)
        train_positions = tuple(sorted(global_positions[train_mask].tolist()))
        test_positions = tuple(sorted(global_positions[test_mask].tolist()))
        dropped_positions = tuple(sorted(global_positions[dropped_mask].tolist()))
        if not train_positions or not test_positions:
            raise ValueError(
                f"fold {fold_id} is empty after applying grouped holdout constraints"
            )
        held_out_groups = tuple(
            sorted(key for key, value in component_folds.items() if value == fold_id)
        )
        held_out_secondary = tuple(
            sorted(key for key, value in secondary_folds.items() if value == fold_id)
        )
        splits.append(
            GroupSplit(
                fold_id=fold_id,
                train_positions=train_positions,
                test_positions=test_positions,
                dropped_positions=dropped_positions,
                held_out_group_ids=held_out_groups,
                held_out_secondary_ids=held_out_secondary,
            )
        )
    return tuple(splits)


def make_nested_group_splits(
    frame: pd.DataFrame,
    *,
    outer_folds: int,
    inner_folds: int,
    outer_seed: int,
    inner_seed: int,
    secondary_holdout: str | None = None,
) -> tuple[NestedGroupSplit, ...]:
    """Create deterministic outer and development-only inner grouped splits.

    Family and twin identifiers are treated as graph edges between participants.
    When a secondary holdout is requested, crossed train/test rows are excluded so
    neither participants nor the secondary meal, food, or cohort unit can cross
    the evaluation boundary.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(outer_seed, int) or not isinstance(inner_seed, int):
        raise TypeError("split seeds must be integers")
    working = frame.copy()
    working["__gmnps_global_position__"] = np.arange(len(working), dtype=int)
    outer = _split_frame(
        working,
        n_folds=outer_folds,
        seed=outer_seed,
        secondary_holdout=secondary_holdout,
    )
    nested = []
    for outer_split in outer:
        development = working.iloc[list(outer_split.train_positions)].copy()
        inner = _split_frame(
            development,
            n_folds=inner_folds,
            seed=inner_seed + outer_split.fold_id,
            secondary_holdout=secondary_holdout,
        )
        nested.append(NestedGroupSplit(outer=outer_split, inner=inner))
    return tuple(nested)


__all__ = ["GroupSplit", "NestedGroupSplit", "make_nested_group_splits"]
