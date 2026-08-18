"""Build Food Compass 2.0 subgroup metadata from FNDDS/WWEIA categories."""
from __future__ import annotations

import re

import pandas as pd


FCS2_SUBGROUP_MAPPING_VERSION = "fcs2_fndds_subgroup_v1"

_MISSING_LABELS = {"", "nan", "none", "null"}
_HEALTHFUL_RE = re.compile(
    r"vegetable|fruit|apple|berry|berries|citrus|legume|bean|lentil|pea|"
    r"whole\s*grain|seafood|fish|nut|seed",
    flags=re.IGNORECASE,
)
_LIMIT_RE = re.compile(
    r"sweet|sugar|candy|dessert|snack|fat|oil|processed",
    flags=re.IGNORECASE,
)


def _foodcode_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _clean_label(series: pd.Series, column: str) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    missing = cleaned.isna() | cleaned.str.casefold().isin(_MISSING_LABELS)
    if missing.any():
        examples = series.loc[missing].head(3).tolist()
        raise ValueError(f"missing {column} labels: {examples}")
    return cleaned.astype(object)


def _direction(label: object) -> str:
    text = str(label).strip()
    if _HEALTHFUL_RE.search(text):
        return "stable_or_up"
    if _LIMIT_RE.search(text):
        return "stable_or_down"
    return "stable"


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _canonical_fndds_categories(fndds_categories: pd.DataFrame) -> pd.DataFrame:
    return fndds_categories.rename(
        columns={
            "Food code": "foodcode",
            "WWEIA Category description": "wweia_category",
            "WWEIA category description": "wweia_category",
            "WWEIA_Category_description": "wweia_category",
        }
    )


def build_fcs2_subgroup_metadata(
    food_summary: pd.DataFrame,
    fndds_categories: pd.DataFrame,
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return audited FCS2 group/subgroup metadata keyed by foodcode.

    The output preserves the row order and `foodcode` values from
    `food_summary`. FNDDS/WWEIA and override food codes are matched using a
    stripped string key so CSV- and spreadsheet-derived inputs can be mixed.
    """

    fndds_categories = _canonical_fndds_categories(fndds_categories)
    _require_columns(food_summary, {"foodcode", "food_group"}, "food_summary")
    _require_columns(fndds_categories, {"foodcode", "wweia_category"}, "fndds_categories")

    base = food_summary.loc[:, ["foodcode", "food_group"]].copy()
    base["_foodcode_key"] = _foodcode_key(base["foodcode"])

    fndds = fndds_categories.loc[:, ["foodcode", "wweia_category"]].copy()
    fndds["_foodcode_key"] = _foodcode_key(fndds["foodcode"])
    fndds = fndds.drop_duplicates("_foodcode_key", keep="first")

    merged = base.merge(
        fndds.loc[:, ["_foodcode_key", "wweia_category"]],
        on="_foodcode_key",
        how="left",
        validate="many_to_one",
    )
    merged["food_subgroup"] = merged["wweia_category"]
    merged["mapping_source"] = "fndds_wweia_category"

    if overrides is not None and not overrides.empty:
        _require_columns(overrides, {"foodcode", "food_subgroup"}, "overrides")
        override_frame = overrides.loc[:, ["foodcode", "food_subgroup"]].copy()
        override_frame["_foodcode_key"] = _foodcode_key(override_frame["foodcode"])
        override_frame = override_frame.drop_duplicates("_foodcode_key", keep="last")
        override_map = override_frame.set_index("_foodcode_key")["food_subgroup"]
        target = merged["_foodcode_key"].isin(override_map.index)
        merged.loc[target, "food_subgroup"] = merged.loc[target, "_foodcode_key"].map(override_map)
        merged.loc[target, "mapping_source"] = "curated_override"

    merged["food_group"] = _clean_label(merged["food_group"], "food_group")
    merged["food_subgroup"] = _clean_label(merged["food_subgroup"], "food_subgroup")
    merged["expected_group_direction"] = merged["food_group"].map(_direction)
    merged["expected_subgroup_direction"] = merged["food_subgroup"].map(_direction)
    merged["mapping_version"] = FCS2_SUBGROUP_MAPPING_VERSION

    return merged.loc[
        :,
        [
            "foodcode",
            "food_group",
            "food_subgroup",
            "expected_group_direction",
            "expected_subgroup_direction",
            "mapping_source",
            "mapping_version",
        ],
    ]
