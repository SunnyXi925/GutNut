import pandas as pd
import pytest

from gmnps.data_sources.fcs2_subgroups import (
    FCS2_SUBGROUP_MAPPING_VERSION,
    build_fcs2_subgroup_metadata,
)


def test_fcs2_subgroup_metadata_covers_foodcodes_and_expected_directions():
    food = pd.DataFrame(
        {
            "foodcode": [11111000, 91705010, 44100300],
            "description": ["Spinach raw", "Apple raw", "Chocolate candy"],
            "food_group": ["Vegetables", "Fruits", "Sweets"],
            "FCS_2_0": [92.0, 88.0, 18.0],
        }
    )
    fndds = pd.DataFrame(
        {
            "foodcode": [11111000, 91705010, 44100300],
            "wweia_category": ["Dark green vegetables", "Apples", "Candy containing chocolate"],
        }
    )

    mapped = build_fcs2_subgroup_metadata(food, fndds)

    assert mapped["foodcode"].tolist() == [11111000, 91705010, 44100300]
    assert mapped["food_subgroup"].tolist() == [
        "Dark green vegetables",
        "Apples",
        "Candy containing chocolate",
    ]
    assert mapped["expected_group_direction"].tolist() == [
        "stable_or_up",
        "stable_or_up",
        "stable_or_down",
    ]
    assert mapped["expected_subgroup_direction"].tolist() == [
        "stable_or_up",
        "stable_or_up",
        "stable_or_down",
    ]
    assert mapped["mapping_source"].eq("fndds_wweia_category").all()
    assert mapped["mapping_version"].eq(FCS2_SUBGROUP_MAPPING_VERSION).all()


@pytest.mark.parametrize("bad_value", ["", "nan", "None", None])
def test_fcs2_subgroup_metadata_rejects_missing_subgroups(bad_value):
    food = pd.DataFrame({"foodcode": [1], "food_group": ["Vegetables"], "FCS_2_0": [90.0]})
    fndds = pd.DataFrame({"foodcode": [1], "wweia_category": [bad_value]})

    with pytest.raises(ValueError, match="missing food_subgroup"):
        build_fcs2_subgroup_metadata(food, fndds)


def test_fcs2_subgroup_metadata_applies_overrides_and_preserves_food_order():
    food = pd.DataFrame(
        {
            "foodcode": [1001, 1002],
            "food_group": ["Mixed dishes", "Fats and oils"],
        }
    )
    fndds = pd.DataFrame(
        {
            "foodcode": ["1002", "1001"],
            "wweia_category": ["Oils", "Other mixed dishes"],
        }
    )
    overrides = pd.DataFrame({"foodcode": ["1001"], "food_subgroup": ["Beans and lentils"]})

    mapped = build_fcs2_subgroup_metadata(food, fndds, overrides=overrides)

    assert mapped["foodcode"].tolist() == [1001, 1002]
    assert mapped["food_subgroup"].tolist() == ["Beans and lentils", "Oils"]
    assert mapped["expected_subgroup_direction"].tolist() == ["stable_or_up", "stable_or_down"]
    assert mapped["mapping_source"].tolist() == ["curated_override", "fndds_wweia_category"]


def test_fcs2_subgroup_metadata_accepts_processed_fndds_column_names():
    food = pd.DataFrame({"foodcode": ["11111000"], "food_group": ["Vegetables"]})
    fndds = pd.DataFrame(
        {
            "Food code": [11111000],
            "WWEIA Category description": ["Dark green vegetables"],
        }
    )

    mapped = build_fcs2_subgroup_metadata(food, fndds)

    assert mapped.loc[0, "food_subgroup"] == "Dark green vegetables"
    assert mapped.loc[0, "expected_subgroup_direction"] == "stable_or_up"


def test_fcs2_subgroup_metadata_rejects_missing_food_groups():
    food = pd.DataFrame({"foodcode": [1], "food_group": [" null "]})
    fndds = pd.DataFrame({"foodcode": [1], "wweia_category": ["Vegetables"]})

    with pytest.raises(ValueError, match="missing food_group"):
        build_fcs2_subgroup_metadata(food, fndds)
