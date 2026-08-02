from gmnps.data_sources.fcs2_pdf import filter_primary_fcs2_foods, parse_fcs2_table_s5_text


def test_parse_table_s5_handles_wrapped_descriptions_and_spaced_groups():
    text = """Foodcode          Description                                                       Food group        FCS 2.0        FCS 1.0        Difference         NOVAa       HSR         Nutri-Score
                  Bread, whole wheat, 100%, made from home recipe or purchased at
       51201060                                                                     1000_Grains                 75             73                2             1         4                   C
                  bakery
82105750   Canola and soybean oil                                                    7000_Fats Oils        78    85   -7    2   3.5   B
a
    non-integer NOVA scores for mixed dishes have been rounded to the nearest whole number
"""

    df, audit = parse_fcs2_table_s5_text(text)

    assert len(df) == 2
    assert df.loc[0, "Foodcode"] == "51201060"
    assert "Bread, whole wheat" in df.loc[0, "Description"]
    assert df.loc[1, "Food_group"] == "7000_Fats Oils"
    assert df.loc[1, "FCS_2_0"] == 78
    assert audit.score_range_ok
    assert audit.fcs_difference_ok
    assert audit.unresolved_line_count == 0
    assert audit.malformed_foodcode_line_count == 0


def test_parse_table_s5_reports_duplicates_and_missing_nutri_score():
    text = """Foodcode          Description                                                       Food group        FCS 2.0        FCS 1.0        Difference         NOVAa       HSR         Nutri-Score
53720510   Snickers Marathon Energy bar                                          1000_Grains           55   66   -11   4   2     D
53720510   Snickers Marathon Energy bar                                          1000_Grains           53   67   -14   4   2
HSR Health Star Rating; FCS Food Compass Score; NS Not specified; NFS Not further specified
"""

    df, audit = parse_fcs2_table_s5_text(text)

    assert len(df) == 2
    assert audit.duplicate_foodcodes == ["53720510"]
    assert audit.missing_nutri_score_count == 1
    assert audit.score_range_ok
    assert audit.fcs_difference_ok
    assert not audit.join_ready
    assert df.loc[1, "Nutri_Score"] == ""


def test_parse_table_s5_reports_malformed_foodcode_lines():
    text = """Foodcode          Description                                                       Food group        FCS 2.0        FCS 1.0        Difference         NOVAa       HSR         Nutri-Score
56204005   Quinoa, no added fat                                              1000_Grains                 89             88                1             1         4                   A
56204006   Broken row without enough score fields
HSR Health Star Rating; FCS Food Compass Score; NS Not specified; NFS Not further specified
"""

    df, audit = parse_fcs2_table_s5_text(text)

    assert len(df) == 1
    assert audit.malformed_foodcode_line_count == 1
    assert "56204006" in audit.malformed_foodcode_line_examples[0]
    assert not audit.join_ready


def test_filter_primary_fcs2_foods_excludes_duplicate_and_missing_labels():
    text = """Foodcode          Description                                                       Food group        FCS 2.0        FCS 1.0        Difference         NOVAa       HSR         Nutri-Score
56204005   Quinoa, no added fat                                              1000_Grains                 89             88                1             1         4                   A
53720510   Snickers Marathon Energy bar                                      1000_Grains                 55             66               -11            4         2                   D
53720510   Snickers Marathon Energy bar                                      1000_Grains                 53             67               -14            4         2                   D
12345678   Missing label food                                                1000_Grains                 44             44                0             1         3
HSR Health Star Rating; FCS Food Compass Score; NS Not specified; NFS Not further specified
"""
    df, _ = parse_fcs2_table_s5_text(text)

    primary, excluded, summary = filter_primary_fcs2_foods(df)

    assert primary["Foodcode"].tolist() == ["56204005"]
    assert set(excluded["Foodcode"]) == {"53720510", "12345678"}
    assert summary["input_rows"] == 4
    assert summary["primary_rows"] == 1
    assert summary["excluded_rows"] == 3
