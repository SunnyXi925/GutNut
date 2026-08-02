# FNDDS Version Strategy For GMNPS

GMNPS must match Food Compass 2.0 Table S5 to the same historical survey-food
universe used by the baseline score. The Food Compass 2.0 supplement describes
9,273 foods and beverages consumed by US adults in NHANES 2001-2002 through
2017-2018 and contained in FNDDS 2001-2002 through 2017-2018.

## Primary Rule

Use cycle-specific USDA FNDDS releases from 2001-2002 to 2017-2018 when
constructing nutrient vectors for Food Compass 2.0 anchored foods.

Do not replace these historical releases with current FoodData Central Survey
Foods, Foundation Foods, Branded Foods or full FoodData Central current-release
tables for primary baseline matching.

## Download Sources

- FoodData Central resources page: `https://fdc.nal.usda.gov/resources`
- FoodData Central download page: `https://fdc.nal.usda.gov/download-datasets`
- USDA ARS Food Surveys Research Group FNDDS downloads:
  `https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds-download-databases/`

Provenance note, accessed 2 August 2026: the FoodData Central download page
exposed recent historical Survey Foods CSV releases, including 2013-2014,
2015-2016 and 2017-2018. Earlier FNDDS cycles should be obtained from the USDA
ARS FNDDS download page.

## Required Historical Cycles

- FNDDS 2001-2002
- FNDDS 2003-2004
- FNDDS 2005-2006
- FNDDS 2007-2008
- FNDDS 2009-2010
- FNDDS 2011-2012
- FNDDS 2013-2014
- FNDDS 2015-2016
- FNDDS 2017-2018

FPED or food-pattern variables should follow the same NHANES/FNDDS cycle range
if they are used in sensitivity analyses.

## Join Policy

- Preserve `Foodcode`, `Description` and `release_year` or `fndds_cycle`.
- Treat `Foodcode` as a cycle-specific identifier, not a globally unique key.
- Use `Foodcode + normalized Description + fndds_cycle` as the primary
  reproducible join key when historical cycle information is available.
- If FCS2 Table S5 lacks an explicit cycle for a duplicated `Foodcode`, keep
  duplicate rows as distinct food records until an official spreadsheet or
  manual provenance review resolves the mapping.
- Use exact food-code joins where possible.
- Avoid fuzzy description joins in primary analyses.
- Record food-code duplicates and discontinued-code mappings before merging
  nutrient values with Food Compass 2.0 scores.
- Convert nutrients to the scoring basis required by each analysis, recording
  whether values are per 100 g, per portion or per 100 kcal.

## Methods Sentence

We anchored Food Compass 2.0 scores to the 9,273 foods and beverages reported in
NHANES 2001-2002 through 2017-2018 and matched nutrient vectors using
cycle-specific USDA FNDDS releases from 2001-2002 to 2017-2018, preserving FNDDS
food codes, descriptions and release years to reduce cross-cycle food-code drift.
