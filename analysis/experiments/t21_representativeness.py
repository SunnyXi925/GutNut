"""Who is in these cohorts, how much do they differ, and what is missing?

Three questions a reader needs before interpreting any personalisation result
are answered quantitatively rather than in prose:

1. what are the participant characteristics of each cohort, and how far are
   they from the adult populations the method would be applied to;
2. how much meal diversity, as opposed to participant diversity, does each
   cohort contain, since few-shot adaptation depends on exposure spread;
3. what is missing, and in which variables.

External reference values are entered as explicit, cited constants rather than
estimated here, and are labelled with their source so a reader can audit the
comparison instead of trusting it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env

# Comparison with a target population requires national survey values that the
# authors must supply and cite; none are hard-coded here, because an unsourced
# constant in a representativeness table is worse than an acknowledged gap. The
# file below is read if it exists and is otherwise reported as outstanding.
REFERENCE_FILE = Path(__file__).resolve().parents[1] / 'provenance' / 'target_population_reference.csv'
REFERENCE_COLUMNS = ['variable', 'population', 'value', 'kind', 'source']


def spread(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)),
                median=float(np.median(x)), q25=float(np.quantile(x, .25)),
                q75=float(np.quantile(x, .75)), min=float(x.min()), max=float(x.max()))


def person_table(frame, cohort, cols, rename):
    p = frame.drop_duplicates('participant_id')
    rows = []
    for c, label in zip(cols, rename):
        if c not in p:
            continue
        s = spread(p[c])
        s.update(cohort=cohort, variable=label,
                 missing_pct=float(100 * p[c].isna().mean()))
        rows.append(s)
    rows.append(dict(cohort=cohort, variable='female_pct',
                     n=int(p.person__female.notna().sum()),
                     mean=float(100 * p.person__female.mean()),
                     missing_pct=float(100 * p.person__female.isna().mean())))
    return rows


def exposure_table(frame, cohort, meal_key):
    g = frame.groupby('participant_id').size()
    rows = dict(cohort=cohort, participants=int(frame.participant_id.nunique()),
                grouping_units=int(frame.family_group.nunique()),
                meals=int(len(frame)),
                meals_per_participant_median=float(g.median()),
                meals_per_participant_min=int(g.min()),
                meals_per_participant_max=int(g.max()),
                distinct_meal_compositions=int(frame[meal_key].nunique()),
                distinct_compositions_per_participant_median=float(
                    frame.groupby('participant_id')[meal_key].nunique().median()))
    for c, label in (('food__fiber_g', 'fibre_g'), ('food__fat_g', 'total_fat_g')):
        if c in frame:
            rows[f'{label}_median'] = float(frame[c].median())
            rows[f'{label}_iqr'] = float(frame[c].quantile(.75) - frame[c].quantile(.25))
            within = frame.groupby('participant_id')[c].apply(
                lambda x: x.quantile(.75) - x.quantile(.25))
            rows[f'{label}_within_person_iqr_median'] = float(within.median())
            rows[f'{label}_within_over_between_iqr'] = float(
                within.median() / (frame[c].quantile(.75) - frame[c].quantile(.25)))
    rows['target_median'] = float(frame.target.median())
    rows['target_iqr'] = float(frame.target.quantile(.75) - frame.target.quantile(.25))
    rows['target_zero_pct'] = float(100 * (frame.target == 0).mean())
    return rows


if __name__ == '__main__':
    env.sha256_manifest([env.cohort_file('cgmacros_functional.parquet'),
                         env.cohort_file('predict.parquet')], 't21')
    cg = env.load_cgmacros()
    pr = env.load_predict()
    uk, us = pr[pr.country.eq('UK')], pr[pr.country.eq('US')]

    people = []
    people += person_table(cg, 'CGMacros', ['person__age', 'person__bmi', 'person__hba1c',
                                            'person__fasting_glucose'],
                           ['age_years', 'bmi_kg_m2', 'hba1c_pct', 'fasting_glucose_mg_dl'])
    for frame, name in ((uk, 'PREDICT_UK'), (us, 'PREDICT_US')):
        people += person_table(frame, name,
                               ['person__age', 'person__bmi_kg_m2', 'person__waist_hip_ratio',
                                'person__habitual_diet__fiber',
                                'person__habitual_diet__amed_score'],
                               ['age_years', 'bmi_kg_m2', 'waist_hip_ratio',
                                'habitual_fibre_g_day', 'mediterranean_diet_score'])
    PEOPLE = pd.DataFrame(people)
    PEOPLE.to_csv(env.RESULTS / 't21_participant_characteristics.csv', index=False)

    EXP = pd.DataFrame([exposure_table(cg, 'CGMacros', 'food_id'),
                        exposure_table(uk, 'PREDICT_UK', 'meal_id'),
                        exposure_table(us, 'PREDICT_US', 'meal_id')])
    EXP.to_csv(env.RESULTS / 't21_exposure_diversity.csv', index=False)

    # Between-cohort standardised differences: computable from the analysed data
    # and therefore auditable, unlike an unsourced national constant.
    comp = []
    for var in sorted(set(PEOPLE.variable) - {'female_pct'}):
        sub = PEOPLE[PEOPLE.variable.eq(var)].set_index('cohort')
        for a, b in [('CGMacros', 'PREDICT_UK'), ('CGMacros', 'PREDICT_US'),
                     ('PREDICT_UK', 'PREDICT_US')]:
            if a not in sub.index or b not in sub.index:
                continue
            ra, rb = sub.loc[a], sub.loc[b]
            pooled = np.sqrt((ra.sd ** 2 + rb.sd ** 2) / 2)
            comp.append(dict(variable=var, cohort_a=a, cohort_b=b,
                             mean_a=float(ra['mean']), mean_b=float(rb['mean']),
                             difference=float(ra['mean'] - rb['mean']),
                             standardised_difference=float((ra['mean'] - rb['mean']) / pooled)
                             if pooled > 0 else np.nan))
    for a, b in [('CGMacros', 'PREDICT_UK'), ('CGMacros', 'PREDICT_US'),
                 ('PREDICT_UK', 'PREDICT_US')]:
        sub = PEOPLE[PEOPLE.variable.eq('female_pct')].set_index('cohort')
        comp.append(dict(variable='female_pct', cohort_a=a, cohort_b=b,
                         mean_a=float(sub.loc[a]['mean']), mean_b=float(sub.loc[b]['mean']),
                         difference=float(sub.loc[a]['mean'] - sub.loc[b]['mean']),
                         standardised_difference=np.nan))
    COMP = pd.DataFrame(comp)
    COMP.to_csv(env.RESULTS / 't21_between_cohort_differences.csv', index=False)

    if REFERENCE_FILE.exists():
        REF = pd.read_csv(REFERENCE_FILE)
        assert list(REF.columns[:len(REFERENCE_COLUMNS)]) == REFERENCE_COLUMNS, REFERENCE_COLUMNS
        ref_status = f'supplied: {len(REF)} rows from {REFERENCE_FILE.name}'
    else:
        REF = pd.DataFrame(columns=REFERENCE_COLUMNS)
        ref_status = ('outstanding: no target-population reference values were supplied, so no '
                      'comparison with a national adult population is reported')
    REF.to_csv(env.RESULTS / 't21_target_population_reference.csv', index=False)

    strata = cg.drop_duplicates('participant_id').stratum.value_counts().to_dict()
    missing = {}
    for name, frame in (('CGMacros', cg), ('PREDICT_UK', uk), ('PREDICT_US', us)):
        p = frame.drop_duplicates('participant_id')
        cols = [c for c in p.columns if c.startswith('person__')]
        missing[name] = dict(person_variables=len(cols),
                             variables_with_any_missing=int(sum(p[c].isna().any() for c in cols)),
                             max_missing_pct=float(100 * max(p[c].isna().mean() for c in cols)),
                             meals_with_missing_target=int(frame.target.isna().sum()))

    env.dump(dict(
        cohorts=dict(CGMacros=dict(participants=int(cg.participant_id.nunique()),
                                   meals=int(len(cg)), strata=strata),
                     PREDICT_UK=dict(participants=int(uk.participant_id.nunique()),
                                     meals=int(len(uk)),
                                     meal_compositions=int(uk.meal_id.nunique())),
                     PREDICT_US=dict(participants=int(us.participant_id.nunique()),
                                     meals=int(len(us)),
                                     meal_compositions=int(us.meal_id.nunique()))),
        participant_characteristics=PEOPLE.round(4).to_dict(orient='records'),
        exposure_diversity=EXP.round(4).to_dict(orient='records'),
        between_cohort_differences=COMP.round(3).to_dict(orient='records'),
        target_population_comparison=ref_status,
        missing_data=missing,
        unavailable_in_the_analysed_extracts=[
            'self-reported race, ethnicity and geography',
            'medication use including glucose-lowering and lipid-lowering drugs',
            'sequencing batch identifiers and library preparation metadata',
            'socioeconomic position and physical activity'],
        interpretation=('Participant diversity, meal diversity and repeated-measure density are '
                        'distinct quantities; the free-living cohort is dense in meals but narrow '
                        'in participants, and the standardized-meal cohort is the opposite.')),
        't21_representativeness')
    pd.set_option('display.width', 280)
    print(PEOPLE.round(3).to_string(index=False))
    print(); print(EXP.round(3).to_string(index=False))
    print(); print(COMP.round(3).to_string(index=False))
    print(); print('target-population comparison:', ref_status)
    print(); print(missing)
