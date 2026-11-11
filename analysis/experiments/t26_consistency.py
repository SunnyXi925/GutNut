"""Numerical consistency check between the manuscript text and the result files.

Every headline number in the main text is recomputed here from the result file
that is supposed to produce it and compared with the printed value at the
precision at which it is printed. The check fails if any number in the text was
edited without being recomputed, or if a result file changed without the text
following it.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env

R = env.RESULTS


def j(n, *path):
    d = json.load(open(R / f'{n}.json'))
    for k in path:
        d = d[k] if not isinstance(k, int) else d[k]
    return float(d)


def csv_val(name, column, **where):
    d = pd.read_csv(R / name)
    for k, v in where.items():
        d = d[d[k].astype(str) == str(v)]
    assert len(d), (name, where)
    return float(d[column].mean())


CHECKS = [
    ('shared RMSE, free-living', 2.364, csv_val('t1_decomposition.csv', 'rmse', model='shared'), 3),
    ('intercept-only RMSE, free-living', 2.041,
     csv_val('t1_decomposition.csv', 'rmse', model='intercept_only'), 3),
    ('full RMSE, free-living', 1.977, csv_val('t1_decomposition.csv', 'rmse', model='full'), 3),
    ('channel increment', 0.064, j('t1_decomposition', 'primary', 'point'), 3),
    ('channel increment per cent', 3.1, j('t1_decomposition', 'primary', 'relative_pct'), 1),
    ('level-correction share', 83.6, j('t1_decomposition', 'level_correction_share_of_total_pct'), 1),
    ('unshrunk personal mean RMSE', 2.121,
     csv_val('t1_decomposition.csv', 'rmse', model='raw_personal_mean_k3'), 3),
    ('standardized-meal increment per cent', 4.5,
     csv_val('t4_external_contrasts.csv', 'relative_pct', cohort='PREDICT_UK', arm='micro',
             contrast='channel_increment_PRIMARY', metric='rmse'), 1),
    ('transport increment per cent', 6.9,
     csv_val('t4_external_contrasts.csv', 'relative_pct', cohort='PREDICT_US_transport',
             arm='micro', contrast='channel_increment_PRIMARY', metric='rmse'), 1),
    ('transport level correction', 0.001,
     csv_val('t4_external_contrasts.csv', 'difference', cohort='PREDICT_US_transport',
             arm='micro', contrast='level_correction', metric='rmse'), 3),
    ('least squares RMSE, free-living', 3.463,
     csv_val('t16_rule_models.csv', 'rmse', cohort='CGMacros', rule='ols_head'), 3),
    ('least squares RMSE, standardized', 9.953,
     csv_val('t16_rule_models.csv', 'rmse', cohort='PREDICT_UK', rule='ols_head'), 3),
    ('correlated BLUP RMSE, free-living', 1.980,
     csv_val('t16_rule_models.csv', 'rmse', cohort='CGMacros', rule='blup_full'), 3),
    ('correlated BLUP RMSE, standardized', 0.294,
     csv_val('t16_rule_models.csv', 'rmse', cohort='PREDICT_UK', rule='blup_full'), 3),
    ('BLUP over offset, free-living, per cent', 3.6,
     csv_val('t16_rule_contrasts.csv', 'relative_pct', cohort='CGMacros',
             contrast='blup_full_vs_eb_intercept'), 1),
    ('BLUP over offset, standardized, per cent', 3.7,
     csv_val('t16_rule_contrasts.csv', 'relative_pct', cohort='PREDICT_UK',
             contrast='blup_full_vs_eb_intercept'), 1),
    ('first-three increment, standardized, per cent', 2.8,
     csv_val('t17_design_selectors.csv', 'channel_increment_pct', cohort='PREDICT_UK',
             pool=5, selector='first'), 1),
    ('D-optimal increment, standardized, per cent', 5.6,
     csv_val('t17_design_selectors.csv', 'channel_increment_pct', cohort='PREDICT_UK',
             pool=5, selector='d_optimal'), 1),
    ('I-optimal RMSE, standardized', 0.2856,
     csv_val('t17_design_selectors.csv', 'full_rmse', cohort='PREDICT_UK', pool=5,
             selector='i_optimal'), 4),
    ('oracle RMSE, standardized', 0.2441,
     csv_val('t17_design_selectors.csv', 'full_rmse', cohort='PREDICT_UK', pool=5,
             selector='oracle'), 4),
    ('captured fraction at three meals', 0.90,
     j('t19_nine_tenths', 'three_meal_fraction_vs_K11', 'point'), 2),
    ('permutation P value', 0.73, j('t3_permutation', 'permutation_reference', 0, 'permutation_p'), 2),
    ('observed microbial advantage', 0.0022,
     j('t3_permutation', 'observed', 'micro_vs_population', 'difference'), 4),
    ('second-sensor increment', 0.092,
     csv_val('t20_robustness.csv', 'difference', analysis='second_sensor_dexcom'), 3),
]

rows = []
for label, printed, computed, digits in CHECKS:
    rows.append(dict(quantity=label, printed=printed, computed=computed, digits=digits,
                     rounded=round(computed, digits),
                     matches=bool(abs(round(computed, digits) - printed) <= 10 ** (-digits) / 1.999)))
T = pd.DataFrame(rows)
T.to_csv(R / 't26_consistency.csv', index=False)
env.dump(dict(checks=len(T), failures=int((~T.matches).sum()),
              failing=T[~T.matches].to_dict(orient='records'),
              note='printed values are those that appear in the main text'), 't26_consistency')
pd.set_option('display.width', 200)
print(T.to_string(index=False))
print('\nfailures:', int((~T.matches).sum()), 'of', len(T))
assert T.matches.all(), 'the manuscript and the result files disagree'
