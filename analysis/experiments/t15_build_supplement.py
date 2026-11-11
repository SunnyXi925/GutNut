"""Build the English Supplementary Information LaTeX from the verified result tables."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from lib import env

SUP = env.ROOT / 'supplementary'
TEXDIR = env.ROOT / 'si_tables'
TEXDIR.mkdir(exist_ok=True)


def esc(x):
    s = str(x)
    for a, b in [('\\', r'\textbackslash '), ('_', r'\_'), ('%', r'\%'), ('&', r'\&'),
                 ('#', r'\#'), ('$', r'\$'), ('{', r'\{'), ('}', r'\}'), ('^', r'\^{}'),
                 ('~', r'\textasciitilde ')]:
        s = s.replace(a, b)
    return s


def render(csv, name, cols=None, rename=None, fmt=3, maxrows=None, small=True):
    d = pd.read_csv(SUP / csv)
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    if maxrows:
        d = d.head(maxrows)
    if rename:
        d = d.rename(columns=rename)
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: '' if pd.isna(v) else f'{v:.{fmt}f}')
    body = ' \\\\\n'.join(' & '.join(esc(v) for v in r) for r in d.itertuples(index=False))
    head = ' & '.join(r'\textbf{' + esc(c).replace('\\_', ' ') + '}' for c in d.columns)
    align = 'l' * len(d.columns)
    size = r'\footnotesize' if small else ''
    tex = (f'{{{size}\\setlength{{\\tabcolsep}}{{3pt}}\n'
           f'\\begin{{longtable}}{{{align}}}\n\\toprule\n{head} \\\\\n\\midrule\n\\endhead\n'
           f'{body} \\\\\n\\bottomrule\n\\end{{longtable}}}}\n')
    (TEXDIR / f'{name}.tex').write_text(tex)
    return len(d)


specs = [
 ('Supplementary_Table_S1_regression_lock.csv', 'S1', ['quantity', 'kind', 'published', 'recomputed', 'match'], None, 4),
 ('Supplementary_Table_S2_cgmacros_models.csv', 'S2', None, None, 4),
 ('Supplementary_Table_S3_cgmacros_contrasts.csv', 'S3',
  ['contrast', 'metric', 'method', 'reference', 'difference', 'diff_lo', 'diff_hi',
   'relative_pct', 'better', 'sign_p'], None, 4),
 ('Supplementary_Table_S4_external_models.csv', 'S4', None, None, 4),
 ('Supplementary_Table_S5_external_contrasts.csv', 'S5',
  ['cohort', 'arm', 'contrast', 'metric', 'difference', 'diff_lo', 'diff_hi',
   'relative_pct', 'better', 'sign_p'], None, 4),
 ('Supplementary_Table_S6_prior_arm_equivalence.csv', 'S6',
  ['cohort', 'arm', 'reference_arm', 'difference', 'lo', 'hi', 'relative_pct',
   'better', 'sign_p', 'draws_equivalent', 'margin'], None, 5),
 ('Supplementary_Table_S7_selected_configurations.csv', 'S7', None, None, 4),
 ('Supplementary_Table_S8_permutation_negative_control.csv', 'S8', None, None, 5),
 ('Supplementary_Table_S9_identifiability_simulation.csv', 'S9',
  ['k', 'rho', 'sigma', 'prior_bias', 'rmse_mac', 'rmse_lipid', 'transfer_rmse',
   'cov50_mac', 'cov80_mac', 'cov95_mac', 'shrink_mac', 'median_cond'], None, 3),
 ('Supplementary_Table_S10_calibration_and_replication.csv', 'S10', None, None, 4),
 ('Supplementary_Table_S11a_cgmacros_characteristics.csv', 'S11a', None, None, 2),
 ('Supplementary_Table_S11b_predict_characteristics.csv', 'S11b', None, None, 2),
 ('Supplementary_Table_S13a_baseline_models.csv', 'S13a', None, None, 4),
 ('Supplementary_Table_S13b_baseline_contrasts.csv', 'S13b',
  ['cohort', 'contrast', 'method', 'reference', 'difference', 'diff_lo', 'diff_hi',
   'relative_pct', 'better', 'participants', 'sign_p'], None, 4),
 ('Supplementary_Table_S14a_channel_models.csv', 'S14a', None, None, 4),
 ('Supplementary_Table_S14b_channel_contrasts.csv', 'S14b', None, None, 4),
 ('Supplementary_Table_S14c_channel_saturation_constants.csv', 'S14c', None, None, 3),
 ('Supplementary_Table_S15a_protocol_models.csv', 'S15a', None, None, 4),
 ('Supplementary_Table_S15b_protocol_contrasts.csv', 'S15b', None, None, 4),
 ('Supplementary_Table_S16a_within_person_ordering.csv', 'S16a', None, None, 4),
 ('Supplementary_Table_S16b_food_score_structure.csv', 'S16b', None, None, 3),
 ('Supplementary_Table_S16c_food_library_groups.csv', 'S16c', None, None, 0),
 ('Supplementary_Table_S17a_calibration_curve.csv', 'S17a', None, None, 4),
 ('Supplementary_Table_S17b_calibration_contrasts.csv', 'S17b',
  ['protocol', 'history_k', 'contrast', 'difference', 'diff_lo', 'diff_hi',
   'relative_pct', 'better', 'sign_p'], None, 4),
 ('Supplementary_Table_S12_cohort_eligibility_audit.csv', 'S12',
  ['resource', 'per_meal_postprandial', 'cgm', 'meal_composition'], None, 0),
]
counts = {}
for csv, name, cols, rename, fmt in specs:
    try:
        counts[name] = render(csv, name, cols, rename, fmt)
    except Exception as e:
        counts[name] = f'ERR {e}'
print(counts)
env.dump(counts, 't15_supplement_tables')
