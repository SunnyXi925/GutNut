"""Assemble submission-ready supplementary tables from the verified result files."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env

OUT = env.ROOT / 'supplementary'
OUT.mkdir(exist_ok=True)
R = env.RESULTS
def j(n): return json.load(open(R / f'{n}.json'))

# S1 regression lock
t0 = j('t0_reproduce')
s1 = pd.DataFrame(t0['checks']).T.reset_index().rename(columns={'index': 'quantity'})
s1.to_csv(OUT / 'Supplementary_Table_S1_regression_lock.csv', index=False)

# S2 CGMacros decomposition
dec = pd.read_csv(R / 't1_decomposition.csv')
dec.to_csv(OUT / 'Supplementary_Table_S2_cgmacros_models.csv', index=False)

# S3 CGMacros contrasts
c = pd.read_csv(R / 't1_contrasts.csv')
cols = ['contrast', 'method_name', 'reference_name', 'metric', 'method', 'reference',
        'difference', 'diff_lo', 'diff_hi', 'relative_pct', 'rel_lo', 'rel_hi',
        'participants', 'better', 'sign_p']
c[cols].to_csv(OUT / 'Supplementary_Table_S3_cgmacros_contrasts.csv', index=False)

# S4 external summary (PREDICT UK + frozen transport)
s = pd.read_csv(R / 't4_external_summary.csv')
s.groupby(['cohort', 'arm', 'model']).agg(
    meals=('meals', 'mean'), people=('people', 'mean'),
    rmse=('rmse', 'mean'), rmse_sd_across_draws=('rmse', 'std'),
    profile_rmse=('profile_rmse', 'mean')).round(5).reset_index().to_csv(
    OUT / 'Supplementary_Table_S4_external_models.csv', index=False)

# S5 external contrasts
e = pd.read_csv(R / 't4_external_contrasts.csv')
e.groupby(['cohort', 'arm', 'contrast', 'metric']).agg(
    method=('method', 'mean'), reference=('reference', 'mean'),
    difference=('difference', 'mean'), diff_lo=('diff_lo', 'mean'), diff_hi=('diff_hi', 'mean'),
    relative_pct=('relative_pct', 'mean'), rel_lo=('rel_lo', 'mean'), rel_hi=('rel_hi', 'mean'),
    participants=('participants', 'mean'), better=('better', 'mean'),
    sign_p=('sign_p', 'mean'), draws=('draw', 'size')).round(5).reset_index().to_csv(
    OUT / 'Supplementary_Table_S5_external_contrasts.csv', index=False)

# S6 prior-arm equivalence
a = pd.read_csv(R / 't4b_arm_contrasts.csv')
a.groupby(['cohort', 'arm', 'reference_arm']).agg(
    difference=('difference', 'mean'), lo=('diff_lo', 'mean'), hi=('diff_hi', 'mean'),
    relative_pct=('relative_pct', 'mean'), participants=('participants', 'mean'),
    better=('better', 'mean'), sign_p=('sign_p', 'mean'),
    draws_equivalent=('equivalent_within_margin', 'sum'), draws=('draw', 'size'),
    margin=('margin', 'mean')).round(6).reset_index().to_csv(
    OUT / 'Supplementary_Table_S6_prior_arm_equivalence.csv', index=False)

# S7 selected configurations per outer fold and arm
pd.read_csv(R / 't4_uk_selected.csv').to_csv(
    OUT / 'Supplementary_Table_S7_selected_configurations.csv', index=False)

# S8 permutation negative control
t3 = j('t3_permutation')
perm = pd.read_csv(R / 't3_permutation_distribution.csv')
pd.DataFrame(t3['permutation_reference']).to_csv(
    OUT / 'Supplementary_Table_S8_permutation_negative_control.csv', index=False)
perm.describe().round(6).reset_index().to_csv(
    OUT / 'Supplementary_Table_S8b_permutation_distribution_summary.csv', index=False)

# S9 identifiability simulation
pd.read_csv(R / 't9_simulation_grid.csv').round(4).to_csv(
    OUT / 'Supplementary_Table_S9_identifiability_simulation.csv', index=False)

# S10 calibration, conditioning, second sensor
t8 = j('t8_calibration')
flat = []
for k, v in t8['coverage'].items():
    flat.append(dict(block='posterior_predictive_coverage', quantity=k, value=round(v, 4)))
for k, v in t8['calibration'].items():
    flat.append(dict(block='calibration', quantity=k, value=round(v, 4)))
for k, v in t8['condition_number'].items():
    flat.append(dict(block='calibration_design_conditioning', quantity=k, value=v))
for k, v in t8['posterior_contraction_pct'].items():
    flat.append(dict(block='posterior_contraction_pct', quantity=k, value=round(v, 2)))
d = t8['dexcom_replication']
for k in ('meals', 'participants', 'shared', 'intercept_only', 'full'):
    flat.append(dict(block='second_sensor_dexcom', quantity=k, value=d[k]))
for k, v in d['channel_increment'].items():
    flat.append(dict(block='second_sensor_dexcom_channel_increment', quantity=k,
                     value=round(v, 5) if isinstance(v, float) else v))
for r in t8['shared_residual_variance']:
    flat.append(dict(block='shared_residual_variance',
                     quantity=f"fold_{r['fold']}_sigma2", value=round(r['shared_sigma2'], 4)))
pd.DataFrame(flat).to_csv(OUT / 'Supplementary_Table_S10_calibration_and_replication.csv', index=False)

# S11 cohort description
pd.read_csv(R / 't10_table1_cgmacros.csv').to_csv(
    OUT / 'Supplementary_Table_S11a_cgmacros_characteristics.csv', index=False)
pd.read_csv(R / 't10_table1_predict.csv').to_csv(
    OUT / 'Supplementary_Table_S11b_predict_characteristics.csv', index=False)
pd.read_csv(R / 't10_responder.csv').to_csv(
    OUT / 'Supplementary_Table_S11c_participant_level_gains.csv', index=False)

# S12 provenance
prov = json.load(open(env.PROV / 'cohort_and_predict_provenance_audit.json'))
rows = [dict(resource=k, path=v['path'], per_meal_postprandial=v['per_meal_postprandial'],
             cgm=v['cgm'], meal_composition=v['meal_composition'], verdict=v['verdict'])
        for k, v in prov['cohort_inventory'].items()]
pd.DataFrame(rows).to_csv(OUT / 'Supplementary_Table_S12_cohort_eligibility_audit.csv', index=False)

files = sorted(p.name for p in OUT.glob('Supplementary_Table_*.csv'))
env.dump(dict(tables=files, count=len(files)), 't12_supplement_manifest')
print('\n'.join(files))
print(len(files), 'supplementary tables written to', OUT)
