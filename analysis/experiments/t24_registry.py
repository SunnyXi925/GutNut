"""Statistical registry, tolerance recount, freezing timeline and headline provenance.

Four things a reader needs in order to audit this paper are produced here from
the result files rather than written by hand:

S18  every reported P value with its test, unit, null hypothesis, pairs, ties,
     tolerance, reference distribution, multiplicity family and role, together
     with Holm-adjusted values inside the confirmatory secondary family;
S19  the same sign tests recomputed at three tolerances, with an explicit flag
     for any conclusion that would change;
S20  what was frozen and when, with the code entry point for each component;
S28  a map from every headline number in the main text to the file, column or
     key and script that produces it, checked numerically against the text.
"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats

OUT = env.ROOT / 'supplementary'
OUT.mkdir(exist_ok=True)
R = env.RESULTS
ART = env.ROOT / 'sn-article.tex'


def load(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def jload(name):
    p = R / f'{name}.json'
    return json.load(open(p)) if p.exists() else None


# ---------------------------------------------------------------- S18 registry
rows = []


def reg(label, source, test, unit, null, pairs, ties, p, family, role, notes=''):
    rows.append(dict(reported_quantity=label, source=source, test=test,
                     experimental_unit=unit, null_hypothesis=null,
                     pairs_compared=pairs, ties=ties,
                     tolerance=stats.COMPARISON_TOLERANCE if test.startswith('sign') else '',
                     reference_distribution=('exact binomial' if test.startswith('sign')
                                             else 'permutation'),
                     p_value=p, multiplicity_family=family, role=role, notes=notes))


t1 = load('t1_contrasts.csv')
pri = t1[(t1.contrast == 'channel_increment_PRIMARY') & (t1.metric == 'rmse')].iloc[0]
reg('nutrient-channel increment, free-living cohort, three meals',
    'results/t1_contrasts.csv', 'sign test, exact two-sided', 'participant',
    'equal participant-balanced squared error for the full and intercept-only updates',
    int(pri.n_compared), int(pri.ties), float(pri.sign_p), 'primary', 'confirmatory primary',
    'designated as primary for reporting, not registered in advance')

t4 = load('t4_external_contrasts.csv')
t4r = t4[t4.metric.eq('rmse')]
uk = t4r[(t4r.cohort == 'PREDICT_UK') & (t4r.arm == 'micro')
         & (t4r.contrast == 'channel_increment_PRIMARY')]
reg('nutrient-channel increment, standardized-meal cohort',
    'results/t4_external_contrasts.csv', 'sign test, exact two-sided', 'participant',
    'equal participant-balanced squared error for the full and intercept-only updates',
    int(uk.n_compared.mean()), int(uk.ties.mean()), float(uk.sign_p.mean()),
    'key secondary', 'confirmatory secondary', 'mean over five calibration-order draws')

us = t4r[(t4r.cohort == 'PREDICT_US_transport') & (t4r.arm == 'micro')]
usc = us[us.contrast == 'channel_increment_PRIMARY']
usl = us[us.contrast == 'level_correction']
reg('frozen cross-cohort transport, channel increment',
    'results/t4_external_contrasts.csv', 'sign test, exact two-sided', 'participant',
    'equal participant-balanced squared error for the full and intercept-only updates',
    int(usc.n_compared.mean()), int(usc.ties.mean()), float(usc.sign_p.mean()),
    'key secondary', 'confirmatory secondary', 'model locked on the source cohort')
reg('frozen cross-cohort transport, response-level correction',
    'results/t4_external_contrasts.csv', 'sign test, exact two-sided', 'participant',
    'equal participant-balanced squared error for the intercept-only update and the shared predictor',
    int(usl.n_compared.mean()), int(usl.ties.mean()), float(usl.sign_p.mean()),
    'key secondary', 'confirmatory secondary', 'model locked on the source cohort')

for tag, label in (('t3_permutation', 'microbial advantage over the population prior, permutation null'),
                   ('t18_permutation_nulls_A', 'microbial advantage, alternative null models A'),
                   ('t18_permutation_nulls_B', 'microbial advantage, alternative null models B')):
    d = jload(tag)
    if not d:
        continue
    if tag == 't3_permutation':
        ref = d['permutation_reference'][0]
        reg(label, f'results/{tag}.json', 'permutation', 'participant',
            'the microbiome-to-participant link carries no information beyond the weighting itself',
            d['permutations'], '', float(ref['permutation_p']), 'key secondary',
            'confirmatory secondary', 'link broken in prior estimation and prior conditioning')
    else:
        for s in d['summary']:
            if s['metric'] != 'micro_vs_population':
                continue
            reg(f"microbial advantage, {s['null']} null", f'results/{tag}.json', 'permutation',
                'participant',
                'the microbiome-to-participant link carries no information beyond the weighting itself',
                d['permutations_per_null'], '', float(s['permutation_p']),
                'exploratory', 'exploratory diagnostic', 'alternative exchangeability unit')

t11 = jload('t11_foodscore')
if t11:
    for key, lab in (('CGMacros_ordering', 'free-living'),
                     ('PREDICT_UK_ordering', 'standardized-meal')):
        blk = t11.get(key, {}).get('full_vs_shared')
        if blk:
            reg(f'within-person food ordering, {lab} cohort', 'results/t11_foodscore.json',
                'sign test, exact two-sided', 'participant',
                'equal within-person ordering concordance for the personalized and shared models',
                int(blk['n_compared']), int(blk['ties']), float(blk['sign_p']),
                'key secondary', 'confirmatory secondary',
                'participants with identical concordance under both models are ties')
    fc = t11.get('PREDICT_UK_ordering', {}).get('full_vs_food_compass')
    if fc:
        reg('within-person ordering against a population nutritional-quality score',
            'results/t11_foodscore.json', 'sign test, exact two-sided', 'participant',
            'equal within-person ordering concordance for the model and the quality score',
            int(fc['n_compared']), int(fc['ties']), float(fc['sign_p']),
            'exploratory', 'exploratory diagnostic',
            'belongs to the shared predictor, not to personal calibration')

S18 = pd.DataFrame(rows)
fam = S18.multiplicity_family.eq('key secondary')
p = S18.loc[fam, 'p_value'].astype(float).to_numpy()
order = np.argsort(p)
m = len(p)
holm = np.empty(m)
running = 0.0
for rank, idx in enumerate(order):
    running = max(running, (m - rank) * p[idx])
    holm[idx] = min(1.0, running)
S18['holm_adjusted_p'] = np.nan
S18.loc[fam, 'holm_adjusted_p'] = holm
S18['significant_at_0.05_uncorrected'] = S18.p_value.astype(float) < 0.05
S18['significant_at_0.05_after_holm'] = np.where(
    fam, S18.holm_adjusted_p < 0.05, np.nan)
S18.to_csv(OUT / 'Supplementary_Table_S18_statistical_registry.csv', index=False)

# ------------------------------------------------------- S19 tolerance recount
tol_rows = []
for fname, keys in (('t1_contrasts.csv', ['contrast', 'metric']),
                    ('t4_external_contrasts.csv', ['cohort', 'arm', 'contrast', 'metric']),
                    ('t16_rule_contrasts.csv', ['cohort', 'contrast']),
                    ('t17_design_contrasts.csv', ['cohort', 'pool', 'selector']),
                    ('t20_robustness.csv', ['analysis']),
                    ('t13_calibration_contrasts.csv', ['protocol', 'history_k', 'contrast'])):
    d = load(fname)
    if d is None or 'sign_p_tol_1e12' not in d.columns:
        continue
    for _, r in d.iterrows():
        ps = [r.sign_p_tol_1e12, r.sign_p_tol_1e10, r.sign_p_tol_1e08]
        sig = [bool(x < 0.05) if pd.notna(x) else None for x in ps]
        tol_rows.append(dict(
            source=fname, key=' / '.join(str(r[k]) for k in keys if k in d.columns),
            pairs=int(r.participants) if 'participants' in d.columns else '',
            better=int(r.better), worse=int(r.worse), ties=int(r.ties),
            min_abs_nonzero_delta=float(r.min_abs_nonzero_delta),
            sign_p_tol_1e12=r.sign_p_tol_1e12, sign_p_tol_1e10=r.sign_p_tol_1e10,
            sign_p_tol_1e08=r.sign_p_tol_1e08,
            conclusion_changes=bool(len(set(x for x in sig if x is not None)) > 1)))
S19 = pd.DataFrame(tol_rows)
S19.to_csv(OUT / 'Supplementary_Table_S19_tolerance_recount.csv', index=False)

# ------------------------------------------------------- S20 freezing timeline
S20 = pd.DataFrame([
    dict(component='nutrient mask (total dietary fibre, total fat)', set_from='nutritional reasoning and the expert review',
         frozen_before='every quantitative evaluation in this paper', refitted_per_fold='no',
         code_entry_point='lib/model/conditional_adapter.py ConditionalAdapter.fit'),
    dict(component='saturating exposure transform', set_from='specification',
         frozen_before='every quantitative evaluation in this paper', refitted_per_fold='no',
         code_entry_point='lib/model/conditional_adapter.py ConditionalAdapter.design'),
    dict(component='saturation constants u_MAC = 5 g, u_LIPID = 20 g', set_from='median non-zero exposure in training data',
         frozen_before='every quantitative evaluation in this paper', refitted_per_fold='no, sensitivity in t5',
         code_entry_point='lib/model/conditional_adapter.py ConditionalAdapter.fit'),
    dict(component='three-coordinate personal structure', set_from='specification',
         frozen_before='every quantitative evaluation in this paper', refitted_per_fold='no',
         code_entry_point='lib/runner.py SUBSETS'),
    dict(component='shared predictor and model family', set_from='training partition',
         frozen_before='each held-out fold', refitted_per_fold='yes',
         code_entry_point='lib/runner.py fit_fold'),
    dict(component='empirical-Bayes prior moments', set_from='training partition',
         frozen_before='each held-out fold', refitted_per_fold='yes',
         code_entry_point='lib/model/conditional_adapter.py ConditionalAdapter.fit'),
    dict(component='shared residual variance', set_from='training partition',
         frozen_before='each held-out fold', refitted_per_fold='yes',
         code_entry_point='lib/model/conditional_adapter.py ConditionalAdapter.fit'),
    dict(component='ridge penalty and Gaussian prior scale of the comparison rules',
         set_from='training partition, deployment protocol', frozen_before='each held-out fold',
         refitted_per_fold='yes', code_entry_point='lib/baselines.py tune_scalar'),
    dict(component='participant-balanced RMSE as the reporting metric',
         set_from='chosen with knowledge of free-living results',
         frozen_before='the standardized-meal analysis', refitted_per_fold='not applicable',
         code_entry_point='lib/stats.py balanced'),
    dict(component='nutrient-channel increment as the primary estimand',
         set_from='designated with knowledge of free-living results',
         frozen_before='the external analysis', refitted_per_fold='not applicable',
         code_entry_point='experiments/t1_decomposition.py CONTRASTS'),
    dict(component='protocol family, prior arms, history grid', set_from='specification',
         frozen_before='the standardized-meal analysis', refitted_per_fold='no',
         code_entry_point='lib/protocols.py, lib/arms.py'),
    dict(component='entire model for the cross-cohort transport', set_from='source cohort only',
         frozen_before='the first target-cohort record was read', refitted_per_fold='no',
         code_entry_point='experiments/t4_external.py train_override'),
    dict(component='numerical comparison tolerance', set_from='double-precision error analysis',
         frozen_before='the recount of every sign test', refitted_per_fold='no',
         code_entry_point='lib/stats.py COMPARISON_TOLERANCE'),
])
S20.to_csv(OUT / 'Supplementary_Table_S20_freezing_timeline.csv', index=False)

# --------------------------------------------- S21-S27 from the new experiments
def dump(name, frame):
    if frame is not None:
        frame.to_csv(OUT / name, index=False)
        return len(frame)
    return 0

made = {}
made['S21a'] = dump('Supplementary_Table_S21a_adaptation_rules.csv', load('t16_rule_models.csv'))
made['S21b'] = dump('Supplementary_Table_S21b_adaptation_rule_contrasts.csv', load('t16_rule_contrasts.csv'))
made['S21c'] = dump('Supplementary_Table_S21c_adaptation_fold_settings.csv', load('t16_fold_settings.csv'))
made['S22'] = dump('Supplementary_Table_S22_calibration_design.csv', load('t17_design_selectors.csv'))
made['S23'] = dump('Supplementary_Table_S23_sensitivity.csv', load('t20_robustness.csv'))
made['S23b'] = dump('Supplementary_Table_S23b_leave_one_out.csv', load('t20_influence.csv'))
made['S24'] = dump('Supplementary_Table_S24_participant_characteristics.csv',
                   load('t21_participant_characteristics.csv'))
made['S25a'] = dump('Supplementary_Table_S25a_exposure_diversity.csv', load('t21_exposure_diversity.csv'))
made['S25b'] = dump('Supplementary_Table_S25b_between_cohort_differences.csv',
                    load('t21_between_cohort_differences.csv'))
made['S26'] = dump('Supplementary_Table_S26_captured_fraction.csv', load('t19_captured_fraction.csv'))
perm = [load(f't18_permutation_summary_{t}.csv') for t in ('A', 'B')]
perm = [p for p in perm if p is not None]
if perm:
    made['S27'] = dump('Supplementary_Table_S27_permutation_nulls.csv',
                       pd.concat(perm, ignore_index=True))
made['S29'] = dump('Supplementary_Table_S29_figure_audit.csv', load('t22_figure_audit.csv'))

# ------------------------------------------------- S28 headline provenance map
HEADLINE = [
    ('2.364', 'shared predictor, free-living cohort', 'results/t1_decomposition.csv',
     "model == 'shared', column rmse", 'experiments/t1_decomposition.py'),
    ('2.041', 'intercept-only update, free-living cohort', 'results/t1_decomposition.csv',
     "model == 'intercept_only', column rmse", 'experiments/t1_decomposition.py'),
    ('1.977', 'full update, free-living cohort', 'results/t1_decomposition.csv',
     "model == 'full', column rmse", 'experiments/t1_decomposition.py'),
    ('0.064', 'nutrient-channel increment', 'results/t1_decomposition.json',
     'primary.point', 'experiments/t1_decomposition.py'),
    ('3.1', 'nutrient-channel increment, per cent', 'results/t1_decomposition.json',
     'primary.relative_pct', 'experiments/t1_decomposition.py'),
    ('83.6', 'share of the total gain from the response level',
     'results/t1_decomposition.json', 'level_correction_share_of_total_pct',
     'experiments/t1_decomposition.py'),
    ('4.5', 'channel increment, standardized-meal cohort', 'results/t4_external_contrasts.csv',
     "cohort PREDICT_UK, arm micro, channel_increment_PRIMARY, relative_pct",
     'experiments/t4_external.py'),
    ('6.9', 'channel increment under frozen transport', 'results/t4_external_contrasts.csv',
     "cohort PREDICT_US_transport, arm micro, channel_increment_PRIMARY, relative_pct",
     'experiments/t4_external.py'),
    ('0.90', 'fraction of the eleven-meal gain captured at three meals',
     'results/t19_nine_tenths.json', 'three_meal_fraction_vs_K11.point',
     'experiments/t19_nine_tenths.py'),
    ('5.6', 'channel increment under D-optimal calibration selection',
     'results/t17_design_selectors.csv',
     "cohort PREDICT_UK, pool 5, selector d_optimal, channel_increment_pct",
     'experiments/t17_calibration_design.py'),
    ('3.6', 'channel increment of the zero-mean BLUP over an offset, free-living cohort',
     'results/t16_rule_contrasts.csv', "contrast blup_full_vs_eb_intercept, relative_pct",
     'experiments/t16_fair_baselines.py'),
    ('69.6', 'between-participant share of food-score variance', 'results/t11_foodscore.json',
     'variance_decomposition', 'experiments/t11_foodscore.py'),
    ('0.73', 'permutation P value for the microbial advantage', 'results/t3_permutation.json',
     'permutation_reference[0].permutation_p', 'experiments/t3_permutation.py'),
]
S28 = pd.DataFrame([dict(value=v, quantity=q, file=f, locator=l, script=s)
                    for v, q, f, l, s in HEADLINE])
if ART.exists():
    text = ART.read_text()
    S28['appears_in_main_text'] = [bool(re.search(re.escape(v), text)) for v in S28.value]
else:
    # the manuscript source is not part of the code repository; the provenance map is
    # still written, with the text-presence column left undetermined
    S28['appears_in_main_text'] = None
S28.to_csv(OUT / 'Supplementary_Table_S28_headline_provenance.csv', index=False)

env.dump(dict(registry_rows=len(S18), tolerance_rows=len(S19),
              tolerance_conclusion_changes=int(S19.conclusion_changes.sum()) if len(S19) else 0,
              freezing_rows=len(S20), tables=made,
              headline_rows=len(S28),
              manuscript_source_present=bool(ART.exists()),
              headline_all_found=(bool(S28.appears_in_main_text.all())
                                  if ART.exists() else None),
              holm_family_size=int(fam.sum()),
              holm_changes_any_conclusion=bool(
                  (S18.loc[fam, 'significant_at_0.05_uncorrected']
                   != S18.loc[fam, 'significant_at_0.05_after_holm']).any())),
         't24_registry')
pd.set_option('display.width', 280)
print(S18[['reported_quantity', 'test', 'pairs_compared', 'ties', 'p_value',
           'multiplicity_family', 'holm_adjusted_p']].to_string(index=False))
print('\ntolerance rows', len(S19), 'conclusion changes',
      int(S19.conclusion_changes.sum()) if len(S19) else 0)
if ART.exists():
    print('headline numbers found in the main text:',
          int(S28.appears_in_main_text.sum()), '/', len(S28))
    print(S28[~S28.appears_in_main_text][['value', 'quantity']].to_string(index=False))
else:
    print('manuscript source not present; provenance map written without the text check')
