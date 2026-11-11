"""Uncertainty calibration, design conditioning, and a second-sensor replication (CGMacros).

Interval honesty is assessed with posterior predictive coverage rather than a calibration
error summary, following arXiv:2112.01477.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import norm
from lib import env, stats, runner
from masked_operator import splits
from conditional_adapter import adapter_configs

LEVELS = (0.5, 0.8, 0.95)
CFG = adapter_configs()['masked_host_taxa_bw1.0_mean0.5']
K = 3


def run(frame, target_col='target'):
    frame = frame.copy()
    if target_col != 'target':
        frame['target'] = frame[target_col]
    frame = frame[frame.target.notna()].reset_index(drop=True)
    rows, sig = [], []
    for fold, (a, b) in enumerate(splits(frame, 5)):
        tr, te = frame.iloc[a].reset_index(drop=True), frame.iloc[b].reset_index(drop=True)
        base, ad = runner.fit_fold(tr)
        m = runner.configure(ad, CFG, tr)
        p = runner.predict_subsets(m, te, K, 5)
        p['outer_fold'] = fold
        p['sigma2'] = m.s2 * m.noise
        rows.append(p)
        sig.append(dict(fold=fold, shared_sigma2=float(m.s2), noise_multiplier=float(m.noise),
                        train_people=int(tr.participant_id.nunique())))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sig)


f = env.load_cgmacros()
P, sigma_table = run(f)
env.sha256_manifest([env.cohort_file('cgmacros_functional.parquet')], 't8')

# --- posterior predictive coverage -------------------------------------------------
# predictive variance = x' V x + sigma^2 ; V is diagonal-approximated by stored sds
var_coef = (P.sd_intercept ** 2 + (P.phi_mac * P.sd_mac) ** 2 + (P.phi_lipid * P.sd_lipid) ** 2)
P['pred_sd'] = np.sqrt(var_coef + P.sigma2)
P['z'] = (P.target - P.full) / P.pred_sd
cov = {}
for l in LEVELS:
    zc = norm.ppf(0.5 + l / 2)
    cov[f'coverage_{int(100*l)}'] = float((P.z.abs() <= zc).mean())
    per = P.groupby('participant_id').apply(lambda g: (g.z.abs() <= zc).mean(), include_groups=False)
    cov[f'coverage_{int(100*l)}_participant_balanced'] = float(per.mean())

lm = np.polyfit(P.full, P.target, 1)
calib = dict(calibration_slope=float(lm[0]), calibration_intercept=float(lm[1]),
             pit_mean=float(norm.cdf(P.z).mean()), pit_sd=float(norm.cdf(P.z).std()),
             standardised_residual_sd=float(P.z.std()))

# --- design conditioning -------------------------------------------------------------
per_person = P.drop_duplicates('participant_id')
cond = dict(median=float(per_person.cond_number.median()),
            q25=float(per_person.cond_number.quantile(.25)),
            q75=float(per_person.cond_number.quantile(.75)),
            q90=float(per_person.cond_number.quantile(.9)),
            max=float(per_person.cond_number.max()),
            share_above_100=float((per_person.cond_number > 100).mean()))
L_full = stats.person_losses(P, 'full')
L_int = stats.person_losses(P, 'intercept_only')
gain = np.sqrt(L_int.mse) - np.sqrt(L_full.mse)
cnum = per_person.set_index('participant_id').cond_number.reindex(gain.index)
good = cnum <= cnum.median()
cond['gain_well_conditioned_median'] = float(gain[good].median())
cond['gain_ill_conditioned_median'] = float(gain[~good].median())
cond['spearman_gain_vs_condition'] = float(pd.Series(gain).corr(cnum, method='spearman'))

# posterior contraction, measured against the conditional prior actually used
contraction = {c: float((1 - P.drop_duplicates('participant_id')[f'sd_{c}'] /
                         P.drop_duplicates('participant_id')[f'prior_sd_{c}']).median() * 100)
               for c in ('intercept', 'mac', 'lipid')}

# --- second-sensor replication -------------------------------------------------------
aux = f[f.target_aux.notna()]
keep = aux.groupby('participant_id').size()
aux = aux[aux.participant_id.isin(keep[keep > 5].index)]
PA, _ = run(aux, 'target_aux')
La_full = stats.person_losses(PA, 'full')
La_int = stats.person_losses(PA, 'intercept_only')
La_shd = stats.person_losses(PA, 'shared')
dex = dict(meals=int(len(PA)), participants=int(PA.participant_id.nunique()),
           shared=stats.balanced(La_shd, 'mse'),
           intercept_only=stats.balanced(La_int, 'mse'),
           full=stats.balanced(La_full, 'mse'),
           channel_increment=stats.paired_bootstrap(La_full, La_int, 'mse', seed=20261108),
           level_correction=stats.paired_bootstrap(La_int, La_shd, 'mse', seed=20261108))

P[['meal_event_id', 'participant_id', 'target', 'shared', 'intercept_only', 'full',
   'pred_sd', 'z', 'cond_number', 'phi_mac', 'phi_lipid', 'beta_mac', 'beta_lipid',
   'sd_mac', 'sd_lipid', 'outer_fold']].to_csv(env.RESULTS / 't8_calibration_rows.csv', index=False)
env.dump(dict(shared_residual_variance=sigma_table.to_dict(orient='records'),
              note_on_sigma=('sigma^2 is a single empirical-Bayes residual variance shared by all '
                             'participants, not a per-participant quantity; with three calibration '
                             'meals and three coefficients there is no residual degree of freedom '
                             'for a personal estimate.'),
              coverage=cov, calibration=calib, condition_number=cond,
              posterior_contraction_pct=contraction,
              dexcom_replication=dex), 't8_calibration')

pd.set_option('display.width', 240)
print('shared sigma^2 per fold:'); print(sigma_table.to_string(index=False))
print('\ncoverage:', {k: round(v, 3) for k, v in cov.items()})
print('calibration:', {k: round(v, 4) for k, v in calib.items()})
print('condition number:', {k: round(v, 2) for k, v in cond.items()})
print('posterior contraction (%):', {k: round(v, 1) for k, v in contraction.items()})
print('\nDexcom replication: shared %.4f intercept %.4f full %.4f' % (dex['shared'], dex['intercept_only'], dex['full']))
ci = dex['channel_increment']
print('  channel increment %.4f (95%% CI %.4f-%.4f) %.2f%%, %d/%d better, p=%.4f'
      % (ci['difference'], ci['diff_lo'], ci['diff_hi'], ci['relative_pct'],
         ci['better'], ci['participants'], ci['sign_p']))
