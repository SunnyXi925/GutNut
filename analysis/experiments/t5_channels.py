"""Does the lipid channel need fat quality rather than total fat? (PREDICT UK)

The two-channel mask is instantiated with different exposures by substituting the columns the
adapter reads, so the upstream model code is untouched. Saturation constants are the median
non-zero exposure in the training fold, which removes the hard-coded 5 g / 20 g choice and
makes the constant a nested, data-driven quantity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols
from masked_operator import splits
from conditional_adapter import adapter_configs

K, QUERY_AFTER, SEED, DRAWS = 3, 3, 20261109, 3
CFG = adapter_configs()['masked_taxa_bw1.0_mean0.5']      # microbial arm, no host information
E = 'exposure__'
VARIANTS = {
    'fibre_totalfat':  (f'{E}Fiber, total dietary (g)', f'{E}Total Fat (g)'),
    'fibre_sfa':       (f'{E}Fiber, total dietary (g)', f'{E}Fatty acids, total saturated (g)'),
    'fibre_mufa':      (f'{E}Fiber, total dietary (g)', f'{E}Fatty acids, total monounsaturated (g)'),
    'fibre_pufa':      (f'{E}Fiber, total dietary (g)', f'{E}Fatty acids, total polyunsaturated (g)'),
    'macmix_totalfat': ('__macmix__',                   f'{E}Total Fat (g)'),
    'macmix_sfa':      ('__macmix__',                   f'{E}Fatty acids, total saturated (g)'),
}
MAC_ITEMS = ['Fiber, total dietary (g)', 'Magnesium (mg)', 'Potassium (mg)', 'Vitamin C (mg)',
             'Folate, food (mcg)', 'Vitamin K (phylloquinone) (mcg)',
             'Vitamin E (alpha-tocopherol) (mg)', 'Carotene, alpha (mcg)', 'Carotene, beta (mcg)',
             'Cryptoxanthin, beta (mcg)', 'Lycopene (mcg)', 'Lutein + zeaxanthin (mcg)']


def add_macmix(frame, scale=None):
    """RMS-standardised sum of the twelve expert-assigned MAC exposures."""
    cols = [E + c for c in MAC_ITEMS]
    a = frame[cols].to_numpy(float)
    if scale is None:
        scale = np.maximum(np.sqrt((a * a).mean(0)), 1e-9)
    out = frame.copy()
    out['__macmix__'] = (a / scale).sum(1)
    return out, scale


def substitute(frame, spec):
    """Expose the chosen pair as food__fiber_g / food__fat_g for the unchanged adapter."""
    out = frame.copy()
    out['food__fiber_g'] = frame[spec[0]].to_numpy(float)
    out['food__fat_g'] = frame[spec[1]].to_numpy(float)
    return out


def median_nonzero(frame, spec):
    u = []
    for c in spec:
        v = frame[c].to_numpy(float)
        v = v[v > 0]
        u.append(float(np.median(v)) if len(v) else 1.0)
    return u


uk = env.load_predict('UK')
env.sha256_manifest([env.cohort_file('predict.parquet')], 't5')
rows, sel = [], []
for fold, (a, b) in enumerate(splits(uk, 5)):
    tr_raw, te_raw = uk.iloc[a].reset_index(drop=True), uk.iloc[b].reset_index(drop=True)
    protocols.assert_fold_disjoint(tr_raw, te_raw)
    tr_m, scale = add_macmix(tr_raw)
    te_m, _ = add_macmix(te_raw, scale)
    for name, spec in VARIANTS.items():
        u = median_nonzero(tr_m, spec)
        trs = protocols.inject_random_order(substitute(tr_m, spec), SEED + fold)
        base, ad = runner.fit_fold(trs)
        ad.unit = np.array(u)                      # data-driven saturation constants
        m = runner.configure(ad, CFG, trs)
        for draw in range(DRAWS):
            tes = protocols.inject_random_order(substitute(te_m, spec), SEED + 500 + 13 * draw)
            p = runner.predict_subsets(m, tes, K, QUERY_AFTER)
            p['variant'] = name; p['draw'] = draw; p['outer_fold'] = fold
            rows.append(p)
        sel.append(dict(fold=fold, variant=name, u_channel1=u[0], u_channel2=u[1],
                        channel1=spec[0], channel2=spec[1]))
    print('fold', fold, 'done', flush=True)

P = pd.concat(rows, ignore_index=True)
pd.DataFrame(sel).to_csv(env.RESULTS / 't5_channel_saturation.csv', index=False)

tab, eff = [], []
for name in VARIANTS:
    for draw in sorted(P.draw.unique()):
        d = P[(P.variant == name) & (P.draw == draw)]
        L = {c: stats.person_losses(d, c) for c in
             ['shared', 'intercept_only', 'intercept_plus_mac', 'intercept_plus_lipid', 'full']}
        for c, v in L.items():
            tab.append(dict(variant=name, draw=draw, model=c, meals=len(d), people=len(v),
                            rmse=stats.balanced(v, 'mse'), profile_rmse=stats.balanced(v, 'profile_mse')))
        for cname, mm, rr in [('channel_increment', 'full', 'intercept_only'),
                              ('lipid_contribution', 'full', 'intercept_plus_mac'),
                              ('mac_contribution', 'full', 'intercept_plus_lipid')]:
            eff.append(dict(variant=name, draw=draw, contrast=cname,
                            **stats.paired_bootstrap(L[mm], L[rr], 'mse', seed=SEED + draw)))
T = pd.DataFrame(tab); E_ = pd.DataFrame(eff)
T.to_csv(env.RESULTS / 't5_channel_models.csv', index=False)
E_.to_csv(env.RESULTS / 't5_channel_contrasts.csv', index=False)

agg = E_.groupby(['variant', 'contrast']).agg(
    difference=('difference', 'mean'), lo=('diff_lo', 'mean'), hi=('diff_hi', 'mean'),
    relative_pct=('relative_pct', 'mean'), better=('better', 'mean'),
    sign_p=('sign_p', 'mean')).round(5)
full = T[T.model == 'full'].groupby('variant').rmse.mean().round(5)
env.dump(dict(calibration_meals=K, draws=DRAWS, seed=SEED, prior_arm='micro (taxa)',
              saturation_rule='median non-zero exposure in the training fold',
              full_model_rmse_by_variant=full.to_dict(),
              contrasts=agg.reset_index().to_dict(orient='records')), 't5_channels')
pd.set_option('display.width', 240)
print('\nfull-model RMSE by channel instantiation:'); print(full.to_string())
print('\ncontrasts:'); print(agg.to_string())
