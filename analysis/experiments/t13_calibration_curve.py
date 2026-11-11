"""How many measured meals are enough? Extend the calibration curve until it flattens.

The reviewer note is that stopping at five meals does not show where the curve plateaus.
Two readings are produced:
  common   - every history length is scored on one shared query set, so lengths are comparable;
  deployment - every history length uses all meals that follow it, which is the realistic setting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols
from masked_operator import splits
from conditional_adapter import adapter_configs

KS = (0, 1, 2, 3, 5, 7, 9, 11)
COMMON_START = 12          # all 44 participants contribute at least 13 meals
CFG = adapter_configs()['masked_host_taxa_bw1.0_mean0.5']
SEED = 20261109

cg = env.load_cgmacros().sort_values(['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
n = cg.groupby('participant_id').size()
assert n.min() > COMMON_START, n.min()
env.sha256_manifest([env.MODEL / 'data/cgmacros_functional.parquet'], 't13')

rows = []
for fold, (a, b) in enumerate(splits(cg, 5)):
    tr, te = cg.iloc[a].reset_index(drop=True), cg.iloc[b].reset_index(drop=True)
    protocols.assert_fold_disjoint(tr, te, 'participant_id')
    base, ad = runner.fit_fold(tr)
    m = runner.configure(ad, CFG, tr)
    for k in KS:
        kk = max(k, 1)
        for proto, qa in (('common', COMMON_START), ('deployment', max(kk, 1))):
            p = runner.predict_subsets(m, te, kk, qa)
            if k == 0:                       # prior only: no personal evidence is used
                p['full'] = p['shared'] + p['prior_mac'] * p['phi_mac'] + p['prior_lipid'] * p['phi_lipid']
                p['intercept_only'] = p['shared']
            p['history_k'] = k; p['protocol'] = proto; p['outer_fold'] = fold
            rows.append(p)
    print('fold', fold, 'done', flush=True)
P = pd.concat(rows, ignore_index=True)

tab, eff = [], []
for proto in ('common', 'deployment'):
    for k in KS:
        d = P[(P.protocol == proto) & (P.history_k == k)]
        L = {c: stats.person_losses(d, c) for c in ('shared', 'intercept_only', 'full')}
        for c, v in L.items():
            tab.append(dict(protocol=proto, history_k=k, model=c, meals=len(d), people=len(v),
                            rmse=stats.balanced(v, 'mse'), profile_rmse=stats.balanced(v, 'profile_mse')))
        if k > 0:
            eff.append(dict(protocol=proto, history_k=k, contrast='channel_increment',
                            **stats.paired_bootstrap(L['full'], L['intercept_only'], 'mse', seed=SEED)))
            eff.append(dict(protocol=proto, history_k=k, contrast='total_vs_shared',
                            **stats.paired_bootstrap(L['full'], L['shared'], 'mse', seed=SEED)))
T, E = pd.DataFrame(tab), pd.DataFrame(eff)
T.to_csv(env.RESULTS / 't13_calibration_curve.csv', index=False)
E.to_csv(env.RESULTS / 't13_calibration_contrasts.csv', index=False)

piv = T[T.model.eq('full')].pivot(index='history_k', columns='protocol', values='rmse')
piv['common_marginal_gain'] = -piv['common'].diff()
env.dump(dict(histories=list(KS), common_query_start=COMMON_START,
              full_rmse=piv.round(4).to_dict(),
              plateau_note=('Marginal gain per added meal on the common query set; the curve is '
                            'reported to eleven meals so the flattening point is visible rather than '
                            'assumed.'),
              channel_increment=E[E.contrast.eq('channel_increment')][
                  ['protocol', 'history_k', 'difference', 'diff_lo', 'diff_hi', 'relative_pct',
                   'better', 'sign_p']].round(5).to_dict(orient='records')), 't13_calibration_curve')
pd.set_option('display.width', 240)
print('\nfull-model RMSE by history length:'); print(piv.round(4).to_string())
print('\nchannel increment:')
print(E[E.contrast.eq('channel_increment')][['protocol', 'history_k', 'difference', 'diff_lo',
      'diff_hi', 'relative_pct', 'better', 'sign_p']].round(4).to_string(index=False))
