"""Three evaluation protocols on the same cohort (CGMacros).

fixed       - the original design: query always starts at the sixth meal, so meals 2-5 are
              discarded when k < 5. Comparable across k, but not a deployment scenario.
deployment  - query starts immediately after the calibration meals.
random      - calibration meals drawn at random rather than taken as the first k.

A claim that survives all three is not an artefact of the evaluation design.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols
from masked_operator import splits
from conditional_adapter import adapter_configs

SEED, DRAWS = 20261109, 5
CFG = adapter_configs()['masked_host_taxa_bw1.0_mean0.5']
KS = (1, 3, 5)
COLS = ['shared', 'raw_personal_mean', 'intercept_only', 'full']

cg = env.load_cgmacros().sort_values(['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
env.sha256_manifest([env.MODEL / 'data/cgmacros_functional.parquet'], 't6')

rows = []
for fold, (a, b) in enumerate(splits(cg, 5)):
    tr, te = cg.iloc[a].reset_index(drop=True), cg.iloc[b].reset_index(drop=True)
    protocols.assert_fold_disjoint(tr, te, 'participant_id')
    base, ad = runner.fit_fold(tr)
    m = runner.configure(ad, CFG, tr)
    for k in KS:
        p = runner.predict_subsets(m, te, k, 5)
        p['protocol'] = 'fixed'; p['history_k'] = k; p['draw'] = 0; p['outer_fold'] = fold
        rows.append(p)
        p = runner.predict_subsets(m, te, k, k)
        p['protocol'] = 'deployment'; p['history_k'] = k; p['draw'] = 0; p['outer_fold'] = fold
        rows.append(p)
        for draw in range(DRAWS):
            ter = protocols.inject_random_order(te, SEED + 31 * draw)
            p = runner.predict_subsets(ter, k, 5) if False else runner.predict_subsets(m, ter, k, 5)
            p['protocol'] = 'random'; p['history_k'] = k; p['draw'] = draw; p['outer_fold'] = fold
            rows.append(p)
    print('fold', fold, 'done', flush=True)
P = pd.concat(rows, ignore_index=True)

tab, eff = [], []
for proto in sorted(P.protocol.unique()):
    for k in KS:
        for draw in sorted(P[(P.protocol == proto) & (P.history_k == k)].draw.unique()):
            d = P[(P.protocol == proto) & (P.history_k == k) & (P.draw == draw)]
            L = {c: stats.person_losses(d, c) for c in COLS}
            for c, v in L.items():
                tab.append(dict(protocol=proto, history_k=k, draw=draw, model=c, meals=len(d),
                                people=len(v), rmse=stats.balanced(v, 'mse'),
                                profile_rmse=stats.balanced(v, 'profile_mse')))
            eff.append(dict(protocol=proto, history_k=k, draw=draw, contrast='channel_increment',
                            **stats.paired_bootstrap(L['full'], L['intercept_only'], 'mse', seed=SEED + draw)))
T, E = pd.DataFrame(tab), pd.DataFrame(eff)
T.to_csv(env.RESULTS / 't6_protocol_models.csv', index=False)
E.to_csv(env.RESULTS / 't6_protocol_contrasts.csv', index=False)

agg = E.groupby(['protocol', 'history_k']).agg(
    meals=('participants', 'mean'), difference=('difference', 'mean'), lo=('diff_lo', 'mean'),
    hi=('diff_hi', 'mean'), relative_pct=('relative_pct', 'mean'), better=('better', 'mean'),
    sign_p=('sign_p', 'mean'), draws=('draw', 'size')).round(5)
mt = T[T.model.isin(['intercept_only', 'full'])].groupby(
    ['protocol', 'history_k', 'model']).agg(rmse=('rmse', 'mean'), meals=('meals', 'mean')).round(4)
env.dump(dict(seed=SEED, draws=DRAWS,
              note=('The fixed protocol keeps the query set identical across k and is therefore the '
                    'comparable one; the deployment protocol uses every meal after calibration and is '
                    'the realistic one; the random protocol removes any ordering confound.'),
              contrasts=agg.reset_index().to_dict(orient='records')), 't6_protocols')
pd.set_option('display.width', 240)
print('\nRMSE by protocol:'); print(mt.to_string())
print('\nchannel increment (full vs intercept-only):'); print(agg.to_string())
