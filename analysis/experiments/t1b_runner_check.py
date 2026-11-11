"""Sanity gate: the local reduced-update runner must land on the published CGMacros values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols
from masked_operator import splits
from conditional_adapter import adapter_configs

f = env.load_cgmacros()
cfgs = adapter_configs()
cfg = dict(cfgs['masked_host_taxa_bw1.0_mean0.5'])
rows = []
for fold, (a, b) in enumerate(splits(f, 5)):
    tr, te = f.iloc[a].reset_index(drop=True), f.iloc[b].reset_index(drop=True)
    protocols.assert_fold_disjoint(tr, te, 'participant_id')
    base, ad = runner.fit_fold(tr)
    m = runner.configure(ad, cfg, tr)
    p = runner.predict_subsets(m, te, k=3, query_after=5)
    p['outer_fold'] = fold
    rows.append(p)
P = pd.concat(rows, ignore_index=True)
assert len(P) == 902 and P.participant_id.nunique() == 44
out = {}
for col in ['shared', 'raw_personal_mean', 'intercept_only', 'intercept_plus_mac',
            'intercept_plus_lipid', 'channels_only', 'full']:
    L = stats.person_losses(P, col)
    out[col] = dict(rmse=stats.balanced(L, 'mse'), profile_rmse=stats.balanced(L, 'profile_mse'))
print(pd.DataFrame(out).T.to_string())
print('\npublished: shared 2.3639 intercept_only 2.0408 full 1.9772 (full depends on the selected config)')
env.dump(out, 't1b_runner_check')
assert abs(out['shared']['rmse'] - 2.3639) < 1e-3, out['shared']
assert abs(out['intercept_only']['rmse'] - 2.0408) < 2e-2, out['intercept_only']
print('runner OK: shared reproduced exactly, intercept-only within tolerance')
