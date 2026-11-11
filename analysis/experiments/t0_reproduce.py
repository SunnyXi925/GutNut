"""Regression lock: recompute every headline number in GutNut v37 from saved predictions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env

PUBLISHED = {
    'shared': 2.364, 'k1': 2.172, 'k3': 1.977, 'k5': 1.961,
    'intercept_only': 2.041, 'nutrients_only': 2.164,
    'intercept_mac': 2.010, 'intercept_lipid': 2.052,
}
PUBLISHED_PROFILE = {'k3': 1.812, 'intercept_only': 1.881, 'nutrients_only': 1.853,
                     'intercept_mac': 1.822, 'intercept_lipid': 1.885, 'shared': 1.881}


def balanced(d, pred):
    e = d[pred].to_numpy(float) - d.target.to_numpy(float)
    g = pd.DataFrame({'pid': d.participant_id.to_numpy(), 'e': e, 'e2': e ** 2})
    z = g.groupby('pid').agg(mse=('e2', 'mean'), bias=('e', 'mean'))
    return float(np.sqrt(z.mse.mean())), float(np.sqrt(np.maximum(0, z.mse - z.bias ** 2).mean()))


env.sha256_manifest([env.SOURCE / 'module_predictions.csv',
                     env.SOURCE / 'microbial_history_predictions.csv',
                     env.SOURCE / 'CG_future_predictions.csv',
                     env.MODEL / 'data/cgmacros_functional.parquet'], 't0')

m = pd.read_csv(env.SOURCE / 'module_predictions.csv')
cg = pd.read_csv(env.SOURCE / 'CG_future_predictions.csv')
shared = cg[cg.model.eq('same_base_without_history') & cg.history_k.eq(0)]
NAME = {'full': 'k3', 'offset_only': 'intercept_only', 'nutrients_only': 'nutrients_only',
        'offset_fibre': 'intercept_mac', 'offset_fat': 'intercept_lipid',
        'history_0': 'prior_only_k0', 'history_1': 'k1', 'history_3': 'k3_dup', 'history_5': 'k5'}
got = {}
for model, d in m.groupby('model'):
    r, p = balanced(d, 'prediction')
    got[NAME.get(model, model)] = dict(rmse=r, profile_rmse=p, meals=len(d),
                                       people=int(d.participant_id.nunique()))

sr, sp = balanced(shared, 'prediction')
got['shared'] = dict(rmse=sr, profile_rmse=sp, meals=len(shared),
                     people=int(shared.participant_id.nunique()))

checks = {}
for k, want in PUBLISHED.items():
    checks[k] = dict(kind='rmse', published=want, recomputed=round(got[k]['rmse'], 4),
                     match=bool(abs(got[k]['rmse'] - want) < 1.5e-3))
for k, want in PUBLISHED_PROFILE.items():
    checks['profile_' + k] = dict(kind='profile_rmse', published=want,
                                  recomputed=round(got[k]['profile_rmse'], 4),
                                  match=bool(abs(got[k]['profile_rmse'] - want) < 1.5e-3))

# microbial prior contrast at k=3, from the history predictions
h = pd.read_csv(env.SOURCE / 'microbial_history_predictions.csv')
h3 = h[h.history_k.eq(3)]
mic = balanced(h3[h3.model.eq('conditional_micro')], 'prediction')[0]
pop = balanced(h3[h3.model.eq('matched_uniform')], 'prediction')[0]
hos = balanced(h3[h3.model.eq('matched_host_prior')], 'prediction')[0]
checks['micro_vs_population_k3'] = dict(kind='difference', published=0.0061,
                                        recomputed=round(pop - mic, 5),
                                        match=bool(abs((pop - mic) - 0.0061) < 1e-3))
checks['micro_vs_host_k3'] = dict(kind='difference', published=0.0042,
                                  recomputed=round(hos - mic, 5),
                                  match=bool(abs((hos - mic) - 0.0042) < 1e-3))

# headline relative reductions
rel_total = 100 * (got['shared']['rmse'] - got['k3']['rmse']) / got['shared']['rmse']
rel_channel = 100 * (got['intercept_only']['rmse'] - got['k3']['rmse']) / got['intercept_only']['rmse']
checks['relative_shared_to_full'] = dict(kind='pct', published=16.4, recomputed=round(rel_total, 3),
                                         match=bool(abs(rel_total - 16.4) < 0.1))
checks['relative_intercept_to_full'] = dict(kind='pct', published=3.1, recomputed=round(rel_channel, 3),
                                            match=bool(abs(rel_channel - 3.1) < 0.1))

# Previously unreported: the zero-meal (prior-only) prediction is worse than the shared predictor.
cold = dict(shared=round(got['shared']['rmse'], 4),
            prior_only_k0=round(got['prior_only_k0']['rmse'], 4),
            cold_start_penalty=round(got['prior_only_k0']['rmse'] - got['shared']['rmse'], 4))

share_of_total = 100 * (got['shared']['rmse'] - got['intercept_only']['rmse']) / (
    got['shared']['rmse'] - got['k3']['rmse'])
env.dump(dict(recomputed=got, checks=checks, cold_start=cold,
              intercept_share_of_total_gain_pct=round(share_of_total, 2)), 't0_reproduce')

bad = {k: v for k, v in checks.items() if not v['match']}
print(pd.DataFrame(checks).T.to_string())
print('\nintercept share of total gain: %.1f%%' % share_of_total)
print('cold start (0 meals) vs shared predictor:', cold)
assert not bad, bad
print('t0 OK -', len(checks), 'published values reproduced')
