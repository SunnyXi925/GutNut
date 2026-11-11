"""Export source data for the revised Figure 2 and Figure 3 panels."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols
from masked_operator import splits
from conditional_adapter import adapter_configs

FD = env.ROOT / 'figure_data'
FD.mkdir(exist_ok=True)
KS = (0, 1, 2, 3, 5, 7, 9, 11)
COMMON_START, SEED, B = 12, 20261109, 2000
CFG = adapter_configs()['masked_host_taxa_bw1.0_mean0.5']

cg = env.load_cgmacros().sort_values(['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
rows = []
for fold, (a, b) in enumerate(splits(cg, 5)):
    tr, te = cg.iloc[a].reset_index(drop=True), cg.iloc[b].reset_index(drop=True)
    base, ad = runner.fit_fold(tr)
    m = runner.configure(ad, CFG, tr)
    for k in KS:
        p = runner.predict_subsets(m, te, max(k, 1), COMMON_START)
        if k == 0:
            p['full'] = p['shared'] + p['prior_mac'] * p['phi_mac'] + p['prior_lipid'] * p['phi_lipid']
            p['intercept_only'] = p['shared']
        p['history_k'] = k
        rows.append(p)
P = pd.concat(rows, ignore_index=True)

# panel 2a: per-participant MSE per history, plus shared predictor
recs, agg = [], []
rng = np.random.default_rng(SEED)
shared_L = stats.person_losses(P[P.history_k.eq(3)], 'shared')
for lab, L in [('Shared', shared_L)] + [(str(k), stats.person_losses(P[P.history_k.eq(k)], 'full')) for k in KS]:
    for pid, v in L.mse.items():
        recs.append(dict(history=lab, participant_id=pid, mse=float(v)))
    a = L.mse.to_numpy()
    ix = rng.integers(0, len(a), size=(B, len(a)))
    draws = np.sqrt(a[ix].mean(1))
    agg.append(dict(panel='2a', model=lab, value=float(np.sqrt(a.mean())),
                    lo=float(np.quantile(draws, .025)), hi=float(np.quantile(draws, .975))))
pd.DataFrame(recs).to_csv(FD / 'V38_2a.csv', index=False)
pd.DataFrame(agg).to_csv(FD / 'V38_2a_intervals.csv', index=False)

# panel 2c: per-participant gain versus the shared predictor, three history lengths
sh = np.sqrt(shared_L.mse)
gain = []
for k in (1, 3, 11):
    L = stats.person_losses(P[P.history_k.eq(k)], 'full')
    g = pd.DataFrame(dict(participant_id=L.index.to_numpy(),
                          gain=(sh.to_numpy() - np.sqrt(L.mse.to_numpy())), history=str(k)))
    gain.append(g)
G = pd.concat(gain, ignore_index=True)
order = G[G.history.eq('3')].sort_values('gain', ascending=False).participant_id.tolist()
G['row'] = G.participant_id.map({p: i + 1 for i, p in enumerate(order)})
G.to_csv(FD / 'V38_2c.csv', index=False)

# panel 3f: permutation reference band for the microbial prior
t3 = json.load(open(env.RESULTS / 't3_permutation.json'))
band = {r['metric']: r for r in t3['permutation_reference']}
pd.DataFrame([dict(reference='matched_uniform', perm_lo=band['micro_vs_population']['perm_q025'],
                   perm_hi=band['micro_vs_population']['perm_q975'],
                   perm_median=band['micro_vs_population']['perm_median'],
                   observed=band['micro_vs_population']['observed'],
                   permutation_p=band['micro_vs_population']['permutation_p']),
              dict(reference='matched_host_prior', perm_lo=band['micro_vs_host']['perm_q025'],
                   perm_hi=band['micro_vs_host']['perm_q975'],
                   perm_median=band['micro_vs_host']['perm_median'],
                   observed=band['micro_vs_host']['observed'],
                   permutation_p=band['micro_vs_host']['permutation_p'])]).to_csv(
    FD / 'V38_3f_permutation.csv', index=False)

# panel 3c annotation: GutNut's own absolute values
L3 = {c: stats.person_losses(P[P.history_k.eq(3)], c) for c in ('full',)}
pd.DataFrame([dict(model='GutNut', rmse=stats.balanced(L3['full'], 'mse'),
                   profile_rmse=stats.balanced(L3['full'], 'profile_mse'))]).to_csv(
    FD / 'V38_3c_reference.csv', index=False)

env.dump(dict(histories=list(KS), common_query_start=COMMON_START, bootstrap=B, seed=SEED,
              files=[p.name for p in sorted(FD.glob('V38_*.csv'))]), 't14_figure_data')
print(pd.DataFrame(agg).round(4).to_string(index=False))
print('\nexported:', [p.name for p in sorted(FD.glob('V38_*.csv'))])
