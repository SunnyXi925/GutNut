"""Five null models for the microbiome-to-participant link.

The single permutation reported in the manuscript breaks the link in both roles
the microbiome plays and does so inside each outer fold. A reviewer cannot tell
from one null whether the absence of a microbial advantage survives stricter
exchangeability, so the link is broken here under five explicit null models that
differ in the exchangeability unit and in which role of the microbiome is
destroyed:

``within_fold``        any training participant may donate to any other, and the
                       same for test participants (the published design);
``within_stratum``     donors are restricted to the same glycaemic stratum, so
                       host-microbiome confounding is preserved;
``host_matched``       donors are restricted to a block of participants with
                       similar host feature vectors;
``conditioning_only``  only the test-side prior assignment is permuted, leaving
                       the population moment estimation untouched;
``estimation_only``    only the training-side moment estimation is permuted,
                       leaving the test-side assignment untouched.

Exchangeability unit: the participant. One complete microbial feature vector is
moved between participants; meals are never permuted, and no host variable is
ever touched. Model selection is not re-run inside a permutation, so the
configuration is held fixed at the manuscript configuration for every replicate.
"""
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from lib import env, stats, runner, arms
from masked_operator import splits
from conditional_adapter import adapter_configs

SEED = 20261111
PERMS = int(os.environ.get('T18_PERMS', 60))
K = 3
BLOCK = 5
NULLS = ('within_fold', 'within_stratum', 'host_matched',
         'conditioning_only', 'estimation_only')

CFGS = {k: v for k, v in adapter_configs().items() if v['food'] == 'masked'}
MICRO_CFG = 'masked_taxa_bw1.0_mean0.5'
HOST_CFG = 'masked_host_bw1.0_mean0.5'
POP_CFG = 'masked_uniform_noise1.0'
assert arms.arm_of(CFGS[MICRO_CFG]) == 'micro'
assert arms.arm_of(CFGS[HOST_CFG]) == 'host'
assert arms.arm_of(CFGS[POP_CFG]) == 'population'

f = env.load_cgmacros()
MICRO_COLS = [c for c in f.columns if c.startswith(('micro__', 'function__'))]
HOST_COLS = sorted(c for c in f.columns if c.startswith('person__'))
env.sha256_manifest([env.cohort_file('cgmacros_functional.parquet')], 't18')
print('microbial columns:', len(MICRO_COLS), 'host columns:', len(HOST_COLS), flush=True)


def blocks_of(frame, null):
    """Exchangeability blocks of participant identifiers for one null model."""
    u = frame.drop_duplicates('participant_id')
    pids = u.participant_id.to_numpy()
    if null == 'within_stratum':
        lab = u.stratum.to_numpy()
    elif null == 'host_matched':
        X = u[HOST_COLS].to_numpy(float)
        X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
        kk = max(2, len(pids) // BLOCK)
        lab = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit_predict(X)
    else:
        lab = np.zeros(len(pids), dtype=int)
    return pids, lab


def derangement(pids, lab, rng):
    """Permute participant labels inside each block, avoiding fixed points."""
    donor = pids.copy()
    for b in np.unique(lab):
        ix = np.flatnonzero(lab == b)
        if len(ix) < 2:
            continue
        sub = pids[ix]
        for _ in range(200):
            cand = rng.permutation(sub)
            if not np.any(cand == sub):
                break
        donor[ix] = cand
    return dict(zip(pids, donor))


def apply_map(frame, mapping):
    block = frame.drop_duplicates('participant_id').set_index('participant_id')[MICRO_COLS]
    out = frame.copy()
    out[MICRO_COLS] = block.loc[[mapping[p] for p in out.participant_id]].to_numpy()
    return out


def run(cfg_name, null=None, rng=None):
    rows = []
    for fold, (a, b) in enumerate(splits(f, 5)):
        tr, te = f.iloc[a].reset_index(drop=True), f.iloc[b].reset_index(drop=True)
        if null is not None:
            if null != 'conditioning_only':
                p, l = blocks_of(tr, null)
                tr = apply_map(tr, derangement(p, l, rng))
            if null != 'estimation_only':
                p, l = blocks_of(te, null)
                te = apply_map(te, derangement(p, l, rng))
        base, ad = runner.fit_fold(tr)
        m = runner.configure(ad, CFGS[cfg_name], tr)
        p = runner.predict_subsets(m, te, K, 5, {'full': (0, 1, 2)})
        p['outer_fold'] = fold
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


t0 = time.time()
obs = {name: run(cfg) for name, cfg in [('micro', MICRO_CFG), ('host', HOST_CFG),
                                        ('population', POP_CFG)]}
L = {k: stats.person_losses(v, 'full') for k, v in obs.items()}
observed = {
    'micro_vs_population': stats.paired_bootstrap(L['micro'], L['population'], 'mse', seed=SEED),
    'micro_vs_host': stats.paired_bootstrap(L['micro'], L['host'], 'mse', seed=SEED),
    'host_vs_population': stats.paired_bootstrap(L['host'], L['population'], 'mse', seed=SEED),
}
obs_rmse = stats.balanced(L['micro'], 'mse')
pop_rmse = stats.balanced(L['population'], 'mse')
host_rmse = stats.balanced(L['host'], 'mse')
print('observed micro %.5f host %.5f population %.5f' % (obs_rmse, host_rmse, pop_rmse), flush=True)

ONLY = [n for n in os.environ.get('T18_NULLS', ','.join(NULLS)).split(',') if n]
assert set(ONLY) <= set(NULLS), ONLY
TAG = os.environ.get('T18_TAG', 'all')
draws = []
for null in ONLY:
    rng = np.random.default_rng(SEED)
    for i in range(PERMS):
        pm = run(MICRO_CFG, null=null, rng=rng)
        Lp = stats.person_losses(pm, 'full')
        draws.append(dict(null=null, replicate=i,
                          rmse=stats.balanced(Lp, 'mse'),
                          profile_rmse=stats.balanced(Lp, 'profile_mse'),
                          vs_population=pop_rmse - stats.balanced(Lp, 'mse'),
                          vs_host=host_rmse - stats.balanced(Lp, 'mse')))
        if (i + 1) % 25 == 0:
            print(f'  {null} {i+1}/{PERMS} t={time.time()-t0:.0f}s', flush=True)
D = pd.DataFrame(draws)
D.to_csv(env.RESULTS / f't18_permutation_draws_{TAG}.csv', index=False)


def locate(value, series, label, null):
    s = np.asarray(series, float)
    ge = int((s >= value).sum())
    return dict(null=null, metric=label, observed=float(value),
                perm_median=float(np.median(s)), perm_sd=float(s.std(ddof=1)),
                perm_q025=float(np.quantile(s, .025)), perm_q975=float(np.quantile(s, .975)),
                quantile_of_observed=float((s < value).mean()),
                n_at_or_above=ge, permutation_p=float((1 + ge) / (1 + len(s))),
                monte_carlo_se_of_p=float(np.sqrt(0.25 / len(s))))

summary = []
for null, g in D.groupby('null', sort=False):
    summary.append(locate(observed['micro_vs_population']['difference'],
                          g.vs_population, 'micro_vs_population', null))
    summary.append(locate(observed['micro_vs_host']['difference'], g.vs_host,
                          'micro_vs_host', null))
    summary.append(locate(-obs_rmse, -g.rmse, 'negative_micro_rmse', null))
S = pd.DataFrame(summary)
S.to_csv(env.RESULTS / f't18_permutation_summary_{TAG}.csv', index=False)

env.dump(dict(
    permutations_per_null=PERMS, nulls=list(ONLY), seed=SEED, calibration_meals=K,
    exchangeability_unit='participant; one complete microbial feature vector is relocated',
    host_block_size=BLOCK, microbial_columns=len(MICRO_COLS), host_columns=len(HOST_COLS),
    model_selection='held fixed at the reported configuration inside every replicate',
    observed=dict(micro_rmse=obs_rmse, host_rmse=host_rmse, population_rmse=pop_rmse,
                  contrasts={k: {kk: v[kk] for kk in ('difference', 'diff_lo', 'diff_hi',
                                                      'better', 'worse', 'ties', 'sign_p')}
                             for k, v in observed.items()}),
    summary=S.to_dict(orient='records'),
    mde_80pct_power_micro_vs_host=stats.mde(observed['micro_vs_host']['diff_se']),
    interpretation=('If the observed microbial advantage lies inside the permutation reference '
                    'range under every null model, then no null model that preserves the stated '
                    'exchangeability unit is required to reproduce it.')), f't18_permutation_nulls_{TAG}')
pd.set_option('display.width', 260)
print(); print(S.round(5).to_string(index=False))
