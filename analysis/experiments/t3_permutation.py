"""Internal negative control for microbial conditioning (CGMacros).

Design follows the negative-control permutation logic of arXiv:2303.01552: break the
microbiome-to-participant correspondence, refit the empirical-Bayes prior, and compare the
observed microbial advantage with its permutation reference distribution.

The link is broken in both roles the microbiome plays: prior estimation (training
participants) and prior conditioning (test participants).
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols, arms
from masked_operator import splits
from conditional_adapter import adapter_configs

SEED = 20261108
PERMS = 200
K = 3
CFGS = {k: v for k, v in adapter_configs().items() if v['food'] == 'masked'}
MICRO_CFG = 'masked_taxa_bw1.0_mean0.5'     # microbial arm, no host information
HOST_CFG = 'masked_host_bw1.0_mean0.5'
POP_CFG = 'masked_uniform_noise1.0'
assert arms.arm_of(CFGS[MICRO_CFG]) == 'micro'
assert arms.arm_of(CFGS[HOST_CFG]) == 'host'
assert arms.arm_of(CFGS[POP_CFG]) == 'population'

f = env.load_cgmacros()
MICRO_COLS = [c for c in f.columns if c.startswith(('micro__', 'function__'))]
print('microbial columns:', len(MICRO_COLS))
env.sha256_manifest([env.MODEL / 'data/cgmacros_functional.parquet'], 't3')


def permute_microbiome(frame, rng):
    """Give every participant another participant's complete microbial vector."""
    pids = frame.participant_id.unique()
    donor = rng.permutation(pids)
    while np.any(donor == pids) and len(pids) > 1:
        donor = rng.permutation(pids)
    block = frame.drop_duplicates('participant_id').set_index('participant_id')[MICRO_COLS]
    mapping = {p: d for p, d in zip(pids, donor)}
    out = frame.copy()
    out[MICRO_COLS] = block.loc[[mapping[p] for p in out.participant_id]].to_numpy()
    return out


def run(cfg_name, permute_rng=None):
    rows = []
    for fold, (a, b) in enumerate(splits(f, 5)):
        tr, te = f.iloc[a].reset_index(drop=True), f.iloc[b].reset_index(drop=True)
        if permute_rng is not None:
            tr = permute_microbiome(tr, permute_rng)
            te = permute_microbiome(te, permute_rng)
        base, ad = runner.fit_fold(tr)
        m = runner.configure(ad, CFGS[cfg_name], tr)
        p = runner.predict_subsets(m, te, K, 5, {'full': (0, 1, 2)})
        p['outer_fold'] = fold
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


t0 = time.time()
obs = {name: run(cfg) for name, cfg in
       [('micro', MICRO_CFG), ('host', HOST_CFG), ('population', POP_CFG)]}
L = {k: stats.person_losses(v, 'full') for k, v in obs.items()}
observed = {
    'micro_vs_population': stats.paired_bootstrap(L['micro'], L['population'], 'mse', seed=SEED),
    'micro_vs_host': stats.paired_bootstrap(L['micro'], L['host'], 'mse', seed=SEED),
    'host_vs_population': stats.paired_bootstrap(L['host'], L['population'], 'mse', seed=SEED),
}
print('observed advantages:')
for k, v in observed.items():
    print(f'  {k:22s} {v["difference"]:+.5f} (95% CI {v["diff_lo"]:+.5f} {v["diff_hi"]:+.5f})')
print('elapsed', round(time.time() - t0, 1), 's')

rng = np.random.default_rng(SEED)
perm = []
for i in range(PERMS):
    pm = run(MICRO_CFG, permute_rng=rng)
    Lp = stats.person_losses(pm, 'full')
    perm.append(dict(replicate=i,
                     rmse=stats.balanced(Lp, 'mse'),
                     profile_rmse=stats.balanced(Lp, 'profile_mse'),
                     vs_population=stats.balanced(L['population'], 'mse') - stats.balanced(Lp, 'mse'),
                     vs_host=stats.balanced(L['host'], 'mse') - stats.balanced(Lp, 'mse')))
    if (i + 1) % 25 == 0:
        print(f'  permutation {i+1}/{PERMS} t={time.time()-t0:.0f}s', flush=True)
P = pd.DataFrame(perm)
P.to_csv(env.RESULTS / 't3_permutation_distribution.csv', index=False)

def locate(observed_value, draws, label):
    ge = int((draws >= observed_value).sum())
    return dict(metric=label, observed=float(observed_value),
                perm_median=float(np.median(draws)),
                perm_q025=float(np.quantile(draws, .025)),
                perm_q975=float(np.quantile(draws, .975)),
                perm_quantile_of_observed=float((draws < observed_value).mean()),
                n_perm_at_or_above=ge,
                permutation_p=float((1 + ge) / (1 + len(draws))))

report = dict(
    permutations=PERMS, seed=SEED, calibration_meals=K,
    microbial_columns=len(MICRO_COLS),
    configs=dict(micro=MICRO_CFG, host=HOST_CFG, population=POP_CFG),
    observed=observed,
    permutation_reference=[
        locate(observed['micro_vs_population']['difference'], P.vs_population.to_numpy(), 'micro_vs_population'),
        locate(observed['micro_vs_host']['difference'], P.vs_host.to_numpy(), 'micro_vs_host')],
    observed_micro_rmse=stats.balanced(L['micro'], 'mse'),
    permuted_micro_rmse_median=float(P.rmse.median()),
    mde_80pct_power_micro_vs_host=stats.mde(observed['micro_vs_host']['diff_se']),
    interpretation=('A real microbiome-to-participant link is not required to obtain the observed '
                    'advantage if the observed value falls inside the permutation reference range.'))
env.dump(report, 't3_permutation')
print(json.dumps(report['permutation_reference'], indent=1))
print('observed micro RMSE %.4f vs permuted median %.4f' % (report['observed_micro_rmse'], report['permuted_micro_rmse_median']))
