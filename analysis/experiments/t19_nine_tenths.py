"""How much of the attainable calibration gain is reached by three meals?

The manuscript states that three meals capture roughly nine tenths of the gain
available by eleven meals. This experiment writes that quantity down as an
estimator, resamples it over participants, and reports the non-monotonicity of
the underlying curve instead of describing it as a plateau.

Definition. On the common query set, let R(k) be the participant-balanced RMSE
after k calibration meals and let K* be the longest history evaluated. The
captured fraction at k is

    F(k) = ( R(0) - R(k) ) / ( R(0) - R(Kstar) ),

the share of the total reduction between no personal evidence and the longest
history. A second, assumption-free variant replaces R(Kstar) with the lowest
RMSE attained at any evaluated history, min_j R(j), because the curve is not
monotone. Both are resampled by participant with the same bootstrap indices
used elsewhere in the paper.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats

SEED = 20261111
B = 2000
KS = (0, 1, 2, 3, 5, 7, 9, 11)
KSTAR = 11
PROTO = 'common'

# Per-participant mean squared errors on the common query set, written by t13.
losses = pd.read_csv(env.RESULTS / 't13_person_mse.csv')
losses = losses[losses.protocol.eq(PROTO) & losses.model.eq('full')]
piv = losses.pivot(index='pid', columns='history_k', values='mse')
piv = piv[[k for k in KS]]
pids = piv.index.to_numpy()
M = piv.to_numpy(float)
rmse = lambda m: np.sqrt(m.mean(0))


def fractions(m):
    r = rmse(m)
    r0 = r[list(KS).index(0)]
    rk = r[list(KS).index(KSTAR)]
    rmin = r.min()
    out = {}
    for k in KS:
        i = list(KS).index(k)
        out[f'F_vs_K{KSTAR}_k{k}'] = (r0 - r[i]) / (r0 - rk)
        out[f'F_vs_best_k{k}'] = (r0 - r[i]) / (r0 - rmin)
    return r, out


r_obs, f_obs = fractions(M)
rng = np.random.default_rng(SEED)
ix = rng.integers(0, len(pids), size=(B, len(pids)))
draw = {k: [] for k in f_obs}
curve = []
for b in range(B):
    _, fb = fractions(M[ix[b]])
    for k in draw:
        draw[k].append(fb[k])
    curve.append(rmse(M[ix[b]]))
curve = np.asarray(curve)

table = []
for j, k in enumerate(KS):
    table.append(dict(history_k=k, rmse=float(r_obs[j]),
                      rmse_lo=float(np.quantile(curve[:, j], .025)),
                      rmse_hi=float(np.quantile(curve[:, j], .975)),
                      marginal_gain=float(r_obs[j - 1] - r_obs[j]) if j else np.nan,
                      captured_fraction_vs_K11=float(f_obs[f'F_vs_K{KSTAR}_k{k}']),
                      captured_fraction_vs_K11_lo=float(np.quantile(draw[f'F_vs_K{KSTAR}_k{k}'], .025)),
                      captured_fraction_vs_K11_hi=float(np.quantile(draw[f'F_vs_K{KSTAR}_k{k}'], .975)),
                      captured_fraction_vs_best=float(f_obs[f'F_vs_best_k{k}']),
                      captured_fraction_vs_best_lo=float(np.quantile(draw[f'F_vs_best_k{k}'], .025)),
                      captured_fraction_vs_best_hi=float(np.quantile(draw[f'F_vs_best_k{k}'], .975))))
T = pd.DataFrame(table)
T.to_csv(env.RESULTS / 't19_captured_fraction.csv', index=False)

dec = np.diff(r_obs)
mono = bool(np.all(dec <= 0))
increases = [dict(from_k=int(KS[i]), to_k=int(KS[i + 1]), delta=float(dec[i]))
             for i in range(len(dec)) if dec[i] > 0]
best_k = int(KS[int(np.argmin(r_obs))])

env.dump(dict(
    protocol=PROTO, histories=list(KS), reference_history=KSTAR, bootstrap=B, seed=SEED,
    estimator=('F(k) = (R(0) - R(k)) / (R(0) - R(Kstar)) on the common query set, '
               'with R the participant-balanced RMSE of the full model'),
    denominator_vs_K11=float(r_obs[0] - r_obs[list(KS).index(KSTAR)]),
    denominator_vs_best=float(r_obs[0] - r_obs.min()),
    three_meal_fraction_vs_K11=dict(point=float(f_obs[f'F_vs_K{KSTAR}_k3']),
                                    lo=float(np.quantile(draw[f'F_vs_K{KSTAR}_k3'], .025)),
                                    hi=float(np.quantile(draw[f'F_vs_K{KSTAR}_k3'], .975))),
    three_meal_fraction_vs_best=dict(point=float(f_obs['F_vs_best_k3']),
                                     lo=float(np.quantile(draw['F_vs_best_k3'], .025)),
                                     hi=float(np.quantile(draw['F_vs_best_k3'], .975))),
    monotone_decreasing=mono, best_history=best_k, increases=increases,
    note=('The curve is not monotone: the listed history-to-history increases are reported '
          'rather than smoothed, and the captured fraction is given against both the longest '
          'history and the best attained history.')), 't19_nine_tenths')
pd.set_option('display.width', 240)
print(T.round(4).to_string(index=False))
print('\nmonotone:', mono, 'best history:', best_k)
print('increases:', increases)
