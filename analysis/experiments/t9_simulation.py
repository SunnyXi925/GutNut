"""Identifiability of the three-coefficient adaptation space (synthetic ground truth).

Answers the question a reviewer will ask about a just-identified model: with k calibration
meals and correlated channel exposures, can the personal coefficients be recovered, and are
the posterior intervals honest? Reported alongside the real-data condition numbers so the
simulation speaks to the actual design, not an idealised one.

This is a software/statistical experiment on generated data and is labelled as such; it is
not a study cohort.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import norm
from lib import env

SEED = 20261109
REPS = 400
KS = (1, 3, 5, 10)
RHOS = (0.0, 0.3, 0.6, 0.9)
NOISES = (0.5, 1.0, 2.0)          # residual sd, mmol l^-1 h
PRIOR_BIAS = (0.0, 1.0)           # prior mean offset in units of prior sd (misspecification)
U = (5.0, 20.0)                   # saturation constants, g
PRIOR_SD = np.array([1.2, 1.0, 1.0])
LEVELS = (0.5, 0.8, 0.95)
Z = {l: float(norm.ppf(0.5 + l / 2)) for l in LEVELS}


def exposures(n, rho, rng):
    """Correlated fibre/fat doses mapped through the saturating link."""
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    fibre = np.abs(z[:, 0]) * 6.0
    fat = np.abs(z[:, 1]) * 22.0
    return np.column_stack([np.ones(n), fibre / (U[0] + fibre), fat / (U[1] + fat)])


rows = []
rng = np.random.default_rng(SEED)
for rho in RHOS:
    for sigma in NOISES:
        for bias in PRIOR_BIAS:
            for k in KS:
                rec = []
                for _ in range(REPS):
                    beta = rng.normal(0, PRIOR_SD)
                    prior_mean = bias * PRIOR_SD * rng.choice([-1.0, 1.0], size=3)
                    cov = np.diag(PRIOR_SD ** 2)
                    X = exposures(k, rho, rng)
                    y = X @ beta + rng.normal(0, sigma, k)
                    inv = np.linalg.inv(cov)
                    V = np.linalg.inv(inv + X.T @ X / sigma ** 2)
                    post = V @ (inv @ prior_mean + X.T @ y / sigma ** 2)
                    sd = np.sqrt(np.diag(V))
                    Xq = exposures(40, rho, rng)
                    pred_err = float(np.sqrt(np.mean((Xq @ post - Xq @ beta) ** 2)))
                    cover = {f'cover{int(100*l)}': np.abs(post - beta) <= sd * Z[l]
                             for l in LEVELS}
                    rec.append(dict(err_intercept=abs(post[0] - beta[0]),
                                    err_mac=abs(post[1] - beta[1]),
                                    err_lipid=abs(post[2] - beta[2]),
                                    sd_mac=sd[1], sd_lipid=sd[2],
                                    shrink_mac=1 - sd[1] / PRIOR_SD[1],
                                    shrink_lipid=1 - sd[2] / PRIOR_SD[2],
                                    cond=float(np.linalg.cond(X)) if k >= 3 else np.nan,
                                    transfer_rmse=pred_err,
                                    **{f'{n}_{c}': v[i] for n, i in
                                       [('intercept', 0), ('mac', 1), ('lipid', 2)]
                                       for c, v in cover.items()}))
                d = pd.DataFrame(rec)
                rows.append(dict(rho=rho, sigma=sigma, prior_bias=bias, k=k, reps=REPS,
                                 rmse_mac=float(np.sqrt((d.err_mac ** 2).mean())),
                                 rmse_lipid=float(np.sqrt((d.err_lipid ** 2).mean())),
                                 rmse_intercept=float(np.sqrt((d.err_intercept ** 2).mean())),
                                 transfer_rmse=float(d.transfer_rmse.mean()),
                                 shrink_mac=float(d.shrink_mac.mean()),
                                 shrink_lipid=float(d.shrink_lipid.mean()),
                                 median_cond=float(d.cond.median()),
                                 **{f'cov{int(100*l)}_{p}': float(d[f'{p}_cover{int(100*l)}'].mean())
                                    for l in LEVELS for p in ('intercept', 'mac', 'lipid')}))
grid = pd.DataFrame(rows)
grid.to_csv(env.RESULTS / 't9_simulation_grid.csv', index=False)

# real-data design conditions, so the grid is anchored to the actual calibration sets
try:
    import pyarrow  # noqa: F401
    real = pd.read_parquet(env.RESULTS / 't4_uk_predictions.parquet')
    real_cond = real.drop_duplicates('participant_id').cond_number
    real_note = dict(source='PREDICT UK three-meal calibration designs',
                     median=float(real_cond.median()),
                     q90=float(real_cond.quantile(.9)), max=float(real_cond.max()))
except Exception as e:
    real_note = dict(error=str(e)[:80])

base = grid[(grid.sigma == 1.0) & (grid.prior_bias == 0.0)]
env.dump(dict(reps=REPS, seed=SEED, levels=LEVELS,
              saturation_constants=dict(u_mac=U[0], u_lipid=U[1]),
              prior_sd=PRIOR_SD.tolist(),
              real_calibration_condition_numbers=real_note,
              headline=base[['k', 'rho', 'rmse_mac', 'rmse_lipid', 'transfer_rmse',
                             'cov95_mac', 'cov95_lipid', 'shrink_mac', 'median_cond']]
              .round(4).to_dict(orient='records'),
              reading=('Coverage near the nominal level with a correct prior shows the closed-form '
                       'intervals are honest; degradation with prior_bias=1 quantifies the cost of a '
                       'misspecified conditional prior; rising transfer_rmse with rho locates the '
                       'identifiability boundary of the two-channel design.')), 't9_simulation')
pd.set_option('display.width', 240)
print(base[['k', 'rho', 'rmse_mac', 'rmse_lipid', 'transfer_rmse', 'cov50_mac', 'cov80_mac',
            'cov95_mac', 'cov95_lipid', 'shrink_mac', 'median_cond']].round(3).to_string(index=False))
print()
print('prior misspecification (bias=1 sd, sigma=1):')
print(grid[(grid.sigma == 1.0) & (grid.prior_bias == 1.0)][
      ['k', 'rho', 'rmse_mac', 'transfer_rmse', 'cov95_mac', 'cov95_lipid']].round(3).to_string(index=False))
print()
print('real calibration condition numbers:', real_note)
