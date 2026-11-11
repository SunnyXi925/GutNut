"""Calibration-set selection: which meals should a person measure first?

The first-three protocol used in the manuscript is a convenience rule. The
random-three protocol weakens the nutrient-channel increment, and a large
fraction of three-meal designs are badly conditioned. This module turns that
observation into an explicit design problem: given a pool of candidate meals
whose nutrient composition is known before the meal is eaten, choose the
``k`` meals that will be measured.

Every selector except ``oracle`` uses only meal composition and the prior, so
each one is deployable. ``oracle`` minimises the realised query error and is
reported only as an attainable upper bound.
"""
from itertools import combinations
import numpy as np
import pandas as pd

SELECTORS = ('first', 'random', 'diverse', 'd_optimal', 'bayes_d_optimal',
             'a_optimal', 'i_optimal', 'min_condition', 'oracle')


def _candidates(pool_size, k):
    return list(combinations(range(pool_size), k))


def reference_moment(adapter, train, max_rows=4000, seed=0):
    """Second-moment matrix of the design over the population exposure distribution.

    I-optimality needs a reference distribution of the exposures at which
    predictions will be requested. The training fold supplies it, so the
    criterion never looks at the held-out participant's own query meals.
    """
    X = adapter.design(train)
    if len(X) > max_rows:
        X = X[np.random.default_rng(seed).choice(len(X), max_rows, replace=False)]
    return X.T @ X / len(X)


def choose(name, A_pool, prior_prec, s2, rng=None, query_error=None, k=3, M_ref=None):
    """Return the chosen row indices of ``A_pool``."""
    combos = _candidates(len(A_pool), k)
    if name == 'first':
        return tuple(range(k))
    if name == 'random':
        return combos[rng.integers(len(combos))]
    if name == 'oracle':
        return combos[int(np.argmin(query_error))]
    scores = []
    for c in combos:
        A = A_pool[list(c)]
        if name == 'd_optimal':
            scores.append(-np.linalg.det(A.T @ A))
        elif name == 'bayes_d_optimal':
            scores.append(-np.linalg.det(prior_prec + A.T @ A / s2))
        elif name == 'a_optimal':
            V = np.linalg.inv(prior_prec + A.T @ A / s2)
            scores.append(np.trace(V))
        elif name == 'i_optimal':
            V = np.linalg.inv(prior_prec + A.T @ A / s2)
            scores.append(float(np.sum(V * M_ref)))
        elif name == 'min_condition':
            scores.append(np.linalg.cond(A))
        elif name == 'diverse':
            P = A[:, 1:]
            d = [np.linalg.norm(P[i] - P[j]) for i, j in combinations(range(len(P)), 2)]
            scores.append(-min(d))
        else:
            raise ValueError(name)
    return combos[int(np.argmin(scores))]


def posterior(prior_prec, prior_mean, A, r, s2, idx=None):
    ix = np.arange(A.shape[1]) if idx is None else np.asarray(idx)
    P = prior_prec[np.ix_(ix, ix)]
    a = A[:, ix]
    cov = np.linalg.inv(P + a.T @ a / s2)
    beta = cov @ (P @ prior_mean[ix] + a.T @ r / s2)
    full = np.zeros(A.shape[1])
    full[ix] = beta
    return full, cov


def evaluate(adapter, test, pool_size, k, selectors=SELECTORS, random_draws=20, seed=0,
             M_ref=None):
    """Score every selector on one test fold with a query set that never moves."""
    from .runner import prior_from_z
    test = test.sort_values(['participant_id', 'timestamp']).reset_index(drop=True)
    base_all = adapter.base.predict(test)
    design_all = adapter.design(test)
    target = test.target.to_numpy(float)
    first = test.drop_duplicates('participant_id')
    zrows = None if adapter.trait == 'uniform' else adapter.trait_matrix(first)
    zindex = {p: i for i, p in enumerate(first.participant_id.to_numpy())}
    s2 = adapter.s2 * adapter.noise
    rng = np.random.default_rng(seed)
    rows = []
    for pid, g in test.groupby('participant_id', sort=False):
        pos = g.index.to_numpy()
        if len(pos) <= pool_size:
            continue
        pool, query_pos = pos[:pool_size], pos[pool_size:]
        A_pool = design_all[pool]
        r_pool = target[pool] - base_all[pool]
        X = design_all[query_pos]
        y = target[query_pos]
        base_q = base_all[query_pos]
        if zrows is not None:
            mu, cov = prior_from_z(adapter, zrows[zindex[pid]])
        else:
            mu, cov = adapter.mu.copy(), adapter.sigma.copy()
        prec = np.linalg.inv(cov)
        combos = _candidates(pool_size, k)
        err, preds = [], []
        for c in combos:
            A = A_pool[list(c)]
            beta, _ = posterior(prec, mu, A, r_pool[list(c)], s2)
            p = base_q + X @ beta
            preds.append(p)
            err.append(float(np.mean((p - y) ** 2)))
        for name in selectors:
            draws = range(random_draws) if name == 'random' else [0]
            for d in draws:
                c = choose(name, A_pool, prec, s2, rng=rng, query_error=err, k=k, M_ref=M_ref)
                A = A_pool[list(c)]
                rsub = r_pool[list(c)]
                beta, pcov = posterior(prec, mu, A, rsub, s2)
                beta_i, _ = posterior(prec, mu, A, rsub, s2, idx=(0,))
                rec = test.iloc[query_pos][['meal_event_id', 'participant_id',
                                            'family_group', 'target']].copy()
                rec['selector'] = name
                rec['calibration_meals'] = ';'.join(str(i) for i in c)
                rec['draw'] = d
                rec['shared'] = base_q
                rec['full'] = base_q + X @ beta
                rec['intercept_only'] = base_q + X @ beta_i
                rec['cond_number'] = float(np.linalg.cond(A))
                rec['log_det_information'] = float(np.linalg.slogdet(prec + A.T @ A / s2)[1])
                rec['posterior_sd_mac'] = float(np.sqrt(pcov[1, 1]))
                rec['posterior_sd_lipid'] = float(np.sqrt(pcov[2, 2]))
                rec['prior_sd_mac'] = float(np.sqrt(cov[1, 1]))
                rec['prior_sd_lipid'] = float(np.sqrt(cov[2, 2]))
                rows.append(rec)
    return pd.concat(rows, ignore_index=True)
