"""Adaptation rules that share one nonlinear population predictor.

The review point this module answers is that comparing GutNut with a linear
mixed model changes two things at once: the function class of the population
predictor and the personal adaptation mechanism. Here the population predictor,
the calibration meals, the query meals and the personal design matrix are held
fixed, and only the rule that maps three calibration residuals to a personal
correction is varied:

``shared``            no personal correction (beta = 0);
``ols_head``          unregularised least squares on three meals;
``ridge_head``        isotropic ridge, penalty chosen inside the training fold;
``bayes_last_layer``  Gaussian prior N(0, tau^2 I), tau chosen inside the fold;
``blup_diag``         zero-mean random effects with a diagonal G;
``blup_full``         zero-mean random effects with the full empirical G, the
                      mixed-model BLUP written on the nonlinear residual;
``eb_population``     empirical-Bayes prior with a fitted mean and full
                      covariance and no side information;
``eb_conditional``    the same prior conditioned on participant side
                      information, which is GutNut as reported;
``eb_intercept``      the conditional prior restricted to the response level.

All rules see exactly the same residuals, so any difference is attributable to
the adaptation rule alone.
"""
import numpy as np
import pandas as pd

RIDGE_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
TAU_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
RULES = ('shared', 'ols_head', 'ridge_head', 'bayes_last_layer', 'blup_diag',
         'blup_full', 'eb_population', 'eb_intercept', 'eb_conditional')


def moment_G(betas, covs, about_zero=True):
    """Method-of-moments random-effect covariance from training participants."""
    B = np.asarray(betas, float)
    inner = np.asarray(covs, float).mean(0)
    if about_zero:
        G = B.T @ B / len(B) + inner
    else:
        d = B - B.mean(0)
        G = d.T @ d / len(B) + inner
    return G + np.eye(G.shape[0]) * 1e-8


def _solve(prior_prec, prior_mean, A, r, s2, idx=None):
    ix = np.arange(A.shape[1]) if idx is None else np.asarray(idx)
    P = prior_prec[np.ix_(ix, ix)]
    a = A[:, ix]
    cov = np.linalg.inv(P + a.T @ a / s2)
    beta = cov @ (P @ prior_mean[ix] + a.T @ r / s2)
    full = np.zeros(A.shape[1])
    full[ix] = beta
    return full, cov


def fixed_rules(adapter, ridge_alpha, tau, G_full, G_diag):
    """Prior precision and prior mean for every zero-mean adaptation rule."""
    d = len(adapter.mu)
    I = np.eye(d)
    zero = np.zeros(d)
    s2 = adapter.s2 * adapter.noise
    return {
        'ridge_head': (I * (ridge_alpha / s2), zero, None),
        'bayes_last_layer': (I / tau ** 2, zero, None),
        'blup_diag': (np.linalg.inv(G_diag), zero, None),
        'blup_full': (np.linalg.inv(G_full), zero, None),
    }


def predict_rules(adapter, test, k, query_after, ridge_alpha, tau, G_full, G_diag):
    """One pass over the test fold, every adaptation rule on the same meals."""
    from .runner import prior_from_z
    test = test.sort_values(['participant_id', 'timestamp']).reset_index(drop=True)
    base_all = adapter.base.predict(test)
    design_all = adapter.design(test)
    first = test.drop_duplicates('participant_id')
    zrows = None if adapter.trait == 'uniform' else adapter.trait_matrix(first)
    zindex = {p: i for i, p in enumerate(first.participant_id.to_numpy())}
    s2 = adapter.s2 * adapter.noise
    fixed = fixed_rules(adapter, ridge_alpha, tau, G_full, G_diag)
    pop_prec = np.linalg.inv(adapter.sigma)
    rows = []
    for pid, g in test.groupby('participant_id', sort=False):
        if len(g) <= query_after:
            continue
        pos = g.index.to_numpy()
        hist_pos, query_pos = pos[:k], pos[query_after:]
        assert not set(test.meal_event_id.to_numpy()[hist_pos]) & set(
            test.meal_event_id.to_numpy()[query_pos]), 'calibration/query leakage'
        A = design_all[hist_pos]
        r = test.target.to_numpy()[hist_pos] - base_all[hist_pos]
        X = design_all[query_pos]
        base_q = base_all[query_pos]
        rec = test.iloc[query_pos][['meal_event_id', 'participant_id', 'family_group',
                                    'food_id', 'target']].copy()
        rec['shared'] = base_q
        rec['ols_head'] = base_q + X @ (np.linalg.pinv(A) @ r)
        for name, (prec, mean, idx) in fixed.items():
            beta, _ = _solve(prec, mean, A, r, s2, idx)
            rec[name] = base_q + X @ beta
        beta, _ = _solve(pop_prec, adapter.mu, A, r, s2)
        rec['eb_population'] = base_q + X @ beta
        if zrows is not None:
            mu_c, cov_c = prior_from_z(adapter, zrows[zindex[pid]])
        else:
            mu_c, cov_c = adapter.mu.copy(), adapter.sigma.copy()
        prec_c = np.linalg.inv(cov_c)
        beta, _ = _solve(prec_c, mu_c, A, r, s2)
        rec['eb_conditional'] = base_q + X @ beta
        beta, _ = _solve(prec_c, mu_c, A, r, s2, idx=(0,))
        rec['eb_intercept'] = base_q + X @ beta
        rec['cond_number'] = float(np.linalg.cond(A))
        rows.append(rec)
    return pd.concat(rows, ignore_index=True)


def tune_scalar(adapter, train, k, query_after, grid, kind):
    """Choose one scalar hyper-parameter inside the training fold only.

    Training participants are scored exactly as test participants would be: the
    first ``k`` meals calibrate, the meals after ``query_after`` are scored. No
    test-fold information enters.
    """
    from . import stats
    train = train.sort_values(['participant_id', 'timestamp']).reset_index(drop=True)
    base_all = adapter.base.predict(train)
    design_all = adapter.design(train)
    s2 = adapter.s2 * adapter.noise
    d = design_all.shape[1]
    I = np.eye(d)
    target = train.target.to_numpy(float)
    best, best_score, trace = None, np.inf, []
    for value in grid:
        prec = I * (value / s2) if kind == 'ridge' else I / value ** 2
        pred, tgt, pids = [], [], []
        for pid, g in train.groupby('participant_id', sort=False):
            if len(g) <= query_after:
                continue
            pos = g.index.to_numpy()
            hp, qp = pos[:k], pos[query_after:]
            A, X = design_all[hp], design_all[qp]
            r = target[hp] - base_all[hp]
            beta, _ = _solve(prec, np.zeros(d), A, r, s2)
            pred.append(base_all[qp] + X @ beta)
            tgt.append(target[qp])
            pids.append(np.repeat(pid, len(qp)))
        frame = pd.DataFrame(dict(participant_id=np.concatenate(pids),
                                  target=np.concatenate(tgt),
                                  prediction=np.concatenate(pred)))
        score = stats.balanced(stats.person_losses(frame, 'prediction'), 'mse')
        trace.append(dict(kind=kind, value=float(value), train_rmse=float(score)))
        if score < best_score:
            best, best_score = value, score
    return float(best), float(best_score), trace
