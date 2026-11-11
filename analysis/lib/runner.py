"""Thin orchestration over model_snapshot's ConditionalAdapter.

Adds: reduced-coordinate updates (intercept-only / channels-only), a raw
personal-mean baseline, and explicit leakage assertions. The upstream model
code is never modified.
"""
import numpy as np, pandas as pd
from masked_operator import BaseModel, BASES, crossfit_base
from conditional_adapter import ConditionalAdapter
from . import protocols

SUBSETS = {
    'full': (0, 1, 2),
    'intercept_only': (0,),
    'channels_only': (1, 2),
    'intercept_plus_mac': (0, 1),
    'intercept_plus_lipid': (0, 2),
}


def fit_fold(train, base_key='hgb7', food='masked', dose='saturating'):
    base = BaseModel(**BASES[base_key]).fit(train)
    residual = train.target.to_numpy() - crossfit_base(train, BASES[base_key])
    adapter = ConditionalAdapter(base, food=food, dose=dose).fit(train, residual)
    return base, adapter


def configure(adapter, cfg, train, cache=None):
    """Clone the adapter with one candidate configuration.

    `cache` memoises the training trait matrix per trait family; recomputing it
    for every candidate dominates runtime on the large PREDICT cohort.
    """
    import copy
    m = copy.copy(adapter)
    for k, v in cfg.items():
        if k != 'food':
            setattr(m, k, v)
    trait = cfg['trait']
    if cache is None:
        m.traits = m.trait_matrix(train.drop_duplicates('participant_id'))
        return m
    if trait not in cache:
        probe = copy.copy(adapter)
        probe.trait = trait
        cache[trait] = probe.trait_matrix(train.drop_duplicates('participant_id'))
    m.traits = cache[trait]
    return m


def prior_from_z(adapter, z):
    """ConditionalAdapter.prior() for a precomputed trait row (vectorisation helper)."""
    if adapter.trait == 'uniform':
        return adapter.mu.copy(), adapter.sigma.copy()
    d = ((adapter.traits - z) ** 2).sum(1)
    a = -d / (2 * adapter.bandwidth ** 2)
    a -= a.max()
    w = np.exp(a); w /= w.sum()
    m = w @ adapter.betas
    diff = adapter.betas - m
    cov = np.einsum('n,ni,nj->ij', w, diff, diff) + np.einsum('n,nij->ij', w, adapter.covs)
    mean = (1 - adapter.mean_mix) * adapter.mu + adapter.mean_mix * m
    cov = (1 - adapter.mix) * adapter.sigma + adapter.mix * cov
    return mean, cov + np.eye(len(mean)) * 1e-6


def predict_subsets(adapter, test, k, query_after, subsets=SUBSETS):
    """Reduced-coordinate posterior updates on the same calibration meals.

    Base predictions and trait rows are computed once for the whole test frame;
    per-participant work is then pure linear algebra on 1-3 coefficients.
    """
    test = test.sort_values(['participant_id', 'timestamp']).reset_index(drop=True)
    base_all = adapter.base.predict(test)
    design_all = adapter.design(test)
    first = test.drop_duplicates('participant_id')
    zrows = (None if adapter.trait == 'uniform'
             else adapter.trait_matrix(first))
    zindex = {pid: i for i, pid in enumerate(first.participant_id.to_numpy())}
    s2 = adapter.s2 * adapter.noise
    out = []
    for pid, g in test.groupby('participant_id', sort=False):
        if len(g) <= query_after:
            continue
        pos = g.index.to_numpy()
        hist_pos, query_pos = pos[:k], pos[query_after:]
        assert not set(test.meal_event_id.to_numpy()[hist_pos]) & set(
            test.meal_event_id.to_numpy()[query_pos]), 'calibration/query leakage'
        z = None if zrows is None else zrows[zindex[pid]]
        mu, cov = prior_from_z(adapter, z)
        A = design_all[hist_pos]
        r = test.target.to_numpy()[hist_pos] - base_all[hist_pos]
        X = design_all[query_pos]
        base_q = base_all[query_pos]
        rec = test.iloc[query_pos][['meal_event_id', 'participant_id', 'family_group',
                                    'food_id', 'target']].copy()
        rec['history_k'] = k
        rec['shared'] = base_q
        rec['raw_personal_mean'] = base_q + float(r.mean())
        for name, idx in subsets.items():
            ix = np.array(idx)
            inv = np.linalg.inv(cov[np.ix_(ix, ix)])
            a_s = A[:, ix]
            post_cov = np.linalg.inv(inv + a_s.T @ a_s / s2)
            post = post_cov @ (inv @ mu[ix] + a_s.T @ r / s2)
            rec[name] = base_q + X[:, ix] @ post
            if name == 'full':
                rec['beta_intercept'], rec['beta_mac'], rec['beta_lipid'] = post
                rec['sd_intercept'], rec['sd_mac'], rec['sd_lipid'] = np.sqrt(np.diag(post_cov))
                rec['prior_sd_intercept'], rec['prior_sd_mac'], rec['prior_sd_lipid'] = np.sqrt(np.diag(cov))
                rec['prior_mac'], rec['prior_lipid'] = mu[1], mu[2]
                rec['phi_mac'], rec['phi_lipid'] = X[:, 1], X[:, 2]
                rec['cond_number'] = float(np.linalg.cond(A))
        out.append(rec)
    return pd.concat(out, ignore_index=True)
