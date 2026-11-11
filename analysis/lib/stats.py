"""Paired participant-bootstrap inference for participant-balanced errors.

Numerical-tie policy
--------------------
Every equality-based statistic in this project (sign tests, better/worse counts)
uses one project-wide tolerance, ``COMPARISON_TOLERANCE``, applied to the paired
per-participant loss difference on its original scale.  A difference whose
absolute value does not exceed the tolerance is a tie and is excluded from the
sign test, following the usual treatment of ties in the exact sign test.

The default was fixed before the recount and is not tuned to any outcome.  Its
justification is numerical: per-participant mean squared errors are of order
1e0 and are accumulated over tens of meals in double precision, so round-off is
of order 1e-15; any difference below 1e-10 is therefore round-off rather than a
difference between two adaptation rules, while 1e-10 remains ten orders of
magnitude below the smallest scientifically meaningful difference.  Sensitivity
over 1e-12, 1e-10 and 1e-8 is reported alongside every recounted test.
"""
import numpy as np, pandas as pd
from math import comb

COMPARISON_TOLERANCE = 1e-10
TOLERANCE_GRID = (1e-12, 1e-10, 1e-8)


def person_losses(frame, pred_col, target_col='target', pid_col='participant_id'):
    e = frame[pred_col].to_numpy(float) - frame[target_col].to_numpy(float)
    g = pd.DataFrame({'pid': frame[pid_col].to_numpy(), 'e': e, 'e2': e ** 2})
    z = g.groupby('pid').agg(mse=('e2', 'mean'), bias=('e', 'mean'), n=('e', 'size'))
    z['profile_mse'] = np.maximum(0.0, z.mse - z.bias ** 2)
    return z


def balanced(losses, column='mse'):
    return float(np.sqrt(losses[column].mean()))


def paired_delta(method, reference, column='mse'):
    """Per-participant loss difference, positive when the method has lower loss."""
    assert reference.index.equals(method.index), 'participant sets must match'
    return reference[column].to_numpy(float) - method[column].to_numpy(float)


def tie_aware_counts(delta, tol=COMPARISON_TOLERANCE):
    """Better / worse / tie counts under one explicit absolute tolerance."""
    delta = np.asarray(delta, float)
    better = int((delta > tol).sum())
    worse = int((delta < -tol).sum())
    ties = int(len(delta) - better - worse)
    return dict(better=better, worse=worse, ties=ties, n_compared=better + worse,
                tolerance=float(tol), sign_p=sign_test(better, better + worse))


def tie_sensitivity(delta, tols=TOLERANCE_GRID):
    """Sign-test sensitivity across the pre-specified tolerance grid."""
    return [dict(**tie_aware_counts(delta, t)) for t in tols]


def paired_bootstrap(method, reference, column='mse', B=2000, seed=20261108):
    """method/reference: person_losses frames aligned on the same participants."""
    pids = method.index
    assert reference.index.equals(pids), 'participant sets must match'
    a = method[column].to_numpy(); b = reference[column].to_numpy()
    n = len(pids)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, n, size=(B, n))
    ma, mb = np.sqrt(a.mean()), np.sqrt(b.mean())
    ba, bb = np.sqrt(a[ix].mean(1)), np.sqrt(b[ix].mean(1))
    diff = bb - ba
    rel = 100 * (bb - ba) / bb
    delta = b - a
    counts = tie_aware_counts(delta, COMPARISON_TOLERANCE)
    nz = np.abs(delta)[np.abs(delta) > 0]
    return dict(method=float(ma), reference=float(mb), difference=float(mb - ma),
                diff_lo=float(np.quantile(diff, .025)), diff_hi=float(np.quantile(diff, .975)),
                diff_se=float(diff.std(ddof=1)),
                relative_pct=float(100 * (mb - ma) / mb),
                rel_lo=float(np.quantile(rel, .025)), rel_hi=float(np.quantile(rel, .975)),
                participants=int(n),
                better=counts['better'], worse=counts['worse'], ties=counts['ties'],
                n_compared=counts['n_compared'], tolerance=counts['tolerance'],
                sign_p=counts['sign_p'],
                min_abs_nonzero_delta=float(nz.min()) if len(nz) else 0.0,
                sign_p_tol_1e12=sign_test(*_bw(delta, 1e-12)),
                sign_p_tol_1e10=sign_test(*_bw(delta, 1e-10)),
                sign_p_tol_1e08=sign_test(*_bw(delta, 1e-8)))


def _bw(delta, tol):
    delta = np.asarray(delta, float)
    better = int((delta > tol).sum()); worse = int((delta < -tol).sum())
    return better, better + worse


def sign_test(better, total):
    """Exact two-sided binomial sign test against p=0.5 among non-tied pairs."""
    if total <= 0:
        return float('nan')
    k = max(better, total - better)
    tail = sum(comb(total, i) for i in range(k, total + 1)) / 2 ** total
    return float(min(1.0, 2 * tail))


def mde(se, power=0.8, alpha=0.05):
    """Minimum detectable effect for a two-sided test at given power."""
    from scipy.stats import norm
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def equivalence(point, lo, hi, margin):
    """TOST-style read-off from a two-sided interval."""
    return dict(margin=float(margin), point=float(point), lo=float(lo), hi=float(hi),
                equivalent=bool(lo > -margin and hi < margin),
                inconclusive=bool(not (lo > -margin and hi < margin) and lo < margin and hi > -margin))
