"""Paired participant-bootstrap inference for participant-balanced errors."""
import numpy as np, pandas as pd
from math import comb


def person_losses(frame, pred_col, target_col='target', pid_col='participant_id'):
    e = frame[pred_col].to_numpy(float) - frame[target_col].to_numpy(float)
    g = pd.DataFrame({'pid': frame[pid_col].to_numpy(), 'e': e, 'e2': e ** 2})
    z = g.groupby('pid').agg(mse=('e2', 'mean'), bias=('e', 'mean'), n=('e', 'size'))
    z['profile_mse'] = np.maximum(0.0, z.mse - z.bias ** 2)
    return z


def balanced(losses, column='mse'):
    return float(np.sqrt(losses[column].mean()))


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
    better = int((a < b).sum())
    return dict(method=float(ma), reference=float(mb), difference=float(mb - ma),
                diff_lo=float(np.quantile(diff, .025)), diff_hi=float(np.quantile(diff, .975)),
                diff_se=float(diff.std(ddof=1)),
                relative_pct=float(100 * (mb - ma) / mb),
                rel_lo=float(np.quantile(rel, .025)), rel_hi=float(np.quantile(rel, .975)),
                participants=int(n), better=better,
                sign_p=sign_test(better, n))


def sign_test(better, total):
    """Exact two-sided binomial sign test against p=0.5."""
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
