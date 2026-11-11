"""Evaluation protocols for few-shot personal calibration."""
import numpy as np, pandas as pd


def inject_random_order(frame, seed, pid_col='participant_id'):
    """Give every participant a seeded random meal order and a synthetic timestamp.

    The standardized meals carry no time order, so a random order removes any
    confound that taking the earliest three meals would introduce. The order is
    stored with the predictions so that the run can be reproduced exactly.
    """
    f = frame.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    order = np.empty(len(f), dtype=int)
    for _, g in f.groupby(pid_col, sort=True):
        pos = np.array(g.index.to_numpy(), copy=True)
        rng.shuffle(pos)
        order[pos] = np.arange(len(pos))
    f['order'] = order
    f['timestamp'] = pd.Timestamp('2020-01-01') + pd.to_timedelta(f['order'].astype(int), unit='D')
    return f


def assert_no_overlap(hist, query):
    assert not set(hist.meal_event_id) & set(query.meal_event_id), 'calibration/query leakage'


def assert_fold_disjoint(train, test, group_col='family_group'):
    assert not set(train.participant_id) & set(test.participant_id), 'participant leakage'
    assert not set(train[group_col]) & set(test[group_col]), f'{group_col} leakage'
