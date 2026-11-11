"""Evaluation protocols for few-shot personal calibration."""
import numpy as np, pandas as pd


def inject_random_order(frame, seed, pid_col='participant_id'):
    """Give每位参与者一个固定种子下的随机餐次顺序,并合成 timestamp 供 predict_history 使用。

    PREDICT 的标准化餐食没有时间序,随机顺序避免「前三餐」带来的顺序混杂;
    顺序本身存盘以便复算。
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
