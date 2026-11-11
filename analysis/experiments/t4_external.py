"""External validation: PREDICT UK few-shot replication and frozen UK -> US transport.

Primary estimand: intercept_only -> full (participant-balanced RMSE).
Prior arms are non-overlapping; the microbial arm contains no host information.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols, arms
from masked_operator import splits
from conditional_adapter import adapter_configs

SEED = 20261108
DRAWS = 5
K = 3
QUERY_AFTER = 3
ARMS = ['population', 'host', 'micro', 'host_micro']
CFGS = {k: v for k, v in adapter_configs().items() if v['food'] == 'masked'}
LOG = env.RESULTS / 't4_external.log'


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(msg + '\n')


def evaluate(frame, cols):
    L = {c: stats.person_losses(frame, c) for c in cols}
    return L


def inner_select(train, draw_seed):
    """Select one config per arm using 3 inner family-grouped folds."""
    rows = []
    for i, (a, b) in enumerate(splits(train, 3)):
        tr, te = train.iloc[a].reset_index(drop=True), train.iloc[b].reset_index(drop=True)
        protocols.assert_fold_disjoint(tr, te)
        tr = protocols.inject_random_order(tr, draw_seed + 100 + i)
        te = protocols.inject_random_order(te, draw_seed + 200 + i)
        base, ad = runner.fit_fold(tr)
        cache = {}
        for name, cfg in CFGS.items():
            m = runner.configure(ad, cfg, tr, cache)
            p = runner.predict_subsets(m, te, K, QUERY_AFTER, {'full': (0, 1, 2)})
            L = stats.person_losses(p, 'full')
            rows.append(dict(inner_fold=i, config=name, rmse=stats.balanced(L, 'mse'),
                             profile_rmse=stats.balanced(L, 'profile_mse')))
    tab = pd.DataFrame(rows).groupby('config')[['rmse', 'profile_rmse']].mean().reset_index()
    return {arm: arms.select(tab, CFGS, arm) for arm in ARMS}, tab


def run_cohort(frame, tag, selections=None, train_override=None):
    """If train_override is given, the model is fitted there and frozen for `frame` (transport)."""
    t0 = time.time()
    sel_record, preds = [], []
    if train_override is not None:
        tr_raw = train_override
        folds = [(None, None)]
    else:
        folds = list(splits(frame, 5))
    for fold, (a, b) in enumerate(folds):
        if train_override is not None:
            tr, te_raw = tr_raw, frame
            assert not set(tr.participant_id) & set(te_raw.participant_id), 'transport leakage'
        else:
            tr, te_raw = frame.iloc[a].reset_index(drop=True), frame.iloc[b].reset_index(drop=True)
            protocols.assert_fold_disjoint(tr, te_raw)
        tr_o = protocols.inject_random_order(tr, SEED + fold)
        sel = selections[fold] if selections else inner_select(tr, SEED + 1000 * fold)[0]
        base, ad = runner.fit_fold(tr_o)
        cache = {}
        for arm in ARMS:
            cfg = CFGS[sel[arm]['config']]
            assert arms.arm_of(cfg) == arm
            m = runner.configure(ad, cfg, tr_o, cache)
            for draw in range(DRAWS):
                te = protocols.inject_random_order(te_raw, SEED + 7000 + 13 * draw)
                p = runner.predict_subsets(m, te, K, QUERY_AFTER)
                p['arm'] = arm; p['draw'] = draw; p['outer_fold'] = fold
                p['config'] = sel[arm]['config']
                preds.append(p)
            sel_record.append(dict(fold=fold, arm=arm, config=sel[arm]['config'],
                                   inner_rmse=sel[arm]['rmse']))
        log(f'[{tag}] fold {fold} done t={time.time()-t0:.0f}s')
    return pd.concat(preds, ignore_index=True), pd.DataFrame(sel_record)


COLS = ['shared', 'raw_personal_mean', 'intercept_only', 'intercept_plus_mac',
        'intercept_plus_lipid', 'channels_only', 'full']
CONTRASTS = [('channel_increment_PRIMARY', 'full', 'intercept_only'),
             ('level_correction', 'intercept_only', 'shared'),
             ('total_gain_context', 'full', 'shared'),
             ('full_vs_raw_mean', 'full', 'raw_personal_mean'),
             ('shrinkage_vs_raw_mean', 'intercept_only', 'raw_personal_mean'),
             ('mac_contribution', 'full', 'intercept_plus_lipid'),
             ('lipid_contribution', 'full', 'intercept_plus_mac')]


def summarise(P, tag):
    out, eff = [], []
    for arm in sorted(P.arm.unique()):
        for draw in sorted(P.draw.unique()):
            d = P[(P.arm == arm) & (P.draw == draw)]
            L = evaluate(d, COLS)
            for c in COLS:
                out.append(dict(cohort=tag, arm=arm, draw=draw, model=c, meals=len(d),
                                people=len(L[c]), rmse=stats.balanced(L[c], 'mse'),
                                profile_rmse=stats.balanced(L[c], 'profile_mse')))
            for name, m, r in CONTRASTS:
                for metric, col in (('rmse', 'mse'), ('profile_rmse', 'profile_mse')):
                    s = stats.paired_bootstrap(L[m], L[r], col, B=2000, seed=SEED + draw)
                    eff.append(dict(cohort=tag, arm=arm, draw=draw, contrast=name,
                                    method_name=m, reference_name=r, metric=metric, **s))
    return pd.DataFrame(out), pd.DataFrame(eff)


if __name__ == '__main__':
    env.sha256_manifest([env.cohort_file('predict.parquet')], 't4')
    log('=== PREDICT UK internal few-shot replication ===')
    uk = env.load_predict('UK')
    log('UK', uk.shape, 'people', uk.participant_id.nunique(), 'families', uk.family_group.nunique())
    Puk, sel_uk = run_cohort(uk, 'UK')
    Puk.to_parquet(env.RESULTS / 't4_uk_predictions.parquet', index=False)
    sel_uk.to_csv(env.RESULTS / 't4_uk_selected.csv', index=False)
    suk, euk = summarise(Puk, 'PREDICT_UK')

    log('=== frozen UK -> US transport ===')
    us = env.load_predict('US')
    log('US', us.shape, 'people', us.participant_id.nunique())
    sel_by_fold = {0: {r.arm: dict(config=r.config, rmse=r.inner_rmse)
                       for r in sel_uk[sel_uk.fold == 0].itertuples()}}
    ukfull = protocols.inject_random_order(uk, SEED)
    Pus, sel_us = run_cohort(us, 'US', selections=sel_by_fold, train_override=ukfull)
    Pus.to_parquet(env.RESULTS / 't4_us_predictions.parquet', index=False)
    sus, eus = summarise(Pus, 'PREDICT_US_transport')

    pd.concat([suk, sus]).to_csv(env.RESULTS / 't4_external_summary.csv', index=False)
    pd.concat([euk, eus]).to_csv(env.RESULTS / 't4_external_contrasts.csv', index=False)

    def headline(e, tag):
        d = e[(e.contrast == 'channel_increment_PRIMARY') & (e.metric == 'rmse')]
        return dict(cohort=tag, point=float(d.difference.mean()),
                    lo=float(d.diff_lo.mean()), hi=float(d.diff_hi.mean()),
                    relative_pct=float(d.relative_pct.mean()),
                    rel_lo=float(d.rel_lo.mean()), rel_hi=float(d.rel_hi.mean()),
                    draws_with_positive_lower_bound=int((d.diff_lo > 0).sum()),
                    draws=int(len(d)))
    env.dump(dict(primary_uk=[headline(euk[euk.arm == a], f'UK/{a}') for a in ARMS],
                  primary_us=[headline(eus[eus.arm == a], f'US/{a}') for a in ARMS],
                  seed=SEED, draws=DRAWS, k=K, query_after=QUERY_AFTER), 't4_external')
    pd.set_option('display.width', 240)
    log(suk.groupby(['arm', 'model'])[['rmse', 'profile_rmse']].mean().to_string())
    log(euk[(euk.contrast == 'channel_increment_PRIMARY') & (euk.metric == 'rmse')]
        .groupby('arm')[['difference', 'diff_lo', 'diff_hi', 'relative_pct', 'better', 'sign_p']].mean().to_string())
    log('t4 complete')
