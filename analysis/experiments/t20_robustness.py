"""Does the primary estimand depend on choices that were not part of the claim?

Four families of decisions are varied one at a time while everything else is
held at the reported configuration, so each row is an answer to "would the
nutrient-channel increment have been reported differently under this choice?":

fold assignment   five different cross-validation seeds;
influence         leave-one-participant-out on the primary contrast, plus
                  trimmed and winsorised versions of the participant losses;
sensor            the paired second continuous glucose monitor as the outcome;
calibration set   the first three meals, three randomly drawn meals, and the
                  information-based designs of t17 are compared through their
                  effect on the same estimand.

Every row carries the tie-aware counts and the tolerance used, so the recount
policy is visible in the sensitivity analysis itself.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols
from masked_operator import splits, BASES
from conditional_adapter import adapter_configs

SEED = 20261111
K = 3
QUERY_AFTER = 5
CFG = adapter_configs()['masked_host_taxa_bw1.0_mean0.5']
FOLD_SEEDS = (0, 1, 2, 3, 4)


def oof(frame, fold_seed=None, target_col='target'):
    f = frame.copy()
    if target_col != 'target':
        f['target'] = f[target_col]
    f = f[f.target.notna()].sort_values(
        ['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
    rows = []
    parts = splits(f, 5) if fold_seed is None else shuffled_splits(f, 5, fold_seed)
    for fold, (a, b) in enumerate(parts):
        tr, te = f.iloc[a].reset_index(drop=True), f.iloc[b].reset_index(drop=True)
        protocols.assert_fold_disjoint(tr, te)
        base, ad = runner.fit_fold(tr)
        m = runner.configure(ad, CFG, tr)
        p = runner.predict_subsets(m, te, K, QUERY_AFTER)
        p['outer_fold'] = fold
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


def shuffled_splits(frame, n, seed):
    """Participant-grouped folds under an explicit seed, families kept together."""
    groups = frame.family_group.to_numpy() if 'family_group' in frame else frame.participant_id.to_numpy()
    uniq = pd.unique(pd.Series(groups))
    rng = np.random.default_rng(10_000 + seed)
    assign = {g: i % n for i, g in enumerate(rng.permutation(uniq))}
    lab = np.array([assign[g] for g in groups])
    return [(np.flatnonzero(lab != i), np.flatnonzero(lab == i)) for i in range(n)]


def primary(P, label, extra=None):
    L = {c: stats.person_losses(P, c) for c in ('shared', 'intercept_only', 'full')}
    d = stats.paired_bootstrap(L['full'], L['intercept_only'], 'mse', seed=SEED)
    row = dict(analysis=label, people=len(L['full']), query_meals=len(P),
               intercept_rmse=stats.balanced(L['intercept_only'], 'mse'),
               full_rmse=stats.balanced(L['full'], 'mse'),
               shared_rmse=stats.balanced(L['shared'], 'mse'), **d)
    row['mde_80'] = stats.mde(d['diff_se'])
    if extra:
        row.update(extra)
    return row


if __name__ == '__main__':
    env.sha256_manifest([env.cohort_file('cgmacros_functional.parquet')], 't20')
    cg = env.load_cgmacros()
    rows, influence = [], []

    print('fold assignment', flush=True)
    base_P = oof(cg)
    rows.append(primary(base_P, 'reported_fold_assignment'))
    for s in FOLD_SEEDS:
        rows.append(primary(oof(cg, fold_seed=s), f'fold_seed_{s}'))
        print('  seed', s, 'done', flush=True)

    print('influence and robust summaries', flush=True)
    L_full = stats.person_losses(base_P, 'full')
    L_int = stats.person_losses(base_P, 'intercept_only')
    delta = stats.paired_delta(L_full, L_int, 'mse')
    pids = L_full.index.to_numpy()
    full_point = stats.balanced(L_int, 'mse') - stats.balanced(L_full, 'mse')
    for i, pid in enumerate(pids):
        keep = np.ones(len(pids), bool); keep[i] = False
        pt = (np.sqrt(L_int.mse.to_numpy()[keep].mean())
              - np.sqrt(L_full.mse.to_numpy()[keep].mean()))
        influence.append(dict(participant_id=pid, delta_mse=float(delta[i]),
                              estimand_without_participant=float(pt),
                              change=float(pt - full_point)))
    INF = pd.DataFrame(influence).sort_values('change')
    INF.to_csv(env.RESULTS / 't20_influence.csv', index=False)

    for name, frac in (('trimmed_10pct', 0.10), ('trimmed_20pct', 0.20)):
        lo, hi = np.quantile(L_full.mse, frac / 2), np.quantile(L_full.mse, 1 - frac / 2)
        keep = (L_full.mse >= lo) & (L_full.mse <= hi)
        sub = base_P[base_P.participant_id.isin(L_full.index[keep])]
        rows.append(primary(sub, name, dict(dropped_participants=int((~keep).sum()))))
    w = base_P.copy()
    for c in ('full', 'intercept_only', 'shared'):
        e = w[c] - w.target
        cap = np.quantile(np.abs(e), 0.99)
        w[c] = w.target + np.clip(e, -cap, cap)
    rows.append(primary(w, 'winsorised_residuals_99pct'))

    print('second sensor', flush=True)
    aux = cg[cg.target_aux.notna()]
    rows.append(primary(oof(cg, target_col='target_aux'), 'second_sensor_dexcom',
                        dict(dropped_participants=int(cg.participant_id.nunique()
                                                      - aux.participant_id.nunique()))))

    print('calibration-set composition', flush=True)
    for kk in (1, 2, 3, 5):
        P = base_P if kk == K else None
        if P is None:
            rowsk = []
            for fold, (a, b) in enumerate(splits(cg.sort_values(
                    ['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True), 5)):
                f = cg.sort_values(['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
                tr, te = f.iloc[a].reset_index(drop=True), f.iloc[b].reset_index(drop=True)
                base, ad = runner.fit_fold(tr)
                m = runner.configure(ad, CFG, tr)
                rowsk.append(runner.predict_subsets(m, te, kk, QUERY_AFTER))
            P = pd.concat(rowsk, ignore_index=True)
        rows.append(primary(P, f'calibration_meals_{kk}'))
        print('  k =', kk, 'done', flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(env.RESULTS / 't20_robustness.csv', index=False)
    rep = T[T.analysis.eq('reported_fold_assignment')].iloc[0]
    env.dump(dict(
        reported=dict(point=float(rep.difference), lo=float(rep.diff_lo), hi=float(rep.diff_hi),
                      relative_pct=float(rep.relative_pct), better=int(rep.better),
                      worse=int(rep.worse), ties=int(rep.ties), sign_p=float(rep.sign_p)),
        tolerance=stats.COMPARISON_TOLERANCE,
        fold_seed_range=[float(T[T.analysis.str.startswith('fold_seed')].difference.min()),
                         float(T[T.analysis.str.startswith('fold_seed')].difference.max())],
        fold_seed_all_positive=bool((T[T.analysis.str.startswith('fold_seed')].difference > 0).all()),
        fold_seed_lower_bound_positive=int((T[T.analysis.str.startswith('fold_seed')].diff_lo > 0).sum()),
        most_influential_participant=dict(
            participant_id=str(INF.iloc[0].participant_id),
            estimand_without_them=float(INF.iloc[0].estimand_without_participant),
            change=float(INF.iloc[0].change)),
        leave_one_out_range=[float(INF.estimand_without_participant.min()),
                             float(INF.estimand_without_participant.max())],
        leave_one_out_all_positive=bool((INF.estimand_without_participant > 0).all()),
        rows=T.to_dict(orient='records')), 't20_robustness')
    pd.set_option('display.width', 280)
    print(); print(T[['analysis', 'people', 'query_meals', 'intercept_rmse', 'full_rmse',
                      'difference', 'diff_lo', 'diff_hi', 'relative_pct', 'better', 'worse',
                      'ties', 'sign_p', 'mde_80']].round(5).to_string(index=False))
    print('\nleave-one-participant-out range of the estimand: %.4f to %.4f'
          % (INF.estimand_without_participant.min(), INF.estimand_without_participant.max()))
