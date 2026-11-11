"""Choosing the calibration meals is part of the method, not a property of the data.

A fixed query set is held out for every participant; the calibration meals are
then chosen from a pool of earlier meals by rules that see only the nutrient
composition of the candidate meals and the prior. Because composition is known
before a meal is eaten, every rule except the oracle can be deployed
prospectively.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols, design
from masked_operator import splits
from conditional_adapter import adapter_configs

SEED = 20261111
K = 3
POOLS = (5, 8)
CFG = adapter_configs()['masked_host_taxa_bw1.0_mean0.5']


def run(frame, tag, pool_size, randomise_seed=None):
    rows = []
    for fold, (a, b) in enumerate(splits(frame, 5)):
        tr = frame.iloc[a].reset_index(drop=True)
        te = frame.iloc[b].reset_index(drop=True)
        protocols.assert_fold_disjoint(tr, te)
        if randomise_seed is not None:
            tr = protocols.inject_random_order(tr, randomise_seed + fold)
            te = protocols.inject_random_order(te, randomise_seed + 100 + fold)
        base, ad = runner.fit_fold(tr)
        m = runner.configure(ad, CFG, tr)
        M_ref = design.reference_moment(m, tr, seed=SEED + fold)
        p = design.evaluate(m, te, pool_size, K, seed=SEED + fold, M_ref=M_ref)
        p['outer_fold'] = fold
        rows.append(p)
        print(f'  [{tag} pool={pool_size}] fold {fold} done', flush=True)
    return pd.concat(rows, ignore_index=True)


def report(P, tag, pool_size):
    tab, eff = [], []
    for name, g in P.groupby('selector', sort=False):
        per_draw = []
        for d, gg in g.groupby('draw', sort=False):
            L = {c: stats.person_losses(gg, c) for c in ('shared', 'intercept_only', 'full')}
            per_draw.append(dict(
                draw=d, full_rmse=stats.balanced(L['full'], 'mse'),
                intercept_rmse=stats.balanced(L['intercept_only'], 'mse'),
                shared_rmse=stats.balanced(L['shared'], 'mse'),
                channel_increment=stats.balanced(L['intercept_only'], 'mse') - stats.balanced(L['full'], 'mse')))
        D = pd.DataFrame(per_draw)
        base = g[g.draw.eq(0)]
        L = {c: stats.person_losses(base, c) for c in ('shared', 'intercept_only', 'full')}
        ci = stats.paired_bootstrap(L['full'], L['intercept_only'], 'mse', seed=SEED)
        tot = stats.paired_bootstrap(L['full'], L['shared'], 'mse', seed=SEED)
        cond = base.groupby('participant_id').cond_number.first()
        contraction = base.groupby('participant_id').apply(
            lambda x: 1 - float(x.posterior_sd_mac.iloc[0] / x.prior_sd_mac.iloc[0]),
            include_groups=False)
        tab.append(dict(
            cohort=tag, pool=pool_size, selector=name, draws=int(D.draw.nunique()),
            people=int(base.participant_id.nunique()), query_meals=int(len(base)),
            shared_rmse=float(D.shared_rmse.mean()),
            intercept_rmse=float(D.intercept_rmse.mean()),
            full_rmse=float(D.full_rmse.mean()),
            full_rmse_sd_over_draws=float(D.full_rmse.std(ddof=1)) if len(D) > 1 else 0.0,
            channel_increment_mean=float(D.channel_increment.mean()),
            channel_increment_lo=float(ci['diff_lo']), channel_increment_hi=float(ci['diff_hi']),
            channel_increment_pct=float(ci['relative_pct']),
            better=int(ci['better']), worse=int(ci['worse']), ties=int(ci['ties']),
            sign_p=float(ci['sign_p']),
            total_gain=float(tot['difference']), total_gain_lo=float(tot['diff_lo']),
            total_gain_hi=float(tot['diff_hi']),
            median_cond_number=float(cond.median()),
            pct_cond_over_100=float(100 * (cond > 100).mean()),
            median_log_det_information=float(base.groupby('participant_id').log_det_information.first().median()),
            median_mac_contraction_pct=float(100 * contraction.median())))
        eff.append(dict(cohort=tag, pool=pool_size, selector=name,
                        contrast='channel_increment', **ci))
    return pd.DataFrame(tab), pd.DataFrame(eff)


if __name__ == '__main__':
    env.sha256_manifest([env.cohort_file('cgmacros_functional.parquet'),
                         env.cohort_file('predict.parquet')], 't17')
    tabs, effs, keeps = [], [], []
    cg = env.load_cgmacros().sort_values(
        ['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
    for pool in POOLS:
        P = run(cg, 'CGMacros', pool)
        t, e = report(P, 'CGMacros', pool)
        tabs.append(t); effs.append(e)
        if pool == 5:
            keeps.append(P.assign(cohort='CGMacros', pool=pool))
    uk = env.load_predict('UK')
    P = run(uk, 'PREDICT_UK', 5, randomise_seed=SEED + 7000)
    t, e = report(P, 'PREDICT_UK', 5)
    tabs.append(t); effs.append(e)

    tab = pd.concat(tabs, ignore_index=True)
    eff = pd.concat(effs, ignore_index=True)
    tab.to_csv(env.RESULTS / 't17_design_selectors.csv', index=False)
    eff.to_csv(env.RESULTS / 't17_design_contrasts.csv', index=False)
    pd.concat(keeps, ignore_index=True).to_parquet(env.RESULTS / 't17_cgmacros_pool5.parquet')

    main = tab[(tab.cohort == 'CGMacros') & (tab.pool == 5)].set_index('selector')
    env.dump(dict(
        calibration_meals=K, pools=list(POOLS), seed=SEED, config=CFG,
        design=('the query set is fixed by the pool boundary, so every selector is scored on '
                'identical meals; only the calibration triple changes'),
        deployable_selectors=[s for s in design.SELECTORS if s != 'oracle'],
        oracle_note='oracle minimises realised query error and is an upper bound, not a method',
        cgmacros_pool5={s: dict(full_rmse=float(main.loc[s].full_rmse),
                                channel_increment=float(main.loc[s].channel_increment_mean),
                                channel_increment_lo=float(main.loc[s].channel_increment_lo),
                                channel_increment_hi=float(main.loc[s].channel_increment_hi),
                                median_cond_number=float(main.loc[s].median_cond_number),
                                pct_cond_over_100=float(main.loc[s].pct_cond_over_100),
                                median_mac_contraction_pct=float(main.loc[s].median_mac_contraction_pct))
                        for s in main.index}), 't17_calibration_design')
    pd.set_option('display.width', 260)
    print(); print(tab.round(4).to_string(index=False))
