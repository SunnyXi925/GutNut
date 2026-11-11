"""Adaptation mechanisms compared under one fixed nonlinear population predictor.

The published mixed-model comparison changes the population function class and
the adaptation rule at the same time. This experiment holds the population
predictor, the calibration meals and the query meals fixed and varies only the
rule that turns three calibration residuals into a personal correction, so the
increment attributable to the structured empirical-Bayes prior can be read off
directly against generic regularisation and against the zero-mean best linear
unbiased predictor written on the same nonlinear residual.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats, runner, protocols, baselines
from masked_operator import splits
from conditional_adapter import adapter_configs

SEED = 20261111
K = 3
CFG = adapter_configs()['masked_host_taxa_bw1.0_mean0.5']


def run_cohort(frame, tag, query_after, randomise_seed=None):
    rows, meta = [], []
    for fold, (a, b) in enumerate(splits(frame, 5)):
        tr = frame.iloc[a].reset_index(drop=True)
        te = frame.iloc[b].reset_index(drop=True)
        protocols.assert_fold_disjoint(tr, te)
        if randomise_seed is not None:
            tr = protocols.inject_random_order(tr, randomise_seed + fold)
            te = protocols.inject_random_order(te, randomise_seed + 100 + fold)
        base, ad = runner.fit_fold(tr)
        m = runner.configure(ad, CFG, tr)
        G_full = baselines.moment_G(m.betas, m.covs, about_zero=True)
        G_diag = np.diag(np.diag(G_full)) + np.eye(len(G_full)) * 1e-8
        alpha, a_score, a_trace = baselines.tune_scalar(
            m, tr, K, query_after, baselines.RIDGE_GRID, 'ridge')
        tau, t_score, t_trace = baselines.tune_scalar(
            m, tr, K, query_after, baselines.TAU_GRID, 'tau')
        p = baselines.predict_rules(m, te, K, query_after, alpha, tau, G_full, G_diag)
        p['outer_fold'] = fold
        rows.append(p)
        meta.append(dict(cohort=tag, fold=fold, ridge_alpha=alpha, ridge_train_rmse=a_score,
                         tau=tau, tau_train_rmse=t_score, s2=float(m.s2 * m.noise),
                         G_full_diag=';'.join('%.5f' % v for v in np.diag(G_full)),
                         G_offdiag_01=float(G_full[0, 1]), G_offdiag_02=float(G_full[0, 2]),
                         G_offdiag_12=float(G_full[1, 2]),
                         prior_mean=';'.join('%.5f' % v for v in m.mu),
                         train_people=int(tr.participant_id.nunique())))
        print(f'  [{tag}] fold {fold}: alpha={alpha} tau={tau}', flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(meta), pd.DataFrame(a_trace + t_trace)


def report(P, tag):
    L = {c: stats.person_losses(P, c) for c in baselines.RULES}
    tab = pd.DataFrame([dict(cohort=tag, rule=c, meals=len(P), people=len(L[c]),
                             rmse=stats.balanced(L[c], 'mse'),
                             profile_rmse=stats.balanced(L[c], 'profile_mse'))
                        for c in baselines.RULES]).sort_values('rmse')
    pairs = [
        ('eb_conditional_vs_blup_full', 'eb_conditional', 'blup_full'),
        ('eb_conditional_vs_blup_diag', 'eb_conditional', 'blup_diag'),
        ('eb_conditional_vs_ridge_head', 'eb_conditional', 'ridge_head'),
        ('eb_conditional_vs_bayes_last_layer', 'eb_conditional', 'bayes_last_layer'),
        ('eb_conditional_vs_eb_population', 'eb_conditional', 'eb_population'),
        ('eb_population_vs_blup_full', 'eb_population', 'blup_full'),
        ('eb_conditional_vs_eb_intercept', 'eb_conditional', 'eb_intercept'),
        ('blup_full_vs_eb_intercept', 'blup_full', 'eb_intercept'),
        ('ridge_head_vs_ols_head', 'ridge_head', 'ols_head'),
        ('eb_intercept_vs_shared', 'eb_intercept', 'shared'),
    ]
    eff = pd.DataFrame([dict(cohort=tag, contrast=n, method_name=x, reference_name=y,
                             **stats.paired_bootstrap(L[x], L[y], 'mse', seed=SEED))
                        for n, x, y in pairs])
    eff['mde_80'] = [stats.mde(s) for s in eff.diff_se]
    return tab, eff


if __name__ == '__main__':
    env.sha256_manifest([env.cohort_file('cgmacros_functional.parquet'),
                         env.cohort_file('predict.parquet')], 't16')
    tabs, effs, metas, traces, preds = [], [], [], [], []
    print('CGMacros')
    cg = env.load_cgmacros().sort_values(
        ['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
    P, M, TR = run_cohort(cg, 'CGMacros', query_after=5)
    t, e = report(P, 'CGMacros')
    tabs.append(t); effs.append(e); metas.append(M); traces.append(TR.assign(cohort='CGMacros'))
    preds.append(P.assign(cohort='CGMacros'))
    print('PREDICT UK')
    uk = env.load_predict('UK')
    P2, M2, TR2 = run_cohort(uk, 'PREDICT_UK', query_after=3, randomise_seed=SEED + 7000)
    t2, e2 = report(P2, 'PREDICT_UK')
    tabs.append(t2); effs.append(e2); metas.append(M2); traces.append(TR2.assign(cohort='PREDICT_UK'))

    tab = pd.concat(tabs, ignore_index=True)
    eff = pd.concat(effs, ignore_index=True)
    tab.to_csv(env.RESULTS / 't16_rule_models.csv', index=False)
    eff.to_csv(env.RESULTS / 't16_rule_contrasts.csv', index=False)
    pd.concat(metas, ignore_index=True).to_csv(env.RESULTS / 't16_fold_settings.csv', index=False)
    pd.concat(traces, ignore_index=True).to_csv(env.RESULTS / 't16_tuning_trace.csv', index=False)
    pd.concat(preds, ignore_index=True).to_parquet(env.RESULTS / 't16_cgmacros_predictions.parquet')

    key = eff[eff.contrast.eq('eb_conditional_vs_blup_full')].set_index('cohort')
    env.dump(dict(
        design=('one nonlinear population predictor, identical calibration and query meals, '
                'nine adaptation rules'),
        calibration_meals=K, seed=SEED, config=CFG,
        tuning=('ridge penalty and Gaussian prior scale are selected inside each training fold '
                'by scoring training participants under the deployment protocol'),
        headline={c: dict(point=float(key.loc[c].difference), lo=float(key.loc[c].diff_lo),
                          hi=float(key.loc[c].diff_hi), relative_pct=float(key.loc[c].relative_pct),
                          better=int(key.loc[c].better), worse=int(key.loc[c].worse),
                          ties=int(key.loc[c].ties), sign_p=float(key.loc[c].sign_p),
                          mde_80=float(key.loc[c].mde_80)) for c in key.index},
        interpretation=('eb_conditional minus blup_full isolates the contribution of the fitted '
                        'prior mean and of side-information conditioning, because both rules use '
                        'the same nonlinear population predictor, the same design matrix and the '
                        'same residual noise estimate.')), 't16_fair_baselines')
    pd.set_option('display.width', 250)
    print(); print(tab.round(4).to_string(index=False))
    print(); print(eff[['cohort', 'contrast', 'method', 'reference', 'difference', 'diff_lo',
                        'diff_hi', 'relative_pct', 'better', 'worse', 'ties', 'sign_p',
                        'mde_80']].round(5).to_string(index=False))
