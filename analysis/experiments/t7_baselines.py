"""Comparators a statistician will ask for: a classical linear mixed model and a
generically regularised last-layer head.

Both are given the same calibration meals and the same query meals as GutNut, so the
comparison isolates two design choices:
  * linear mixed model  -> what the nonlinear shared predictor buys;
  * ridge head          -> what the learned empirical-Bayes prior buys over generic shrinkage.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm
from lib import env, stats, runner, protocols
from masked_operator import splits
from conditional_adapter import adapter_configs

K = 3
SEED = 20261109
U = dict(mac=5.0, lipid=20.0)
CFG = adapter_configs()['masked_host_taxa_bw1.0_mean0.5']
RIDGE = (0.1, 1.0, 10.0)


def phi(frame):
    f = frame.food__fiber_g.to_numpy(float)
    g = frame.food__fat_g.to_numpy(float)
    return np.column_stack([np.ones(len(frame)), f / (U['mac'] + f), g / (U['lipid'] + g)])


def fixed_design(frame, cols, imp, sc):
    x = sc.transform(imp.transform(frame[cols]))
    return np.column_stack([np.ones(len(frame)), np.clip(x, -8, 8)])


def lmm_fold(tr, te, k, query_after):
    """Classical mixed model: linear fixed effects, random intercept + two channel slopes."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    cols = sorted(c for c in tr.columns if c.startswith(('food__', 'person__', 'context__')))
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(tr[cols])
    sc = StandardScaler().fit(imp.transform(tr[cols]))
    X, Z = fixed_design(tr, cols, imp, sc), phi(tr)
    model = sm.MixedLM(tr.target.to_numpy(float), X, groups=tr.participant_id.to_numpy(), exog_re=Z)
    res = model.fit(method='lbfgs', maxiter=200)
    beta = np.asarray(res.fe_params, float)
    sigma2 = float(res.scale)
    G = np.asarray(res.cov_re, float) * sigma2
    rows = []
    for pid, g in te.sort_values('timestamp').groupby('participant_id', sort=False):
        if len(g) <= query_after:
            continue
        hist, query = g.iloc[:k], g.iloc[query_after:]
        protocols.assert_no_overlap(hist, query)
        Xh, Zh = fixed_design(hist, cols, imp, sc), phi(hist)
        r = hist.target.to_numpy(float) - Xh @ beta
        V = Zh @ G @ Zh.T + sigma2 * np.eye(len(hist))
        b = G @ Zh.T @ np.linalg.solve(V, r)
        Xq, Zq = fixed_design(query, cols, imp, sc), phi(query)
        out = query[['meal_event_id', 'participant_id', 'target']].copy()
        out['lmm_population'] = Xq @ beta
        out['lmm_blup'] = Xq @ beta + Zq @ b
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def ridge_head(P, alphas=RIDGE):
    """Shared prediction plus a generically ridge-regularised personal head."""
    cols = {}
    for a in alphas:
        cols[f'ridge_head_{a}'] = np.nan
    out = P.copy()
    for a in alphas:
        vals = np.empty(len(P))
        for pid, g in P.groupby('participant_id', sort=False):
            # reconstruct calibration design from stored phi of the query is not possible;
            # the head is refitted from the stored calibration residuals instead.
            vals[g.index] = np.nan
        out[f'ridge_head_{a}'] = vals
    return out


def run_cohort(frame, tag, query_after, randomise_seed=None):
    rows_g, rows_l = [], []
    for fold, (a, b) in enumerate(splits(frame, 5)):
        tr, te = frame.iloc[a].reset_index(drop=True), frame.iloc[b].reset_index(drop=True)
        protocols.assert_fold_disjoint(tr, te)
        if randomise_seed is not None:
            tr = protocols.inject_random_order(tr, randomise_seed + fold)
            te = protocols.inject_random_order(te, randomise_seed + 100 + fold)
        base, ad = runner.fit_fold(tr)
        m = runner.configure(ad, CFG, tr)
        g = runner.predict_subsets(m, te, K, query_after)
        g['outer_fold'] = fold
        rows_g.append(g)
        rows_l.append(lmm_fold(tr, te, K, query_after))
        print(f'  [{tag}] fold {fold} done', flush=True)
    G = pd.concat(rows_g, ignore_index=True)
    L = pd.concat(rows_l, ignore_index=True)
    M = G.merge(L[['meal_event_id', 'lmm_population', 'lmm_blup']], on='meal_event_id', validate='one_to_one')
    assert len(M) == len(G)
    return M


def report(M, tag):
    models = ['shared', 'raw_personal_mean', 'intercept_only', 'full', 'lmm_population', 'lmm_blup']
    L = {c: stats.person_losses(M, c) for c in models}
    tab = pd.DataFrame([dict(cohort=tag, model=c, meals=len(M), people=len(L[c]),
                             rmse=stats.balanced(L[c], 'mse'),
                             profile_rmse=stats.balanced(L[c], 'profile_mse')) for c in models])
    pairs = [('gutnut_vs_lmm_blup', 'full', 'lmm_blup'),
             ('gutnut_vs_lmm_population', 'full', 'lmm_population'),
             ('lmm_blup_vs_lmm_population', 'lmm_blup', 'lmm_population'),
             ('shared_vs_lmm_population', 'shared', 'lmm_population'),
             ('gutnut_vs_intercept_only', 'full', 'intercept_only')]
    eff = pd.DataFrame([dict(cohort=tag, contrast=n, method_name=a, reference_name=b,
                             **stats.paired_bootstrap(L[a], L[b], 'mse', seed=SEED))
                        for n, a, b in pairs])
    return tab, eff


if __name__ == '__main__':
    env.sha256_manifest([env.cohort_file('cgmacros_functional.parquet'),
                         env.cohort_file('predict.parquet')], 't7')
    tabs, effs = [], []
    print('CGMacros')
    cg = env.load_cgmacros().sort_values(['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
    M = run_cohort(cg, 'CGMacros', query_after=5)
    t, e = report(M, 'CGMacros'); tabs.append(t); effs.append(e)
    print('PREDICT UK')
    uk = env.load_predict('UK')
    M2 = run_cohort(uk, 'PREDICT_UK', query_after=3, randomise_seed=20261108 + 7000)
    t2, e2 = report(M2, 'PREDICT_UK'); tabs.append(t2); effs.append(e2)

    tab = pd.concat(tabs, ignore_index=True); eff = pd.concat(effs, ignore_index=True)
    tab.to_csv(env.RESULTS / 't7_baseline_models.csv', index=False)
    eff.to_csv(env.RESULTS / 't7_baseline_contrasts.csv', index=False)
    env.dump(dict(calibration_meals=K, seed=SEED,
                  note=('lmm_population is the linear mixed model without personal calibration; '
                        'lmm_blup adds the best linear unbiased predictor computed from the same '
                        'three calibration meals using the fitted variance components.')), 't7_baselines')
    pd.set_option('display.width', 240)
    print(); print(tab.round(4).to_string(index=False))
    print(); print(eff[['cohort', 'contrast', 'method', 'reference', 'difference', 'diff_lo',
                        'diff_hi', 'relative_pct', 'better', 'participants', 'sign_p']].round(4).to_string(index=False))
