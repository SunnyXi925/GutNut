"""Three-layer decomposition of the calibration gain, with a raw personal-mean baseline.

New primary estimand: intercept-only -> full (the nutrient-channel increment).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats
from masked_operator import BaseModel, BASES, splits

SEED = 20261108
f = env.load_cgmacros()
env.sha256_manifest([env.SOURCE / 'module_predictions.csv',
                     env.SOURCE / 'CG_future_predictions.csv',
                     env.cohort_file('cgmacros_functional.parquet')], 't1')

# --- raw personal mean: shared prediction + arithmetic mean residual of first k meals ---
f = f.sort_values(['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
rows = []
for fold, (a, b) in enumerate(splits(f, 5)):
    tr, te = f.iloc[a], f.iloc[b]
    base = BaseModel(**BASES['hgb7']).fit(tr)
    te = te.copy()
    te['base_pred'] = base.predict(te)
    for pid, g in te.groupby('participant_id', sort=False):
        g = g.sort_values('timestamp')
        assert len(g) > 5
        for k in (1, 3, 5):
            hist, query = g.iloc[:k], g.iloc[5:]
            assert not set(hist.meal_event_id) & set(query.meal_event_id)
            shift = float((hist.target - hist.base_pred).mean())
            q = query[['meal_event_id', 'participant_id', 'target']].copy()
            q['prediction'] = query.base_pred + shift
            q['history_k'] = k
            q['outer_fold'] = fold
            rows.append(q)
raw = pd.concat(rows, ignore_index=True)
assert len(raw[raw.history_k.eq(3)]) == 902

# --- saved comparators ---
mod = pd.read_csv(env.SOURCE / 'module_predictions.csv')
cg = pd.read_csv(env.SOURCE / 'CG_future_predictions.csv')
shared = cg[cg.model.eq('same_base_without_history') & cg.history_k.eq(0)]
pick = lambda name: mod[mod.model.eq(name)]

models = {
    'shared': shared,
    'raw_personal_mean_k3': raw[raw.history_k.eq(3)],
    'intercept_only': pick('offset_only'),
    'intercept_plus_mac': pick('offset_fibre'),
    'intercept_plus_lipid': pick('offset_fat'),
    'channels_without_intercept': pick('nutrients_only'),
    'full': pick('full'),
}
losses = {k: stats.person_losses(v, 'prediction') for k, v in models.items()}
for k, v in losses.items():
    assert len(v) == 44, (k, len(v))

table = pd.DataFrame([
    dict(model=k, meals=len(models[k]), people=len(losses[k]),
         rmse=stats.balanced(losses[k], 'mse'),
         profile_rmse=stats.balanced(losses[k], 'profile_mse'))
    for k in models])
table = table.sort_values('rmse').reset_index(drop=True)

CONTRASTS = [
    ('level_correction', 'intercept_only', 'shared'),
    ('channel_increment_PRIMARY', 'full', 'intercept_only'),
    ('total_gain_context', 'full', 'shared'),
    ('shrinkage_vs_raw_mean', 'intercept_only', 'raw_personal_mean_k3'),
    ('full_vs_raw_mean', 'full', 'raw_personal_mean_k3'),
    ('mac_contribution', 'full', 'intercept_plus_lipid'),
    ('lipid_contribution', 'full', 'intercept_plus_mac'),
    ('intercept_necessity', 'full', 'channels_without_intercept'),
]
eff = []
for name, m, r in CONTRASTS:
    for metric, col in (('rmse', 'mse'), ('profile_rmse', 'profile_mse')):
        d = stats.paired_bootstrap(losses[m], losses[r], col, B=2000, seed=SEED)
        eff.append(dict(contrast=name, method_name=m, reference_name=r, metric=metric, **d))
eff = pd.DataFrame(eff)

table.to_csv(env.RESULTS / 't1_decomposition.csv', index=False)
eff.to_csv(env.RESULTS / 't1_contrasts.csv', index=False)

prim = eff[(eff.contrast == 'channel_increment_PRIMARY') & (eff.metric == 'rmse')].iloc[0]
lvl = eff[(eff.contrast == 'level_correction') & (eff.metric == 'rmse')].iloc[0]
env.dump(dict(
    primary_estimand='intercept_only -> full, participant-balanced RMSE, CGMacros, k=3',
    primary=dict(point=prim.difference, lo=prim.diff_lo, hi=prim.diff_hi,
                 relative_pct=prim.relative_pct, rel_lo=prim.rel_lo, rel_hi=prim.rel_hi,
                 better=int(prim.better), sign_p=prim.sign_p, se=prim.diff_se),
    mde_80pct_power=stats.mde(prim.diff_se),
    level_correction_share_of_total_pct=float(100 * lvl.difference / (lvl.difference + prim.difference)),
    bootstrap=2000, seed=SEED), 't1_decomposition')

pd.set_option('display.width', 220)
print(table.to_string(index=False))
print()
print(eff[eff.metric.eq('rmse')][['contrast', 'method_name', 'reference_name', 'method', 'reference',
                                  'difference', 'diff_lo', 'diff_hi', 'relative_pct', 'rel_lo', 'rel_hi',
                                  'better', 'sign_p']].to_string(index=False))
print()
print('PRIMARY channel increment: %.4f (95%% CI %.4f-%.4f) = %.2f%% (%.2f-%.2f), %d/44 better, sign p=%.4f'
      % (prim.difference, prim.diff_lo, prim.diff_hi, prim.relative_pct, prim.rel_lo, prim.rel_hi,
         prim.better, prim.sign_p))
print('MDE at 80%% power: %.4f' % stats.mde(prim.diff_se))
