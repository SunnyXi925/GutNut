"""Honest validation of the personalised food score.

Three questions, each answered with observed data rather than with a projection:
  Q1 does the personal profile improve the ordering of *observed* responses within a person?
  Q2 how much of the food-score matrix is personal at all (variance decomposition)?
  Q3 does the personal term change which foods a person would be shown (rank shift)?

Note an analytic fact used as an internal check: the personal intercept is a constant within a
participant, so it cannot change any within-participant ordering. Ordering can only move
through the nutrient-channel terms.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from lib import env, stats, runner, protocols
from masked_operator import splits
from conditional_adapter import adapter_configs

SEED, K, TAU = 20261109, 3, 1.0
U = dict(mac=5.0, lipid=20.0)
CFG = adapter_configs()['masked_host_taxa_bw1.0_mean0.5']
LIB = env.require('food library')


def pair_concordance(obs, pred):
    """Within-participant pairwise concordance; ties take half credit."""
    o = np.asarray(obs, float); p = np.asarray(pred, float)
    n = len(o)
    if n < 2:
        return np.nan, 0
    do = o[:, None] - o[None, :]
    dp = p[:, None] - p[None, :]
    iu = np.triu_indices(n, 1)
    do, dp = do[iu], dp[iu]
    keep = do != 0
    if not keep.any():
        return np.nan, 0
    do, dp = do[keep], dp[keep]
    # lower predicted response should accompany lower observed response
    agree = np.where(dp == 0, 0.5, (np.sign(do) == np.sign(dp)).astype(float))
    return float(agree.mean()), int(len(do))


def ordering_table(P, cols, tag, extra=None):
    rows = []
    for pid, g in P.groupby('participant_id', sort=False):
        rec = dict(cohort=tag, participant_id=pid, meals=len(g))
        for c in cols:
            rec[c], rec['pairs'] = pair_concordance(g.target, g[c])
        if extra is not None and extra in g:
            # a nutritional-quality reference scores higher for healthier foods,
            # so it is used with its native direction reversed into a response proxy
            rec['food_compass'], _ = pair_concordance(g.target, -g[extra])
        rows.append(rec)
    return pd.DataFrame(rows)


def balanced_mean(frame, col):
    v = frame[col].dropna()
    return float(v.mean()), int(len(v))


def paired_delta(frame, a, b, B=2000, seed=SEED):
    d = frame[[a, b]].dropna()
    diff = (d[a] - d[b]).to_numpy()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(diff), size=(B, len(diff)))
    draws = diff[ix].mean(1)
    return dict(point=float(diff.mean()), lo=float(np.quantile(draws, .025)),
                hi=float(np.quantile(draws, .975)), participants=int(len(diff)),
                better=int((diff > 0).sum()), sign_p=stats.sign_test(int((diff > 0).sum()), int(len(diff))))


# ---------- Q1: ordering of observed responses --------------------------------------
results = {}
ord_tables = []
cg = env.load_cgmacros().sort_values(['participant_id', 'timestamp', 'meal_event_id']).reset_index(drop=True)
rows, atlas = [], []
meta = pd.read_parquet(LIB / 'metadata.parquet')
expo = pd.read_parquet(LIB / 'food_exposures.parquet')
assert len(meta) == 1226 and len(expo) == 1226
food = pd.DataFrame(dict(
    food__energy_kcal=100.0,
    food__protein_g=expo['Protein (g)'].to_numpy(float),
    food__carbohydrate_g=expo['Carbohydrate (g)'].to_numpy(float),
    food__sugar_g=expo['Sugars, total (g)'].to_numpy(float),
    food__fiber_g=expo['Fiber, total dietary (g)'].to_numpy(float),
    food__fat_g=expo['Total Fat (g)'].to_numpy(float)), index=expo.index)
food['food_group'] = meta.food_group.reindex(food.index)
food['fcs2'] = meta.fcs2.reindex(food.index)
phi_mac = food.food__fiber_g / (U['mac'] + food.food__fiber_g)
phi_lipid = food.food__fat_g / (U['lipid'] + food.food__fat_g)

for fold, (a, b) in enumerate(splits(cg, 5)):
    tr, te = cg.iloc[a].reset_index(drop=True), cg.iloc[b].reset_index(drop=True)
    protocols.assert_fold_disjoint(tr, te, 'participant_id')
    base, ad = runner.fit_fold(tr)
    m = runner.configure(ad, CFG, tr)
    p = runner.predict_subsets(m, te, K, 5)
    p['outer_fold'] = fold
    rows.append(p)
    # food-library projection for this fold's test participants
    host_cols = [c for c in cg.columns if c.startswith('person__')]
    for pid, g in p.groupby('participant_id', sort=False):
        q = food.copy()
        for c in host_cols:
            q[c] = te.loc[te.participant_id.eq(pid), c].iloc[0]
        for c in base.features.context_cols:
            q[c] = base.features.context_reference[c]
        shared = base.standardized(q)
        bm, bl = g.beta_mac.iloc[0], g.beta_lipid.iloc[0]
        adj = phi_mac.to_numpy() * bm + phi_lipid.to_numpy() * bl
        atlas.append(pd.DataFrame(dict(
            participant_id=pid, outer_fold=fold, food_id=food.index,
            food_group=food.food_group.to_numpy(), fcs2=food.fcs2.to_numpy(),
            shared_response=shared, adjustment=adj,
            score_shared=100 * expit(-shared / TAU),
            score_personal=100 * expit(-(shared + adj) / TAU))))
P = pd.concat(rows, ignore_index=True)
A = pd.concat(atlas, ignore_index=True)
A.to_parquet(env.RESULTS / 't11_food_atlas.parquet', index=False)

oc = ordering_table(P, ['shared', 'intercept_only', 'full'], 'CGMacros')
ord_tables.append(oc)
assert np.allclose(oc.shared.dropna(), oc.intercept_only.dropna()), \
    'the personal intercept must not change within-participant ordering'
results['CGMacros_ordering'] = dict(
    shared=balanced_mean(oc, 'shared')[0], full=balanced_mean(oc, 'full')[0],
    participants=int(oc.full.notna().sum()),
    full_vs_shared=paired_delta(oc, 'full', 'shared'),
    note='intercept-only ordering is identical to shared by construction (verified)')

# PREDICT, which also carries a nutritional-quality reference for every meal
uk = env.load_predict('UK')
rows2 = []
for fold, (a, b) in enumerate(splits(uk, 5)):
    tr, te = uk.iloc[a].reset_index(drop=True), uk.iloc[b].reset_index(drop=True)
    tr = protocols.inject_random_order(tr, SEED + fold)
    te = protocols.inject_random_order(te, SEED + 100 + fold)
    base, ad = runner.fit_fold(tr)
    m = runner.configure(ad, CFG, tr)
    p = runner.predict_subsets(m, te, K, 3)
    p = p.merge(te[['meal_event_id', 'fcs2']], on='meal_event_id', validate='one_to_one')
    p['outer_fold'] = fold
    rows2.append(p)
P2 = pd.concat(rows2, ignore_index=True)
op = ordering_table(P2, ['shared', 'intercept_only', 'full'], 'PREDICT_UK', extra='fcs2')
ord_tables.append(op)
results['PREDICT_UK_ordering'] = dict(
    shared=balanced_mean(op, 'shared')[0], full=balanced_mean(op, 'full')[0],
    food_compass=balanced_mean(op, 'food_compass')[0],
    participants=int(op.full.notna().sum()),
    full_vs_shared=paired_delta(op, 'full', 'shared'),
    full_vs_food_compass=paired_delta(op, 'full', 'food_compass'),
    shared_vs_food_compass=paired_delta(op, 'shared', 'food_compass'))
pd.concat(ord_tables, ignore_index=True).to_csv(env.RESULTS / 't11_ordering.csv', index=False)

# ---------- Q2: how personal is the score matrix? ----------------------------------
gm = A.score_personal.mean()
ss_total = ((A.score_personal - gm) ** 2).sum()
food_eff = A.groupby('food_id').score_personal.mean() - gm
part_eff = A.groupby('participant_id').score_personal.mean() - gm
ss_food = (A.food_id.map(food_eff) ** 2).sum()
ss_part = (A.participant_id.map(part_eff) ** 2).sum()
resid = A.score_personal - gm - A.food_id.map(food_eff) - A.participant_id.map(part_eff)
results['variance_decomposition'] = dict(
    cells=int(len(A)), foods=int(A.food_id.nunique()), participants=int(A.participant_id.nunique()),
    food_share_pct=float(100 * ss_food / ss_total),
    participant_share_pct=float(100 * ss_part / ss_total),
    interaction_share_pct=float(100 * (resid ** 2).sum() / ss_total),
    mean_abs_adjustment_score_points=float((A.score_personal - A.score_shared).abs().mean()),
    sd_adjustment_score_points=float((A.score_personal - A.score_shared).std()))

# ---------- Q3: does the personal term change what a person is shown? --------------
rk = []
for pid, g in A.groupby('participant_id', sort=False):
    rs = g.score_shared.rank(ascending=False)
    rp = g.score_personal.rank(ascending=False)
    top_s = set(g.food_id[rs <= 10]); top_p = set(g.food_id[rp <= 10])
    rk.append(dict(participant_id=pid,
                   spearman=float(spearmanr(g.score_shared, g.score_personal).statistic),
                   median_abs_rank_change=float((rp - rs).abs().median()),
                   p90_abs_rank_change=float((rp - rs).abs().quantile(.9)),
                   top10_overlap=len(top_s & top_p),
                   spearman_vs_fcs2_shared=float(spearmanr(g.score_shared, g.fcs2).statistic),
                   spearman_vs_fcs2_personal=float(spearmanr(g.score_personal, g.fcs2).statistic)))
R = pd.DataFrame(rk)
R.to_csv(env.RESULTS / 't11_rank_shift.csv', index=False)
results['rank_shift'] = {c: float(R[c].median()) for c in R.columns if c != 'participant_id'}
results['rank_shift']['foods'] = int(A.food_id.nunique())

# ---------- food-library provenance -------------------------------------------------
grp = meta.food_group.value_counts().sort_index()
results['food_library'] = dict(
    foods=int(len(meta)), source='FNDDS 2017-18 food codes (food_id prefix "2017-18:")',
    exposure_basis='per 100 kcal', food_groups=int(meta.food_group.nunique()),
    group_counts=grp.to_dict(),
    fcs2_available=int(meta.fcs2.notna().sum()),
    nutrient_fields=int(pd.read_parquet(LIB / 'food_exposures.parquet').shape[1]),
    fibre_zero_pct=float(100 * (food.food__fiber_g == 0).mean()),
    fat_zero_pct=float(100 * (food.food__fat_g == 0).mean()))
grp.rename('foods').to_frame().to_csv(env.RESULTS / 't11_food_library_groups.csv')

env.sha256_manifest([LIB / 'metadata.parquet', LIB / 'food_exposures.parquet',
                     env.cohort_file('cgmacros_functional.parquet'),
                     env.cohort_file('predict.parquet')], 't11')
env.dump(results, 't11_foodscore')

import json
print(json.dumps(results, ensure_ascii=False, indent=1, default=float))
