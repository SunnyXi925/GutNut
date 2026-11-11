"""Source data for Figure 6: adaptation rules and calibration-set design."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env

FD = env.ROOT / 'figure_data'
FD.mkdir(exist_ok=True)

RULE_LABEL = {
    'shared': 'No personal update', 'ols_head': 'Least squares (no shrinkage)',
    'ridge_head': 'Isotropic ridge', 'bayes_last_layer': 'Bayesian last layer',
    'blup_diag': 'Diagonal random effects', 'blup_full': 'Correlated random effects',
    'eb_population': 'Empirical Bayes, population prior',
    'eb_conditional': 'Empirical Bayes, conditional prior',
    'eb_intercept': 'Response level only'}
ORDER = ['shared', 'ols_head', 'ridge_head', 'bayes_last_layer', 'eb_intercept',
         'blup_diag', 'eb_population', 'eb_conditional', 'blup_full']
SEL_LABEL = {'first': 'First three', 'random': 'Random three', 'diverse': 'Most separated',
             'min_condition': 'Minimum condition', 'a_optimal': 'A-optimal',
             'i_optimal': 'I-optimal', 'd_optimal': 'D-optimal',
             'bayes_d_optimal': 'Bayesian D-optimal', 'oracle': 'Oracle (upper bound)'}
SEL_ORDER = ['first', 'random', 'diverse', 'min_condition', 'a_optimal', 'i_optimal',
             'd_optimal', 'bayes_d_optimal', 'oracle']

m = pd.read_csv(env.RESULTS / 't16_rule_models.csv')
m['label'] = m.rule.map(RULE_LABEL)
m['rank'] = m.rule.map({r: i for i, r in enumerate(ORDER)})
m['clipped'] = m.rule.eq('ols_head')
m.sort_values(['cohort', 'rank']).to_csv(FD / 'V39_6a.csv', index=False)

c = pd.read_csv(env.RESULTS / 't16_rule_contrasts.csv')
c = c[c.contrast.str.startswith('eb_conditional_vs')].copy()
c['reference_label'] = c.reference_name.map(RULE_LABEL)
c['rank'] = c.reference_name.map({r: i for i, r in enumerate(ORDER)})
c[['cohort', 'contrast', 'reference_name', 'reference_label', 'rank', 'difference',
   'diff_lo', 'diff_hi', 'relative_pct', 'better', 'worse', 'ties', 'sign_p']].sort_values(
    ['cohort', 'rank']).to_csv(FD / 'V39_6b.csv', index=False)

d = pd.read_csv(env.RESULTS / 't17_design_selectors.csv')
d = d[d.pool.eq(5)].copy()
d['label'] = d.selector.map(SEL_LABEL)
d['rank'] = d.selector.map({s: i for i, s in enumerate(SEL_ORDER)})
d['oracle'] = d.selector.eq('oracle')
d['increment_lo_pct'] = d.channel_increment_pct * d.channel_increment_lo / d.channel_increment_mean
d['increment_hi_pct'] = d.channel_increment_pct * d.channel_increment_hi / d.channel_increment_mean
d['log10_cond'] = np.log10(np.clip(d.median_cond_number, 1, 1e18))
d.sort_values(['cohort', 'rank']).to_csv(FD / 'V39_6cd.csv', index=False)

env.dump(dict(panels=['V39_6a', 'V39_6b', 'V39_6cd'],
              rule_order=ORDER, selector_order=SEL_ORDER,
              note=('Panel a is the adaptation-rule ladder, panel b the paired contrasts against '
                    'the reported update, panels c and d the calibration-design selectors at a '
                    'five-meal candidate pool.')), 't25_figure6_data')
print(m.sort_values(['cohort', 'rank'])[['cohort', 'label', 'rmse']].round(4).to_string(index=False))
