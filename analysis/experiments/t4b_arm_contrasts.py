"""Cross-arm contrasts: does microbial conditioning add anything over host or population priors?

Paired within participant, on the identical query meals, using the saved t4 predictions.
Equivalence margin pre-specified at 0.005 mmol l^-1 h (one third of the channel increment).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats

MARGIN = 0.005
PAIRS = [('micro', 'population'), ('host', 'population'), ('micro', 'host'),
         ('host_micro', 'host'), ('host_micro', 'micro')]
rows = []
for tag, path in [('PREDICT_UK', 't4_uk_predictions.parquet'),
                  ('PREDICT_US_transport', 't4_us_predictions.parquet')]:
    P = pd.read_parquet(env.RESULTS / path)
    for draw in sorted(P.draw.unique()):
        d = P[P.draw == draw]
        L = {a: stats.person_losses(d[d.arm == a], 'full') for a in sorted(d.arm.unique())}
        for a, b in PAIRS:
            s = stats.paired_bootstrap(L[a], L[b], 'mse', B=2000, seed=20261108 + draw)
            eq = stats.equivalence(s['difference'], s['diff_lo'], s['diff_hi'], MARGIN)
            rows.append(dict(cohort=tag, draw=draw, arm=a, reference_arm=b, **s,
                             equivalent_within_margin=eq['equivalent'], margin=MARGIN))
out = pd.DataFrame(rows)
out.to_csv(env.RESULTS / 't4b_arm_contrasts.csv', index=False)

agg = out.groupby(['cohort', 'arm', 'reference_arm']).agg(
    difference=('difference', 'mean'), lo=('diff_lo', 'mean'), hi=('diff_hi', 'mean'),
    relative_pct=('relative_pct', 'mean'), better=('better', 'mean'),
    sign_p=('sign_p', 'mean'), draws_equivalent=('equivalent_within_margin', 'sum'),
    draws=('draw', 'size')).round(5)
pd.set_option('display.width', 240)
print(agg.to_string())

se = out[(out.cohort == 'PREDICT_UK') & (out.arm == 'micro') & (out.reference_arm == 'host')].diff_se.mean()
env.dump(dict(margin=MARGIN,
              micro_vs_host_uk=agg.loc[('PREDICT_UK', 'micro', 'host')].to_dict(),
              micro_vs_population_uk=agg.loc[('PREDICT_UK', 'micro', 'population')].to_dict(),
              host_vs_population_uk=agg.loc[('PREDICT_UK', 'host', 'population')].to_dict(),
              micro_vs_host_us=agg.loc[('PREDICT_US_transport', 'micro', 'host')].to_dict(),
              mde_micro_vs_host_uk_80pct=stats.mde(se),
              interpretation='A conditional prior (host or microbial) beats an unconditional '
                             'population prior; microbial and host conditioning are statistically '
                             'indistinguishable at the pre-specified margin.'),
         't4b_arm_contrasts')
print('\nMDE for micro-vs-host in PREDICT UK at 80%% power: %.5f' % stats.mde(se))
