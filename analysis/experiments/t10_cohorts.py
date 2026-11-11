"""Cohort description that v37 omits: Table 1, glycaemic strata, responder analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from lib import env, stats

cg = env.load_cgmacros()
person = cg.drop_duplicates('participant_id').set_index('participant_id')
strat = cg.groupby('participant_id').stratum.first()

rows = []
def describe(sub, label):
    p = person.loc[sub]
    r = dict(group=label, participants=len(p),
             meals=int(cg[cg.participant_id.isin(sub)].shape[0]))
    for c in ['person__age', 'person__bmi', 'person__hba1c', 'person__fasting_glucose']:
        r[c.replace('person__', '') + '_median'] = round(float(p[c].median()), 2)
        r[c.replace('person__', '') + '_iqr'] = '%.2f-%.2f' % (p[c].quantile(.25), p[c].quantile(.75))
    r['female_n'] = int(p['person__female'].sum())
    tgt = cg[cg.participant_id.isin(sub)].target
    r['iAUC_median'] = round(float(tgt.median()), 3)
    r['iAUC_iqr'] = '%.3f-%.3f' % (tgt.quantile(.25), tgt.quantile(.75))
    r['iAUC_zero_pct'] = round(float(100 * (tgt == 0).mean()), 2)
    r['meals_per_person_median'] = int(cg[cg.participant_id.isin(sub)].groupby('participant_id').size().median())
    return r

rows.append(describe(person.index, 'all'))
for s in sorted(strat.unique()):
    rows.append(describe(strat[strat == s].index, s))
table1 = pd.DataFrame(rows)
table1.to_csv(env.RESULTS / 't10_table1_cgmacros.csv', index=False)

# PREDICT description
pr = env.load_predict()
prow = []
for c in sorted(pr.country.unique()):
    s = pr[pr.country.eq(c)]
    p = s.drop_duplicates('participant_id')
    prow.append(dict(cohort='PREDICT_' + c, participants=int(p.participant_id.nunique()),
                     families=int(s.family_group.nunique()), meals=len(s),
                     meal_compositions=int(s.meal_id.nunique()),
                     age_median=round(float(p['person__age'].median()), 1),
                     bmi_median=round(float(p['person__bmi_kg_m2'].median()), 1),
                     female_n=int(p['person__female'].sum()),
                     target_median=round(float(s.target.median()), 3),
                     target_zero_pct=round(float(100 * (s.target == 0).mean()), 2)))
pd.DataFrame(prow).to_csv(env.RESULTS / 't10_table1_predict.csv', index=False)

# Dexcom device replication: which meals carry a paired second sensor
aux = cg[cg.target_aux.notna()]
device = dict(meals_with_libre=int(len(cg)), meals_with_paired_dexcom=int(len(aux)),
              participants_with_any_dexcom=int(aux.participant_id.nunique()),
              pearson_libre_dexcom=float(np.corrcoef(aux.target, aux.target_aux)[0, 1]),
              spearman_libre_dexcom=float(aux[['target', 'target_aux']].corr(method='spearman').iloc[0, 1]),
              median_abs_difference=float((aux.target - aux.target_aux).abs().median()),
              note='target=Libre positive excursion x2; target_aux=Dexcom positive excursion x2. '
                   'Same meals, two sensors: a measurement-replication check, not a second endpoint.')

# Responder analysis on the published k=3 predictions
mod = pd.read_csv(env.SOURCE / 'module_predictions.csv')
full = stats.person_losses(mod[mod.model.eq('full')], 'prediction')
ioff = stats.person_losses(mod[mod.model.eq('offset_only')], 'prediction')
cgp = pd.read_csv(env.SOURCE / 'CG_future_predictions.csv')
shd = stats.person_losses(cgp[cgp.model.eq('same_base_without_history') & cgp.history_k.eq(0)], 'prediction')
resp = pd.DataFrame(dict(
    channel_gain=np.sqrt(ioff.mse) - np.sqrt(full.mse),
    total_gain=np.sqrt(shd.mse) - np.sqrt(full.mse),
    query_meals=full.n))
resp['stratum'] = strat.reindex(resp.index)
for c in ['person__age', 'person__bmi', 'person__hba1c', 'person__fasting_glucose', 'person__female']:
    resp[c.replace('person__', '')] = person[c].reindex(resp.index)
resp['target_sd'] = cg.groupby('participant_id').target.std().reindex(resp.index)
resp['target_mean'] = cg.groupby('participant_id').target.mean().reindex(resp.index)
resp['channel_responder'] = resp.channel_gain > 0
resp.to_csv(env.RESULTS / 't10_responder.csv')

contrast = resp.groupby('channel_responder')[['age', 'bmi', 'hba1c', 'fasting_glucose',
                                              'target_mean', 'target_sd', 'query_meals']].median()
by_stratum = resp.groupby('stratum').agg(n=('channel_gain', 'size'),
                                         median_channel_gain=('channel_gain', 'median'),
                                         responders=('channel_responder', 'sum'),
                                         median_total_gain=('total_gain', 'median'))

env.dump(dict(device_replication=device,
              responders=int(resp.channel_responder.sum()), participants=int(len(resp)),
              sign_p=stats.sign_test(int(resp.channel_responder.sum()), int(len(resp))),
              responder_vs_nonresponder_medians=contrast.round(3).to_dict(),
              by_stratum=by_stratum.round(4).to_dict(),
              spearman_gain_vs_target_sd=float(resp[['channel_gain', 'target_sd']].corr(method='spearman').iloc[0, 1]),
              spearman_gain_vs_hba1c=float(resp[['channel_gain', 'hba1c']].corr(method='spearman').iloc[0, 1])),
         't10_cohorts')

pd.set_option('display.width', 240)
print('== CGMacros Table 1 =='); print(table1.to_string(index=False))
print('\n== PREDICT =='); print(pd.DataFrame(prow).to_string(index=False))
print('\n== device replication =='); print({k: v for k, v in device.items() if k != 'note'})
print('\n== channel gain by stratum =='); print(by_stratum.round(4).to_string())
print('\n== responder vs non-responder (medians) =='); print(contrast.round(3).to_string())
