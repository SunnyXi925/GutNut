"""Ordered input names and fixed candidate settings; contains no participant data."""

PERSON_NUM = ['age', 'bmi_kg_m2', 'waist_hip_ratio']



PERSON_CAT = ['sex', 'smoking_status', 'diabetes_history', 'medications', 'antibiotics_recent']


HABITUAL = ['habitual_diet__' + s for s in (
    'amed_score', 'carbohydrate_energy', 'coffee', 'fat_energy', 'fiber', 'fruit',
    'healthy_plant_score', 'protein_energy', 'vegetables', 'whole_grains',
)]



SPECIES = ['microbiome__species__' + s for s in (
    'Akkermansia_muciniphila', 'Alistipes_inops', 'Bacteroides_uniformis',
    'Bifidobacterium_adolescentis', 'Clostridium_CAG58', 'Eubacterium_rectale',
    'Faecalibacterium_prausnitzii', 'Firmicutes_bacterium_CAG95', 'Oscillibacter_57_20',
    'Prevotella_copri', 'Roseburia_hominis', 'Ruminococcus_torques',
)]



PATHWAYS = [f'microbiome__pathway__MetaCyc_PWY_{k:03d}' for k in range(1, 7)]



MICRO_COLUMNS = SPECIES + PATHWAYS


SEED=20260930



MEALS=tuple(f'Meal{i}' for i in range(2,9))



FOOD=['energy_kcal','carbohydrate_g','sugar_g','fat_g','protein_g','fiber_g']



CONTEXT=['fasting_duration_h','mvpa_minutes_pre2h','light_activity_minutes_pre2h','sleep_duration_h',
         'sleep_efficiency','time_since_wakeup_h','time_since_previous_meal_h','clock_sin','clock_cos']


FG = 'fasting_lab__glucose'



NUMERIC = PERSON_NUM + HABITUAL + [FG]



GRID = [dict(name='ridge100', kind='ridge', alpha=100.),
        dict(name='hgb120_leaves7', kind='hgb', max_iter=120,max_leaf_nodes=7),
        dict(name='hgb300_leaves15', kind='hgb', max_iter=300,max_leaf_nodes=15)]



NULL = dict(name='recipe_training_mean',kind='null')
