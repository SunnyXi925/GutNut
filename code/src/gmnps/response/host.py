"""Host and directly augmented response learners with training-only transforms."""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .schema import SPECIES, PATHWAYS, MICRO_COLUMNS, NUMERIC, PERSON_CAT, FOOD, CONTEXT, SEED
from .microbiome import stable_clr, person_weights, real_reference

GROUPS = ('host_only', 'CLR_micro')

class HomeFastingGlucoseFeatures:
    """Explicit prospective allowlists with current-training transforms only."""
    def __init__(self, group):
        if group not in GROUPS:
            raise ValueError(group)
        self.group = group

    def micro(self, people):
        return np.column_stack([stable_clr(people[columns].to_numpy(float)) for columns in (SPECIES, PATHWAYS)])

    def fit(self, people, meals):
        if not people.index.is_unique or set(meals.participant_id) != set(people.index):
            raise ValueError('Unique people with observed meals required')
        self.fit_ids = people.index.tolist()
        self.fit_events = meals.meal_event_id.tolist()
        self.host_numeric = list(NUMERIC)
        self.host_imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(people[self.host_numeric])
        self.host_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(people[PERSON_CAT].astype(str))
        h = np.column_stack([self.host_imputer.transform(people[self.host_numeric]),
                             self.host_encoder.transform(people[PERSON_CAT].astype(str))])
        self.host_scaler = StandardScaler().fit(h)
        self.event_imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(meals[FOOD + CONTEXT])
        self.event_scaler = StandardScaler().fit(self.event_imputer.transform(meals[FOOD + CONTEXT]),
                                                sample_weight=person_weights(meals))
        self.recipe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(meals[['meal_id']].astype(str))
        self.feature_names = (FOOD + CONTEXT + list(self.recipe_encoder.get_feature_names_out(['meal_id'])) +
                              self.host_numeric + list(self.host_encoder.get_feature_names_out(PERSON_CAT)))
        self.reference_id, self.reference, self.reference_distance = None, {}, None
        if self.group != 'host_only':
            self.micro_scaler = StandardScaler().fit(self.micro(people))
            self.reference_id, self.reference, self.reference_distance = real_reference(people)
            self.feature_names += MICRO_COLUMNS
        return self

    def transform(self, people, meals):
        if not people.index.is_unique:
            raise ValueError('Person index must be unique')
        ix = people.index.get_indexer(meals.participant_id)
        if (ix < 0).any():
            raise ValueError('Unmatched meal participant')
        h = np.column_stack([self.host_imputer.transform(people[self.host_numeric]),
                             self.host_encoder.transform(people[PERSON_CAT].astype(str))])
        parts = [self.event_scaler.transform(self.event_imputer.transform(meals[FOOD + CONTEXT])),
                 self.recipe_encoder.transform(meals[['meal_id']].astype(str)),
                 self.host_scaler.transform(h)[ix]]
        if self.group != 'host_only':
            parts.append(self.micro_scaler.transform(self.micro(people))[ix])
        x = np.column_stack(parts)
        if not np.isfinite(x).all():
            raise ValueError('Finite prediction design required')
        return x



class HomeFastingGlucoseModel:
    def __init__(self, group, spec):
        if group not in GROUPS:
            raise ValueError(group)
        self.endpoint, self.group, self.spec = 'signed_iauc_120', group, dict(spec)

    def fit(self, people, meals, target):
        start = time.monotonic()
        target = np.asarray(target, float)
        if len(target) != len(meals) or not np.isfinite(target).all():
            raise ValueError('Finite training-only target required')
        if meals.duplicated(['participant_id', 'meal_id']).any():
            raise ValueError('One meal per person/recipe required')
        w = person_weights(meals)
        self.offset = float(np.average(target, weights=w))
        self.scale = max(float(np.sqrt(np.average((target - self.offset) ** 2, weights=w))), 1e-8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.features, self.model = None, None
            if self.spec['kind'] == 'null':
                self.recipe_means = pd.Series(target).groupby(meals.meal_id.reset_index(drop=True)).mean().to_dict()
            else:
                self.features = HomeFastingGlucoseFeatures(self.group).fit(people, meals)
                if self.spec['kind'] == 'ridge':
                    self.model = Ridge(alpha=self.spec['alpha'])
                elif self.spec['kind'] == 'hgb':
                    self.model = HistGradientBoostingRegressor(
                        max_iter=self.spec['max_iter'], max_leaf_nodes=self.spec['max_leaf_nodes'],
                        min_samples_leaf=30, learning_rate=.06, l2_regularization=10.,
                        early_stopping=False, random_state=SEED)
                else:
                    raise ValueError(self.spec)
                self.model.fit(self.features.transform(people, meals), (target - self.offset) / self.scale,
                               sample_weight=w)
        self.warnings = [str(x.message) for x in caught]
        self.fit_ids, self.fit_events = people.index.tolist(), meals.meal_event_id.tolist()
        self.fit_families = sorted(people.family_group.unique())
        self.person_weight_range = pd.Series(w).groupby(meals.participant_id.reset_index(drop=True)).sum().agg(['min', 'max']).tolist()
        self.seconds = time.monotonic() - start
        return self

    def predict(self, people, meals):
        if self.model is None:
            result = meals.meal_id.map(self.recipe_means).to_numpy(float)
            if not np.isfinite(result).all():
                raise ValueError('Unseen recipe in recipe-only null')
            return result
        return self.offset + self.scale * self.model.predict(self.features.transform(people, meals))

    def reference_people(self, people):
        p = people.copy()
        if self.features is not None and self.group != 'host_only':
            columns = list(self.features.reference)
            p[columns] = np.broadcast_to(list(self.features.reference.values()), (len(p), len(columns)))
        return p

    def predict_reference(self, people, meals):
        return self.predict(self.reference_people(people), meals)
