"""Compositional model transform, training identities and family-preserving controls."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from .schema import MICRO_COLUMNS, SPECIES, PATHWAYS, SEED

N_FOLDS = 3

def sequential_row_sum(array):
    total = np.zeros(len(array), dtype=np.float64)
    for column in range(array.shape[1]):
        total += array[:, column]
    return total



def stable_clr(array, pseudocount=1e-6):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError('Finite nonnegative abundance matrix required')
    total = sequential_row_sum(array)
    if (total <= 0).any():
        raise ValueError('Nonempty microbial panel required')
    logged = np.log(array / total[:, None] + pseudocount)
    return logged - sequential_row_sum(logged)[:, None] / logged.shape[1]


def _labels(index):
    """Return JSON-safe identity labels without changing lookup semantics."""
    return [x.item() if isinstance(x, np.generic) else x for x in index]



def _validate_index(frame, name):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f'{name} must be a DataFrame indexed by participant ID')
    if isinstance(frame.index, pd.MultiIndex) or not frame.index.is_unique or frame.index.hasnans:
        raise ValueError(f'{name} requires unique, nonmissing participant IDs')
    if len(set(map(str, frame.index))) != len(frame):
        raise ValueError(f'{name} IDs must have distinct string representations')
    if not frame.columns.is_unique:
        raise ValueError(f'{name} requires unique columns')



def _family_labels(people):
    _validate_index(people, 'people')
    if len(people) == 0 or 'family_group' not in people:
        raise ValueError('Nonempty people with family_group required')
    families = people.family_group
    if families.isna().any():
        raise ValueError('Nonmissing family groups required')
    result = families.astype(str)
    if result.nunique() != families.nunique():
        raise ValueError('Families must have distinct string representations')
    return result



def microbial_clr(people):
    """Fixed 12-species CLR followed by the fixed six-pathway CLR."""
    _validate_index(people, 'microbial frame')
    if not set(MICRO_COLUMNS).issubset(people.columns):
        raise ValueError('All fixed 12 species and six pathway columns required')
    result = np.column_stack([
        stable_clr(people.loc[:, block].to_numpy(dtype=np.float64))
        for block in (SPECIES, PATHWAYS)
    ])
    if not np.isfinite(result).all():
        raise ValueError('Finite nonnegative nonempty microbial blocks required')
    return np.ascontiguousarray(result)



def family_folds(people, seed=SEED):
    """Hash-order all current-training families, then assign round robin."""
    families = _family_labels(people)
    ordered = sorted(families.unique(), key=lambda f: (
        hashlib.sha256(f'{seed}|{f}'.encode('utf-8')).hexdigest(), f))
    lookup = {family: k % N_FOLDS for k, family in enumerate(ordered)}
    return families.map(lookup).to_numpy(dtype=np.int64)



def _rng_seed(seed, context_label, fit_label, block_size):
    payload = json.dumps([int(seed), context_label, fit_label, int(block_size)],
                         ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:8], 'big')



def _mapping(pair_ids, families, mode, seed, context_label, fit_label):
    """Map complete target vectors within the current allowed family blocks."""
    positions = {pid: k for k, pid in enumerate(pair_ids)}
    by_family = {}
    for pid in pair_ids:
        by_family.setdefault(families.loc[pid], []).append(pid)
    for ids in by_family.values():
        ids.sort(key=str)
    strata = {}
    for family, ids in by_family.items():
        strata.setdefault(len(ids), []).append(family)
    donors = list(pair_ids)
    records = []
    for size, members in sorted(strata.items()):
        members.sort()
        rng_seed = _rng_seed(seed, context_label, fit_label, size)
        order = list(members)
        shift = 0
        if mode == 'shuffle' and len(order) >= 2:
            rng = np.random.default_rng(rng_seed)
            order = [order[i] for i in rng.permutation(len(order))]
            shift = int(rng.integers(1, len(order)))
        sources = order[shift:] + order[:shift]
        for recipient, source in zip(order, sources):
            for recipient_id, source_id in zip(by_family[recipient], by_family[source]):
                donors[positions[recipient_id]] = source_id
        records.append({
            'paired_members_per_family': size, 'families': order,
            'source_families': sources, 'nonzero_cycle_shift': shift,
            'rng_seed': rng_seed, 'unmoved_singleton_stratum': len(order) == 1,
        })
    if set(donors) != set(pair_ids) or len(set(donors)) != len(donors):
        raise RuntimeError('Auxiliary mapping must be a closed bijection')
    return donors, records


def person_weights(frame):
    w=1/frame.participant_id.map(frame.participant_id.value_counts()).to_numpy(float)
    return w/w.mean()


def closed_blocks(people):
    result = []
    for columns in [SPECIES, PATHWAYS]:
        values = people[columns].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values.sum(1) <= 0).any():
            raise ValueError('Finite nonnegative microbial blocks required')
        result.append(values / values.sum(1, keepdims=True))
    return result



def real_reference(people):
    """Closest real panel to the separately closed arithmetic mean; lexical ties."""
    ordered = people.loc[sorted(people.index, key=str)]
    closed = np.column_stack(closed_blocks(ordered))
    mean = closed.mean(0)
    distance = .5 * np.sum((np.sqrt(closed) - np.sqrt(mean)) ** 2, axis=1)
    winner = int(np.argmin(distance))
    ref_id = ordered.index[winner]
    return ref_id, ordered.iloc[winner][MICRO_COLUMNS].astype(float).to_dict(), float(distance[winner])
