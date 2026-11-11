"""Paths, data loading and provenance for the GutNut analysis.

Input locations are read from the environment so that no machine-specific path is
committed. Set these before running any experiment:

    GUTNUT_MODEL_ROOT    directory holding the model package (default: analysis/lib/model)
    GUTNUT_SOURCE_DATA   directory holding the saved prediction tables (default: ./source_data)
    GUTNUT_FOOD_LIBRARY  directory holding the food library (default: ./food_library)
    GUTNUT_COHORT_DATA   directory holding the cohort parquet files (default: GUTNUT_SOURCE_DATA)

See README.md for the files each location must contain.
"""
from pathlib import Path
import hashlib, json, os, sys

ROOT = Path(__file__).resolve().parents[1]        # analysis/
REPO = ROOT.parent                                 # repository root

MODEL = Path(os.environ.get('GUTNUT_MODEL_ROOT', ROOT / 'model')).expanduser()
SOURCE = Path(os.environ.get('GUTNUT_SOURCE_DATA', REPO / 'source_data')).expanduser()
FOOD_LIBRARY = Path(os.environ.get('GUTNUT_FOOD_LIBRARY', REPO / 'food_library')).expanduser()
COHORT_DATA = Path(os.environ.get('GUTNUT_COHORT_DATA', SOURCE)).expanduser()

RESULTS = ROOT / 'results'
PROV = ROOT / 'provenance'
RESULTS.mkdir(parents=True, exist_ok=True)
PROV.mkdir(parents=True, exist_ok=True)

if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

REQUIRED = {
    'model': (MODEL, ['masked_operator.py', 'conditional_adapter.py',
                      'functional_operator.py', 'masks.py']),
    'source data': (SOURCE, ['module_predictions.csv', 'CG_future_predictions.csv',
                             'microbial_history_predictions.csv',
                             'microbial_history_coefficients.csv']),
    'food library': (FOOD_LIBRARY, ['metadata.parquet', 'food_exposures.parquet']),
}


def require(which):
    """Fail early with an actionable message when an input location is incomplete."""
    folder, names = REQUIRED[which]
    missing = [n for n in names if not (folder / n).exists()]
    if missing:
        raise SystemExit(
            f'{which} is incomplete at {folder}\n  missing: {", ".join(missing)}\n'
            f'  set the matching GUTNUT_* environment variable; see README.md')
    return folder


def cohort_file(name):
    """Resolve a cohort parquet from GUTNUT_COHORT_DATA, then the model package."""
    for candidate in (COHORT_DATA / name, MODEL / 'data' / name):
        if candidate.exists():
            return candidate
    raise SystemExit(f'{name} not found; set GUTNUT_COHORT_DATA to the directory holding it')


def load_cgmacros():
    import pandas as pd
    f = pd.read_parquet(cohort_file('cgmacros_functional.parquet')).reset_index(drop=True)
    assert len(f) == 1122 and f.participant_id.nunique() == 44
    return f


def load_predict(country=None):
    import pandas as pd
    f = pd.read_parquet(SOURCE / 'predict.parquet')
    if country is not None:
        f = f[f.country.eq(country)]
    return f.reset_index(drop=True)


def sha256_manifest(paths, name):
    m = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    (PROV / f'{name}_inputs.json').write_text(json.dumps(m, indent=2))
    return m


def dump(obj, name):
    (RESULTS / f'{name}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float))
    return obj
