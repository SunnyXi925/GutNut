"""Paths, data loading and provenance for the NCS evidence rebuild."""
from pathlib import Path
import hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/Users/fengxi.25/Downloads/1017GutNut/model_snapshot')
SOURCE = Path('/Users/fengxi.25/Downloads/1017GutNut/source_data')
RESULTS = ROOT / 'results'
PROV = ROOT / 'provenance'
RESULTS.mkdir(parents=True, exist_ok=True)
PROV.mkdir(parents=True, exist_ok=True)

for p in (MODEL / 'project_support', MODEL / 'code'):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_cgmacros():
    import pandas as pd
    f = pd.read_parquet(MODEL / 'data/cgmacros_functional.parquet').reset_index(drop=True)
    assert len(f) == 1122 and f.participant_id.nunique() == 44
    return f


def load_predict(country=None):
    import pandas as pd
    f = pd.read_parquet(MODEL / 'data/predict.parquet')
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
