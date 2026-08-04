# Task 6 Report

## Summary

Implemented a manifest-driven FCS2.0-style supplementary-material generator.
The runner now writes `supplement/supplementary_material.tex` and
`supplement/supplement_manifest.csv`; every required table is registered, with
unavailable upstream sources explicitly marked `missing_source`.

## Changed Files

- `code/src/gmnps/supplement/__init__.py`
- `code/src/gmnps/supplement/fcs_style.py`
- `code/src/scripts/run_personalized_calibration_experiments.py`
- `code/src/tests/test_fcs_style_supplement.py`

## Tests

- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_fcs_style_supplement.py code/src/tests/test_git_hygiene.py -q`
- `PYTHONPATH=code/src .venv/bin/python -c "import importlib.util; spec=importlib.util.spec_from_file_location('runner','code/src/scripts/run_personalized_calibration_experiments.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"`
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_personalized_calibration_runner.py -q`
