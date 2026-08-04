# Task 4 Report

## Summary

Implemented signed knowledge-graph path validation and enumeration, preserving directed edge traversal and multiplying edge signs across every hop. Added class-balanced prediction metrics with a `single_class_collapse` flag when one predicted label represents at least 95% of predictions.

## Changed Files

- `code/src/gmnps/knowledge_graph/signed_paths.py`
- `code/src/gmnps/knowledge_graph/adjudicator_eval.py`
- `code/src/tests/test_signed_paths.py`
- `code/src/tests/test_adjudicator_eval.py`

## Tests

```text
PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_signed_paths.py code/src/tests/test_adjudicator_eval.py code/src/tests/test_evidence_graph.py -q
6 passed in 0.51s
```

## Reviewer Fix

Typed node identifiers are now used for adjacency and cycle tracking, preventing traversal across nodes that share text IDs but have different types. Added a regression test covering the false nutrient-to-disease path this could produce.
