"""GMNPS — Gut-Microbiome-informed Nutrient Profiling System.

Top-level reproducible package matching the manuscript Methods §3:

    gmnps.data           — Layer A (cMD) + Layer D (GMMAD v2) bundle loader
    gmnps.preprocessing  — Eq. 1  CLR transform + prevalence filtering
    gmnps.knowledge      — B-matrix (signed) + nutrient↔metabolite bridge
    gmnps.models         — Eq. 2 H = f(M),  Eq. 3-4 state-space,
                            Eq. 5 ElasticNet personalised W solver
    gmnps.inference      — frozen-bundle inference engine for new samples
    gmnps.food_compass   — GMNPS food-scoring (Result §2.1) and FCS join
    gmnps.validation     — independent-cohort experiments E1/E3/E4/E5 +
                            Cox empirical weights + partial-correlation audit
    gmnps.utils          — IO / logging / seed helpers
"""
__version__ = "1.0.0"

_LAZY_EXPORTS = {
    "clr_transform": ("gmnps.preprocessing.clr", "clr_transform"),
    "preprocess_microbiome": ("gmnps.preprocessing.clr", "preprocess_microbiome"),
    "build_signed_normalized_b": ("gmnps.knowledge.b_matrix", "build_signed_normalized_b"),
    "build_composite_target": ("gmnps.models.health_state", "build_composite_target"),
    "train_health_state": ("gmnps.models.health_state", "train_health_state"),
    "fit_population_covariance": ("gmnps.models.state_space", "fit_population_covariance"),
    "find_target_microbiomes": ("gmnps.models.state_space", "find_target_microbiomes"),
    "ElasticNetWeightSolver": ("gmnps.models.weights", "ElasticNetWeightSolver"),
    "GMNPSPredictor": ("gmnps.inference.predictor", "GMNPSPredictor"),
    "GenusAligner": ("gmnps.inference.aligner", "GenusAligner"),
    "align_new_sample": ("gmnps.inference.aligner", "align_new_sample"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'gmnps' has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
