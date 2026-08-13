"""Locked beta normalization and bounded Food Compass attribute calibration."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from gmnps.scoring.fcs2_attribute_mapping import (
    PRIMARY_ATTRIBUTE_MAPPINGS,
    PRIMARY_MAPPING_VERSION,
    SENSITIVITY_ATTRIBUTE_MAPPINGS,
    build_food_specific_response,
)
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES, NOT_CALCULATED, NotCalculated


BETA_NORMALIZATION_METHOD_VERSION = "median_mad_iqr_sd_v1"
LOCKED_BETA_TEMPERATURE = 2.0
LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE = 2.0
ATTRIBUTE_POINT_FRACTION_MODES: Mapping[str, float] = MappingProxyType(
    {"primary": 0.20, "low": 0.10, "high": 0.30}
)

_MAD_NORMAL_CONSISTENCY = 1.4826
_IQR_NORMAL_CONSISTENCY = 1.349
_ALLOWED_SCALE_METHODS = frozenset({"mad", "iqr", "standard_deviation", "unit"})
_ACTIVE_ATTRIBUTES = tuple(name for name, rule in FCS2_RULES.items() if rule.active)
_HEX_DIGITS = frozenset("0123456789abcdef")
_CALIBRATION_DIAGNOSTIC_COLUMNS = (
    "baseline_points",
    "attribute_response",
    "raw_delta",
    "point_delta",
    "lower_bound",
    "upper_bound",
    "lambda_points",
    "fraction_cap",
    "mode",
    "clipped",
    "not_calculated",
)


@dataclass(frozen=True)
class BetaNormalizationState:
    """Immutable development-fitted nutrient normalization parameters."""

    nutrient_order: tuple[str, ...]
    median_values: tuple[float, ...]
    scale_values: tuple[float, ...]
    scale_method_values: tuple[str, ...]
    fit_n: int
    fit_id_sha256: str
    method_version: str
    temperature: float
    state_fingerprint: str

    def __post_init__(self) -> None:
        _validate_normalization_state(self)

    @property
    def medians(self) -> Mapping[str, float]:
        return MappingProxyType(dict(zip(self.nutrient_order, self.median_values)))

    @property
    def scales(self) -> Mapping[str, float]:
        return MappingProxyType(dict(zip(self.nutrient_order, self.scale_values)))

    @property
    def scale_methods(self) -> Mapping[str, str]:
        return MappingProxyType(dict(zip(self.nutrient_order, self.scale_method_values)))

    @property
    def median(self) -> Mapping[str, float]:
        """Singular alias for manifest consumers."""

        return self.medians

    @property
    def scale(self) -> Mapping[str, float]:
        """Singular alias for manifest consumers."""

        return self.scales


@dataclass(frozen=True)
class AttributeResponseResult:
    """Responses bound to their normalization and reviewed mapping provenance."""

    normalization_state: BetaNormalizationState
    normalization_fingerprint: str
    mapping_version: str
    reviewed_targets: tuple[str, ...]
    responses: pd.DataFrame
    response_fingerprint: str
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class AttributeCalibrationResult:
    """Calibrated points bound to validated response and audit provenance."""

    points: pd.DataFrame
    deltas: pd.DataFrame
    diagnostics: pd.DataFrame
    normalization_fingerprint: str
    mapping_version: str
    reviewed_targets: tuple[str, ...]
    response_fingerprint: str
    calibration_mode: str
    calibration_fraction: float
    calibration_fingerprint: str

    def __post_init__(self) -> None:
        validate_calibration_result(self)


def _require_frame(value: object, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    if value.empty:
        raise ValueError(f"{label} must not be empty")
    if not value.index.is_unique:
        raise ValueError(f"{label} contains duplicate index labels")
    if not value.columns.is_unique:
        raise ValueError(f"{label} contains duplicate column labels")
    if isinstance(value.index, pd.MultiIndex):
        has_missing_index_label = any(
            pd.isna(value.index.get_level_values(level)).any()
            for level in range(value.index.nlevels)
        )
    else:
        has_missing_index_label = value.index.hasnans
    if has_missing_index_label:
        raise ValueError(f"{label} index labels must not be missing")
    return value


def _require_nonempty_string_index(index: pd.Index, label: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in index):
        raise ValueError(f"{label} participant IDs must be unique nonempty strings")


def _finite_numeric_frame(value: pd.DataFrame, label: str) -> np.ndarray:
    if any(dtype == bool for dtype in value.dtypes):
        raise ValueError(f"{label} values must be finite numbers, not booleans")
    try:
        numeric = value.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} values must be finite numbers") from error
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} values must be finite numbers")
    return numeric


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX_DIGITS
    )


def _sha256_payload(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_scalar(value: object) -> object:
    if isinstance(value, NotCalculated):
        return {"type": "not_calculated"}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return {"type": "nan"}
        if not isfinite(numeric):
            raise ValueError("fingerprinted numeric values must be finite or NaN")
        return numeric
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"unsupported fingerprint value type: {type(value).__name__}")


def _frame_payload(frame: pd.DataFrame) -> dict[str, object]:
    if isinstance(frame.index, pd.MultiIndex):
        index_values = [
            [_canonical_scalar(item) for item in index_value]
            for index_value in frame.index.tolist()
        ]
    else:
        index_values = [[_canonical_scalar(item)] for item in frame.index.tolist()]
    return {
        "columns": [_canonical_scalar(column) for column in frame.columns],
        "index_names": [_canonical_scalar(name) for name in frame.index.names],
        "index_values": index_values,
        "values": [
            [_canonical_scalar(item) for item in row]
            for row in frame.to_numpy(dtype=object).tolist()
        ],
    }


def _calibration_fingerprint(
    *,
    points: pd.DataFrame,
    deltas: pd.DataFrame,
    diagnostics: pd.DataFrame,
    normalization_fingerprint: str,
    mapping_version: str,
    reviewed_targets: tuple[str, ...],
    response_fingerprint: str,
    calibration_mode: str,
    calibration_fraction: float,
) -> str:
    return _sha256_payload(
        {
            "calibration_fraction": float(calibration_fraction),
            "calibration_mode": calibration_mode,
            "deltas": _frame_payload(deltas),
            "diagnostics": _frame_payload(diagnostics),
            "mapping_version": mapping_version,
            "normalization_fingerprint": normalization_fingerprint,
            "points": _frame_payload(points),
            "response_fingerprint": response_fingerprint,
            "reviewed_targets": list(reviewed_targets),
        }
    )


def _normalization_fingerprint(
    *,
    nutrient_order: tuple[str, ...],
    median_values: tuple[float, ...],
    scale_values: tuple[float, ...],
    scale_method_values: tuple[str, ...],
    fit_n: int,
    fit_id_sha256: str,
    method_version: str,
    temperature: float,
) -> str:
    return _sha256_payload(
        {
            "fit_id_sha256": fit_id_sha256,
            "fit_n": fit_n,
            "median_values": [float(value) for value in median_values],
            "method_version": method_version,
            "nutrient_order": list(nutrient_order),
            "scale_method_values": list(scale_method_values),
            "scale_values": [float(value) for value in scale_values],
            "temperature": float(temperature),
        }
    )


def _validate_normalization_state(state: BetaNormalizationState) -> None:
    if not isinstance(state, BetaNormalizationState):
        raise TypeError("state must be a BetaNormalizationState")
    vectors = (
        state.nutrient_order,
        state.median_values,
        state.scale_values,
        state.scale_method_values,
    )
    if any(not isinstance(vector, tuple) for vector in vectors):
        raise ValueError("normalization state vectors must be immutable tuples")
    lengths = {len(vector) for vector in vectors}
    if lengths != {len(state.nutrient_order)}:
        raise ValueError("normalization state vectors must have equal lengths")
    if not state.nutrient_order or len(set(state.nutrient_order)) != len(state.nutrient_order):
        raise ValueError("normalization state requires unique nutrient labels")
    if any(not isinstance(label, str) or not label for label in state.nutrient_order):
        raise ValueError("normalization state nutrient labels must be nonempty strings")
    if isinstance(state.fit_n, bool) or not isinstance(state.fit_n, int) or state.fit_n <= 0:
        raise ValueError("normalization state fit_n must be a positive integer")
    if not _is_sha256(state.fit_id_sha256):
        raise ValueError("normalization state fit-ID hash must be a SHA-256 hex digest")
    if state.method_version != BETA_NORMALIZATION_METHOD_VERSION:
        raise ValueError("normalization state method version is not locked")
    if (
        isinstance(state.temperature, bool)
        or not isinstance(state.temperature, (int, float))
        or not isfinite(float(state.temperature))
        or float(state.temperature) != LOCKED_BETA_TEMPERATURE
    ):
        raise ValueError("normalization state temperature must be exactly 2.0")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or not isfinite(float(value))
        for value in state.median_values
    ):
        raise ValueError("normalization state medians must be finite")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or not isfinite(float(value))
        or float(value) <= 0.0
        for value in state.scale_values
    ):
        raise ValueError("normalization state scales must be finite and positive")
    if any(method not in _ALLOWED_SCALE_METHODS for method in state.scale_method_values):
        raise ValueError("normalization state scale methods are not recognized")
    if not _is_sha256(state.state_fingerprint):
        raise ValueError("normalization state fingerprint must be a SHA-256 hex digest")
    expected_fingerprint = _normalization_fingerprint(
        nutrient_order=state.nutrient_order,
        median_values=state.median_values,
        scale_values=state.scale_values,
        scale_method_values=state.scale_method_values,
        fit_n=state.fit_n,
        fit_id_sha256=state.fit_id_sha256,
        method_version=state.method_version,
        temperature=float(state.temperature),
    )
    if state.state_fingerprint != expected_fingerprint:
        raise ValueError("normalization state fingerprint does not match its contents")


def _fit_id_hash(index: pd.Index) -> str:
    _require_nonempty_string_index(index, "development_beta")
    return sha256("\n".join(sorted(index)).encode("utf-8")).hexdigest()


def fit_beta_normalization(development_beta: pd.DataFrame) -> BetaNormalizationState:
    """Fit the locked nutrient-wise normalization on development beta only."""

    development_beta = _require_frame(development_beta, "development_beta")
    _require_nonempty_string_index(development_beta.index, "development_beta")
    if any(not isinstance(column, str) or not column for column in development_beta.columns):
        raise ValueError("development_beta nutrient labels must be nonempty strings")

    values = _finite_numeric_frame(development_beta, "development_beta")
    medians = np.median(values, axis=0)
    median_absolute_deviations = np.median(np.abs(values - medians), axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    standard_deviations = np.std(values, axis=0, ddof=0)

    scales: list[float] = []
    methods: list[str] = []
    for mad, low_quartile, high_quartile, standard_deviation in zip(
        median_absolute_deviations, q25, q75, standard_deviations
    ):
        mad_scale = float(mad) * _MAD_NORMAL_CONSISTENCY
        iqr_scale = float(high_quartile - low_quartile) / _IQR_NORMAL_CONSISTENCY
        if mad_scale > 0.0:
            scales.append(mad_scale)
            methods.append("mad")
        elif iqr_scale > 0.0:
            scales.append(iqr_scale)
            methods.append("iqr")
        elif float(standard_deviation) > 0.0:
            scales.append(float(standard_deviation))
            methods.append("standard_deviation")
        else:
            scales.append(1.0)
            methods.append("unit")

    nutrient_order = tuple(development_beta.columns)
    median_values = tuple(float(value) for value in medians)
    scale_values = tuple(scales)
    scale_method_values = tuple(methods)
    fit_id_sha256 = _fit_id_hash(development_beta.index)
    fingerprint = _normalization_fingerprint(
        nutrient_order=nutrient_order,
        median_values=median_values,
        scale_values=scale_values,
        scale_method_values=scale_method_values,
        fit_n=len(development_beta),
        fit_id_sha256=fit_id_sha256,
        method_version=BETA_NORMALIZATION_METHOD_VERSION,
        temperature=LOCKED_BETA_TEMPERATURE,
    )
    return BetaNormalizationState(
        nutrient_order=nutrient_order,
        median_values=median_values,
        scale_values=scale_values,
        scale_method_values=scale_method_values,
        fit_n=len(development_beta),
        fit_id_sha256=fit_id_sha256,
        method_version=BETA_NORMALIZATION_METHOD_VERSION,
        temperature=LOCKED_BETA_TEMPERATURE,
        state_fingerprint=fingerprint,
    )


def transform_beta(state: BetaNormalizationState, beta: pd.DataFrame) -> pd.DataFrame:
    """Apply only validated, frozen development parameters to raw beta."""

    _validate_normalization_state(state)
    beta = _require_frame(beta, "beta")
    expected = set(state.nutrient_order)
    supplied = set(beta.columns)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if extra:
            details.append(f"extra columns: {extra}")
        raise ValueError("beta columns do not match frozen state; " + "; ".join(details))

    aligned = beta.loc[:, state.nutrient_order]
    values = _finite_numeric_frame(aligned, "beta")
    medians = np.asarray(state.median_values, dtype=float)
    scales = np.asarray(state.scale_values, dtype=float)
    robust_z = (values - medians) / scales
    transformed = np.tanh(robust_z / LOCKED_BETA_TEMPERATURE)
    return pd.DataFrame(transformed, index=beta.index.copy(), columns=list(state.nutrient_order))


def _mapping_rows(mapping_version: str):
    if mapping_version == PRIMARY_MAPPING_VERSION:
        return PRIMARY_ATTRIBUTE_MAPPINGS
    try:
        return SENSITIVITY_ATTRIBUTE_MAPPINGS[mapping_version]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown reviewed mapping version: {mapping_version}") from error


def _reviewed_targets(mapping_version: str) -> tuple[str, ...]:
    rows = _mapping_rows(mapping_version)
    target_set = {row.attribute for row in rows if row.role == "effect"}
    return tuple(attribute for attribute in _ACTIVE_ATTRIBUTES if attribute in target_set)


def _response_fingerprint(
    normalization_fingerprint: str,
    mapping_version: str,
    reviewed_targets: tuple[str, ...],
    responses: pd.DataFrame,
) -> str:
    return _sha256_payload(
        {
            "columns": list(responses.columns),
            "index": [list(index_value) for index_value in responses.index],
            "mapping_version": mapping_version,
            "normalization_fingerprint": normalization_fingerprint,
            "responses": responses.to_numpy(dtype=float).tolist(),
            "reviewed_targets": list(reviewed_targets),
        }
    )


def _validate_response_result(result: AttributeResponseResult) -> None:
    if not isinstance(result, AttributeResponseResult):
        raise TypeError("attribute_responses must be an AttributeResponseResult")
    _validate_normalization_state(result.normalization_state)
    if result.normalization_fingerprint != result.normalization_state.state_fingerprint:
        raise ValueError("normalization fingerprint does not match the validated state")
    expected_targets = _reviewed_targets(result.mapping_version)
    if result.reviewed_targets != expected_targets:
        raise ValueError("reviewed target set does not match the named mapping version")

    responses = _require_frame(result.responses, "attribute_responses.responses")
    if not isinstance(responses.index, pd.MultiIndex) or responses.index.nlevels != 2:
        raise ValueError("response matrix must use a two-level individual_id, food_id index")
    if list(responses.index.names) != ["individual_id", "food_id"]:
        raise ValueError("response matrix index levels must be named individual_id and food_id")
    if tuple(responses.columns) != _ACTIVE_ATTRIBUTES:
        raise ValueError("response matrix must contain the exact active Food Compass attributes")
    response_values = _finite_numeric_frame(responses, "attribute_responses.responses")
    outside_targets = [
        attribute
        for attribute in _ACTIVE_ATTRIBUTES
        if attribute not in expected_targets
        and np.any(response_values[:, responses.columns.get_loc(attribute)] != 0.0)
    ]
    if outside_targets:
        raise ValueError(
            "nonzero response outside reviewed target set is prohibited: " f"{outside_targets}"
        )
    expected_fingerprint = _response_fingerprint(
        result.normalization_fingerprint,
        result.mapping_version,
        result.reviewed_targets,
        responses,
    )
    if not _is_sha256(result.response_fingerprint) or result.response_fingerprint != expected_fingerprint:
        raise ValueError("response fingerprint does not match response provenance and values")


def attribute_response(
    state: BetaNormalizationState,
    beta: pd.DataFrame,
    food_exposures: pd.DataFrame,
    *,
    mapping_version: str = PRIMARY_MAPPING_VERSION,
) -> AttributeResponseResult:
    """Transform raw beta and build provenance-bound individual-food responses."""

    _validate_normalization_state(state)
    transformed_beta = transform_beta(state, beta)
    food_exposures = _require_frame(food_exposures, "food_exposures")
    reviewed_targets = _reviewed_targets(mapping_version)
    if food_exposures.attrs.get("basis") != "per_100_kcal":
        raise ValueError("food_exposures must declare attrs['basis'] = 'per_100_kcal'")

    response_rows: list[pd.Series] = []
    response_keys: list[tuple[object, object]] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for individual_id, beta_row in transformed_beta.iterrows():
        for food_id, exposure_row in food_exposures.iterrows():
            exposure_row.attrs["basis"] = "per_100_kcal"
            built = build_food_specific_response(beta_row, exposure_row, mapping=mapping_version)
            response_rows.append(built.response)
            response_keys.append((individual_id, food_id))
            diagnostics = built.diagnostics.copy()
            diagnostics.insert(0, "food_id", food_id)
            diagnostics.insert(0, "individual_id", individual_id)
            diagnostics["mapping_version"] = mapping_version
            diagnostics["normalization_fingerprint"] = state.state_fingerprint
            diagnostic_frames.append(diagnostics)

    response_index = pd.MultiIndex.from_tuples(
        response_keys, names=["individual_id", "food_id"]
    )
    responses = pd.DataFrame(response_rows, index=response_index).loc[:, _ACTIVE_ATTRIBUTES]
    response_fingerprint = _response_fingerprint(
        state.state_fingerprint,
        mapping_version,
        reviewed_targets,
        responses,
    )
    result = AttributeResponseResult(
        normalization_state=state,
        normalization_fingerprint=state.state_fingerprint,
        mapping_version=mapping_version,
        reviewed_targets=reviewed_targets,
        responses=responses,
        response_fingerprint=response_fingerprint,
        diagnostics=pd.concat(diagnostic_frames, ignore_index=True),
    )
    _validate_response_result(result)
    return result


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _exact_bool(value: object, expected: bool) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value) is expected


def _numeric_nan(value: object) -> bool:
    return isinstance(value, (float, np.floating)) and np.isnan(float(value))


def validate_calibration_result(
    result: AttributeCalibrationResult,
) -> AttributeCalibrationResult:
    """Strictly validate a Task 3 result and its complete provenance chain."""

    if not isinstance(result, AttributeCalibrationResult):
        raise TypeError("result must be an AttributeCalibrationResult")
    if not _is_sha256(result.normalization_fingerprint):
        raise ValueError("calibration normalization fingerprint must be a SHA-256 digest")
    if not _is_sha256(result.response_fingerprint):
        raise ValueError("calibration response fingerprint must be a SHA-256 digest")
    if not isinstance(result.mapping_version, str):
        raise ValueError("calibration mapping version must be a reviewed version")
    expected_targets = _reviewed_targets(result.mapping_version)
    if (
        not isinstance(result.reviewed_targets, tuple)
        or result.reviewed_targets != expected_targets
    ):
        raise ValueError("calibration reviewed target set does not match its mapping version")
    if (
        not isinstance(result.calibration_mode, str)
        or result.calibration_mode not in ATTRIBUTE_POINT_FRACTION_MODES
    ):
        raise ValueError("calibration mode is not a locked named mode")
    expected_fraction = float(ATTRIBUTE_POINT_FRACTION_MODES[result.calibration_mode])
    fraction = _finite_number(result.calibration_fraction, "calibration fraction")
    if fraction != expected_fraction:
        raise ValueError("calibration fraction does not match its named mode")

    points = _require_frame(result.points, "calibration points")
    deltas = _require_frame(result.deltas, "calibration deltas")
    diagnostics = _require_frame(result.diagnostics, "calibration diagnostics")
    if not isinstance(points.index, pd.MultiIndex) or points.index.nlevels != 2:
        raise ValueError("calibration points must use an individual_id, food_id index")
    if tuple(points.index.names) != ("individual_id", "food_id"):
        raise ValueError("calibration point index labels are not canonical")
    if not points.index.equals(deltas.index):
        raise ValueError("calibration point and delta row labels must be identical")
    if tuple(points.columns) != _ACTIVE_ATTRIBUTES:
        raise ValueError("calibration points must contain the exact active attribute labels")
    if not points.columns.equals(deltas.columns):
        raise ValueError("calibration point and delta attribute labels must be identical")
    for individual_id, food_id in points.index:
        if not isinstance(individual_id, str) or not individual_id:
            raise ValueError("calibration individual IDs must be nonempty strings")
        if not isinstance(food_id, str) or not food_id:
            raise ValueError("calibration food IDs must be nonempty strings")

    if not isinstance(diagnostics.index, pd.MultiIndex) or diagnostics.index.nlevels != 3:
        raise ValueError("calibration diagnostics must use a three-level canonical index")
    expected_diagnostic_index = pd.MultiIndex.from_tuples(
        [
            (individual_id, food_id, attribute)
            for individual_id, food_id in points.index
            for attribute in _ACTIVE_ATTRIBUTES
        ],
        names=["individual_id", "food_id", "attribute"],
    )
    if not diagnostics.index.equals(expected_diagnostic_index):
        raise ValueError("calibration diagnostics row labels are not canonical")
    if tuple(diagnostics.columns) != _CALIBRATION_DIAGNOSTIC_COLUMNS:
        raise ValueError("calibration diagnostics columns are not canonical")

    for individual_id, food_id in points.index:
        for attribute in _ACTIVE_ATTRIBUTES:
            key = (individual_id, food_id, attribute)
            point = points.loc[(individual_id, food_id), attribute]
            delta = deltas.loc[(individual_id, food_id), attribute]
            diagnostic = diagnostics.loc[key]
            rule = FCS2_RULES[attribute]
            lower = min(float(rule.low_points), float(rule.high_points))
            upper = max(float(rule.low_points), float(rule.high_points))
            expected_lambda = expected_fraction * (upper - lower)
            if diagnostic["mode"] != result.calibration_mode:
                raise ValueError("calibration diagnostic mode is mixed or inconsistent")
            numeric_metadata = {
                "lower_bound": lower,
                "upper_bound": upper,
                "lambda_points": expected_lambda,
                "fraction_cap": expected_fraction,
            }
            for column, expected in numeric_metadata.items():
                actual = _finite_number(
                    diagnostic[column], f"calibration diagnostic {column}"
                )
                if actual != expected:
                    raise ValueError(
                        f"calibration diagnostic {column} is inconsistent with published bounds"
                    )

            if point is NOT_CALCULATED or delta is NOT_CALCULATED:
                if point is not NOT_CALCULATED or delta is not NOT_CALCULATED:
                    raise ValueError("calibration NOT_CALCULATED point and delta must match")
                if rule.kind != "log_ratio":
                    raise ValueError("only ratio attributes may be NOT_CALCULATED")
                for column in (
                    "baseline_points",
                    "attribute_response",
                    "raw_delta",
                    "point_delta",
                ):
                    if not _numeric_nan(diagnostic[column]):
                        raise ValueError(
                            "calibration NOT_CALCULATED numeric diagnostics must be NaN"
                        )
                if not _exact_bool(diagnostic["clipped"], False):
                    raise ValueError("calibration NOT_CALCULATED clipped must be exactly False")
                if not _exact_bool(diagnostic["not_calculated"], True):
                    raise ValueError(
                        "calibration NOT_CALCULATED flag must be exactly True"
                    )
                continue

            numeric_point = _finite_number(point, "calibration point")
            numeric_delta = _finite_number(delta, "calibration delta")
            baseline = _finite_number(
                diagnostic["baseline_points"], "calibration diagnostic baseline_points"
            )
            response = _finite_number(
                diagnostic["attribute_response"],
                "calibration diagnostic attribute_response",
            )
            raw_delta = _finite_number(
                diagnostic["raw_delta"], "calibration diagnostic raw_delta"
            )
            point_delta = _finite_number(
                diagnostic["point_delta"], "calibration diagnostic point_delta"
            )
            if not lower <= baseline <= upper:
                raise ValueError("calibration diagnostic baseline is outside published bounds")
            expected_raw_delta = expected_lambda * float(
                np.tanh(response / LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE)
            )
            candidate = baseline + expected_raw_delta
            expected_point = min(max(candidate, lower), upper)
            expected_delta = expected_point - baseline
            if not all(
                np.isclose(actual, expected, atol=1e-12, rtol=0.0)
                for actual, expected in (
                    (raw_delta, expected_raw_delta),
                    (point_delta, expected_delta),
                    (numeric_point, expected_point),
                    (numeric_delta, expected_delta),
                )
            ):
                raise ValueError("calibration points, deltas, and diagnostics are inconsistent")
            if not _exact_bool(diagnostic["clipped"], expected_point != candidate):
                raise ValueError("calibration clipped diagnostic is not canonical")
            if not _exact_bool(diagnostic["not_calculated"], False):
                raise ValueError("calibration not_calculated diagnostic must be exactly False")
            if attribute not in result.reviewed_targets and response != 0.0:
                raise ValueError(
                    "nonzero calibration response outside reviewed target set is prohibited: "
                    f"{attribute}"
                )

    expected_fingerprint = _calibration_fingerprint(
        points=points,
        deltas=deltas,
        diagnostics=diagnostics,
        normalization_fingerprint=result.normalization_fingerprint,
        mapping_version=result.mapping_version,
        reviewed_targets=result.reviewed_targets,
        response_fingerprint=result.response_fingerprint,
        calibration_mode=result.calibration_mode,
        calibration_fraction=fraction,
    )
    if (
        not _is_sha256(result.calibration_fingerprint)
        or result.calibration_fingerprint != expected_fingerprint
    ):
        raise ValueError("calibration fingerprint does not match complete result contents")
    return result


def calibrate_attribute_points(
    baseline_points: pd.DataFrame,
    attribute_responses: AttributeResponseResult,
    *,
    mode: str = "primary",
) -> AttributeCalibrationResult:
    """Apply one locked range-based mode to native Food Compass points."""

    baseline_points = _require_frame(baseline_points, "baseline_points")
    _validate_response_result(attribute_responses)
    responses = attribute_responses.responses
    if baseline_points.index.name != "food_id":
        raise ValueError("baseline_points must be indexed by food_id")
    if not isinstance(mode, str) or mode not in ATTRIBUTE_POINT_FRACTION_MODES:
        raise ValueError(
            "mode must be one of the locked attribute calibration modes: "
            + ", ".join(ATTRIBUTE_POINT_FRACTION_MODES)
        )
    fraction_cap = ATTRIBUTE_POINT_FRACTION_MODES[mode]

    if set(baseline_points.columns) != set(responses.columns):
        raise ValueError("baseline_points must contain the exact response attributes")
    baseline_points = baseline_points.loc[:, _ACTIVE_ATTRIBUTES]
    baseline_foods = set(baseline_points.index)
    response_foods = set(responses.index.get_level_values("food_id"))
    if baseline_foods != response_foods:
        raise ValueError("baseline_points must contain the exact response food IDs")

    point_columns: dict[str, list[object]] = {column: [] for column in baseline_points.columns}
    delta_columns: dict[str, list[object]] = {column: [] for column in baseline_points.columns}
    diagnostic_rows: list[dict[str, object]] = []
    diagnostic_index: list[tuple[object, object, str]] = []

    for individual_id, food_id in responses.index:
        for attribute in baseline_points.columns:
            rule = FCS2_RULES[attribute]
            baseline = baseline_points.loc[food_id, attribute]
            lower_bound = min(float(rule.low_points), float(rule.high_points))
            upper_bound = max(float(rule.low_points), float(rule.high_points))
            lambda_points = fraction_cap * (upper_bound - lower_bound)
            response = responses.loc[(individual_id, food_id), attribute]
            diagnostic_index.append((individual_id, food_id, attribute))

            if isinstance(baseline, NotCalculated):
                point_columns[attribute].append(baseline)
                delta_columns[attribute].append(baseline)
                diagnostic_rows.append(
                    {
                        "baseline_points": np.nan,
                        "attribute_response": np.nan,
                        "raw_delta": np.nan,
                        "point_delta": np.nan,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "lambda_points": lambda_points,
                        "fraction_cap": fraction_cap,
                        "mode": mode,
                        "clipped": False,
                        "not_calculated": True,
                    }
                )
                continue

            numeric_baseline = _finite_number(baseline, f"baseline_points[{food_id}, {attribute}]")
            if not lower_bound <= numeric_baseline <= upper_bound:
                raise ValueError(
                    f"baseline_points[{food_id}, {attribute}] is outside published bounds "
                    f"[{lower_bound}, {upper_bound}]"
                )
            numeric_response = _finite_number(
                response,
                f"attribute_responses[{individual_id}, {food_id}, {attribute}]",
            )
            raw_delta = lambda_points * float(
                np.tanh(numeric_response / LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE)
            )
            candidate = numeric_baseline + raw_delta
            calibrated = min(max(candidate, lower_bound), upper_bound)
            point_delta = calibrated - numeric_baseline
            point_columns[attribute].append(calibrated)
            delta_columns[attribute].append(point_delta)
            diagnostic_rows.append(
                {
                    "baseline_points": numeric_baseline,
                    "attribute_response": numeric_response,
                    "raw_delta": raw_delta,
                    "point_delta": point_delta,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "lambda_points": lambda_points,
                    "fraction_cap": fraction_cap,
                    "mode": mode,
                    "clipped": calibrated != candidate,
                    "not_calculated": False,
                }
            )

    output_index = responses.index.copy()
    points = pd.DataFrame(point_columns, index=output_index)
    deltas = pd.DataFrame(delta_columns, index=output_index)
    diagnostics = pd.DataFrame(
        diagnostic_rows,
        index=pd.MultiIndex.from_tuples(
            diagnostic_index,
            names=["individual_id", "food_id", "attribute"],
        ),
    ).loc[:, _CALIBRATION_DIAGNOSTIC_COLUMNS]
    calibration_fingerprint = _calibration_fingerprint(
        points=points,
        deltas=deltas,
        diagnostics=diagnostics,
        normalization_fingerprint=attribute_responses.normalization_fingerprint,
        mapping_version=attribute_responses.mapping_version,
        reviewed_targets=attribute_responses.reviewed_targets,
        response_fingerprint=attribute_responses.response_fingerprint,
        calibration_mode=mode,
        calibration_fraction=fraction_cap,
    )
    return AttributeCalibrationResult(
        points=points,
        deltas=deltas,
        diagnostics=diagnostics,
        normalization_fingerprint=attribute_responses.normalization_fingerprint,
        mapping_version=attribute_responses.mapping_version,
        reviewed_targets=attribute_responses.reviewed_targets,
        response_fingerprint=attribute_responses.response_fingerprint,
        calibration_mode=mode,
        calibration_fraction=fraction_cap,
        calibration_fingerprint=calibration_fingerprint,
    )
