"""Native Food Compass domain recomposition with a fixed official-score residual."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from gmnps.scoring.attribute_calibration import (
    ATTRIBUTE_POINT_FRACTION_MODES,
    AttributeCalibrationResult,
)
from gmnps.scoring.fcs2_attribute_mapping import PRIMARY_ATTRIBUTE_MAPPINGS
from gmnps.scoring.fcs2_attribute_rules import (
    FCS2_RULES,
    FCS_MAX,
    FCS_MIN,
    NOT_CALCULATED,
    UNSCALED_MAX,
    UNSCALED_MIN,
    NotCalculated,
    aggregate_domains,
    fcs_to_unscaled,
    unscaled_to_fcs,
)


FINAL_DEVIATION_CAP_MODES: Mapping[str, float] = MappingProxyType(
    {"primary": 12.0, "low": 8.0, "high": 15.0}
)
RECOMPOSITION_METHOD_VERSION = "native_domain_fixed_residual_v1"

_ACTIVE_ATTRIBUTES = tuple(name for name, rule in FCS2_RULES.items() if rule.active)
_PERSONALIZED_ATTRIBUTES = frozenset(
    row.attribute for row in PRIMARY_ATTRIBUTE_MAPPINGS if row.role == "effect"
)
_RULE_ORDER = {name: position for position, name in enumerate(FCS2_RULES)}
_DOMAIN_ATTRIBUTES: dict[str, tuple[str, ...]] = {}
for _attribute in _ACTIVE_ATTRIBUTES:
    _domain = FCS2_RULES[_attribute].domain
    _DOMAIN_ATTRIBUTES.setdefault(_domain, tuple())
    _DOMAIN_ATTRIBUTES[_domain] += (_attribute,)
_DOMAIN_ORDER = tuple(_DOMAIN_ATTRIBUTES)
_TOP_K = {"vitamins": 5, "minerals": 5, "specific_lipids": 3}
_ANCHOR_INTERPRETATION = "latent_anchor_from_supplied_official_fcs"
_ANCHOR_KIND = "published_score_implied_latent_unscaled_anchor"
_ANCHOR_NOTE = (
    "The fixed residual preserves the supplied official FCS2. When that score is a "
    "rounded S5 integer, its inverse-scaled value is the implied latent anchor, not the "
    "unpublished raw domain sum."
)
_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX_DIGITS


def _json_value(value: object) -> object:
    if value is NOT_CALCULATED:
        return {"not_calculated": True}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if isfinite(numeric) else None
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def _fingerprint(payload: object) -> str:
    canonical = json.dumps(
        _json_value(payload),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


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


def _require_string_labels(values: tuple[object, ...], label: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must be unique nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _require_food_index(index: pd.Index, label: str) -> tuple[str, ...]:
    if isinstance(index, pd.MultiIndex):
        raise ValueError(f"{label} food IDs must use a one-level index")
    values = tuple(index)
    _require_string_labels(values, f"{label} food IDs")
    return values


def _point_value(value: object, attribute: str, label: str) -> float | NotCalculated:
    rule = FCS2_RULES[attribute]
    if value is NOT_CALCULATED:
        if rule.kind != "log_ratio":
            raise ValueError(f"{label}[{attribute}] can be NOT_CALCULATED only for a ratio")
        return NOT_CALCULATED
    if isinstance(value, NotCalculated):
        raise ValueError(f"{label}[{attribute}] must use the canonical NOT_CALCULATED sentinel")
    numeric = _finite_number(value, f"{label}[{attribute}]")
    lower = min(float(rule.low_points), float(rule.high_points))
    upper = max(float(rule.low_points), float(rule.high_points))
    if not lower <= numeric <= upper:
        raise ValueError(f"{label}[{attribute}] is outside published bounds [{lower}, {upper}]")
    return numeric


def _baseline_payload(
    food_ids: tuple[str, ...],
    official_fcs_values: tuple[float, ...],
    inverse_unscaled_values: tuple[float, ...],
    attribute_labels: tuple[str, ...],
    baseline_attribute_point_values: tuple[tuple[object, ...], ...],
    domain_labels: tuple[str, ...],
    baseline_domain_contribution_values: tuple[tuple[float, ...], ...],
    recomputed_domains: tuple[str, ...],
    fixed_residual_values: tuple[float, ...],
    source_provenance: str,
    anchor_interpretation: str,
) -> dict[str, object]:
    return {
        "anchor_interpretation": anchor_interpretation,
        "attribute_labels": attribute_labels,
        "baseline_attribute_point_values": baseline_attribute_point_values,
        "baseline_domain_contribution_values": baseline_domain_contribution_values,
        "domain_labels": domain_labels,
        "fixed_residual_values": fixed_residual_values,
        "food_ids": food_ids,
        "inverse_unscaled_values": inverse_unscaled_values,
        "method_version": RECOMPOSITION_METHOD_VERSION,
        "official_fcs_values": official_fcs_values,
        "recomputed_domains": recomputed_domains,
        "source_provenance": source_provenance,
    }


@dataclass(frozen=True)
class BaselineDecomposition:
    """Immutable official-score anchor and complete selected-domain baseline."""

    food_ids: tuple[str, ...]
    official_fcs_values: tuple[float, ...]
    inverse_unscaled_values: tuple[float, ...]
    attribute_labels: tuple[str, ...]
    baseline_attribute_point_values: tuple[tuple[object, ...], ...]
    domain_labels: tuple[str, ...]
    baseline_domain_contribution_values: tuple[tuple[float, ...], ...]
    recomputed_domains: tuple[str, ...]
    fixed_residual_values: tuple[float, ...]
    source_provenance: str
    anchor_interpretation: str
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_baseline_decomposition(self)

    @property
    def official_fcs(self) -> pd.Series:
        return pd.Series(
            self.official_fcs_values,
            index=pd.Index(self.food_ids, name="food_id"),
            name="FCS2",
        )

    @property
    def inverse_unscaled(self) -> pd.Series:
        return pd.Series(
            self.inverse_unscaled_values,
            index=pd.Index(self.food_ids, name="food_id"),
            name="inverse_unscaled_baseline",
        )

    @property
    def baseline_points(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.baseline_attribute_point_values,
            index=pd.Index(self.food_ids, name="food_id"),
            columns=self.attribute_labels,
        )

    @property
    def baseline_domain_contributions(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.baseline_domain_contribution_values,
            index=pd.Index(self.food_ids, name="food_id"),
            columns=self.domain_labels,
            dtype=float,
        )

    @property
    def fixed_residual(self) -> pd.Series:
        return pd.Series(
            self.fixed_residual_values,
            index=pd.Index(self.food_ids, name="food_id"),
            name="fixed_residual",
        )

    @property
    def diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "official_fcs": self.official_fcs_values,
                "inverse_unscaled_baseline": self.inverse_unscaled_values,
                "selected_domain_sum": [
                    sum(row) for row in self.baseline_domain_contribution_values
                ],
                "fixed_residual": self.fixed_residual_values,
                "official_score_is_integer": [
                    float(value).is_integer() for value in self.official_fcs_values
                ],
                "official_anchor_kind": _ANCHOR_KIND,
                "anchor_interpretation": self.anchor_interpretation,
                "scientific_boundary_note": _ANCHOR_NOTE,
                "decomposition_fingerprint": self.fingerprint,
            },
            index=pd.Index(self.food_ids, name="food_id"),
        )


def _validate_baseline_decomposition(value: BaselineDecomposition) -> None:
    if not isinstance(value, BaselineDecomposition):
        raise TypeError("baseline must be a BaselineDecomposition")
    tuple_fields = (
        value.food_ids,
        value.official_fcs_values,
        value.inverse_unscaled_values,
        value.attribute_labels,
        value.baseline_attribute_point_values,
        value.domain_labels,
        value.baseline_domain_contribution_values,
        value.recomputed_domains,
        value.fixed_residual_values,
    )
    if any(not isinstance(field, tuple) for field in tuple_fields):
        raise ValueError("baseline decomposition vectors must be immutable tuples")
    _require_string_labels(value.food_ids, "baseline food IDs")
    _require_string_labels(value.attribute_labels, "baseline attribute labels")
    _require_string_labels(value.domain_labels, "baseline domain labels")
    _require_string_labels(value.recomputed_domains, "recomputed domains")
    if value.domain_labels != value.recomputed_domains:
        raise ValueError("baseline domain labels must equal the recomputed domain set")
    if any(domain not in _DOMAIN_ATTRIBUTES for domain in value.recomputed_domains):
        raise ValueError("baseline decomposition contains an unknown domain")
    expected_attributes = tuple(
        attribute
        for attribute in _ACTIVE_ATTRIBUTES
        if FCS2_RULES[attribute].domain in value.recomputed_domains
    )
    if value.attribute_labels != expected_attributes:
        raise ValueError("baseline decomposition attribute labels do not match its domains")
    food_count = len(value.food_ids)
    if any(
        len(vector) != food_count
        for vector in (
            value.official_fcs_values,
            value.inverse_unscaled_values,
            value.baseline_attribute_point_values,
            value.baseline_domain_contribution_values,
            value.fixed_residual_values,
        )
    ):
        raise ValueError("baseline decomposition food vectors have inconsistent lengths")
    for row in value.baseline_attribute_point_values:
        if not isinstance(row, tuple) or len(row) != len(value.attribute_labels):
            raise ValueError("baseline attribute point rows have inconsistent lengths")
    for row in value.baseline_domain_contribution_values:
        if not isinstance(row, tuple) or len(row) != len(value.domain_labels):
            raise ValueError("baseline domain rows have inconsistent lengths")
    for score in value.official_fcs_values:
        numeric = _finite_number(score, "official FCS2")
        if not FCS_MIN <= numeric <= FCS_MAX:
            raise ValueError("official FCS2 must be between 1 and 100")
    for actual, score in zip(value.inverse_unscaled_values, value.official_fcs_values):
        if _finite_number(actual, "inverse unscaled baseline") != fcs_to_unscaled(score):
            raise ValueError("inverse unscaled baseline does not match official FCS2")
    for row in value.baseline_attribute_point_values:
        for attribute, point in zip(value.attribute_labels, row):
            _point_value(point, attribute, "baseline decomposition")
    for food_position, row in enumerate(value.baseline_attribute_point_values):
        scores = dict(zip(value.attribute_labels, row))
        expected_domains = aggregate_domains(scores)
        actual_domains = dict(
            zip(value.domain_labels, value.baseline_domain_contribution_values[food_position])
        )
        if expected_domains.keys() != actual_domains.keys() or any(
            not np.isclose(expected_domains[domain], actual_domains[domain], atol=1e-12, rtol=0.0)
            for domain in expected_domains
        ):
            raise ValueError("baseline domain contributions do not match Task 1 aggregation")
        expected_residual = value.inverse_unscaled_values[food_position] - sum(
            expected_domains.values()
        )
        if not np.isclose(
            expected_residual, value.fixed_residual_values[food_position], atol=1e-12, rtol=0.0
        ):
            raise ValueError("fixed residual does not match the official-score decomposition")
    if value.anchor_interpretation != _ANCHOR_INTERPRETATION:
        raise ValueError("baseline anchor interpretation is not the locked scientific boundary")
    if value.source_provenance != "content_fingerprint_bound":
        raise ValueError("baseline source provenance is not locked")
    expected_fingerprint = _fingerprint(
        _baseline_payload(
            value.food_ids,
            value.official_fcs_values,
            value.inverse_unscaled_values,
            value.attribute_labels,
            value.baseline_attribute_point_values,
            value.domain_labels,
            value.baseline_domain_contribution_values,
            value.recomputed_domains,
            value.fixed_residual_values,
            value.source_provenance,
            value.anchor_interpretation,
        )
    )
    if not _is_sha256(value.fingerprint) or value.fingerprint != expected_fingerprint:
        raise ValueError("baseline decomposition fingerprint does not match its contents")


def decompose_official_baseline(
    official_fcs: pd.Series,
    baseline_points: pd.DataFrame,
    *,
    recomputed_domains: tuple[str, ...] | list[str],
) -> BaselineDecomposition:
    """Bind selected native domains to a fixed residual from supplied official FCS2."""

    if not isinstance(official_fcs, pd.Series) or official_fcs.empty:
        raise TypeError("official_fcs must be a nonempty pandas Series")
    if not isinstance(baseline_points, pd.DataFrame) or baseline_points.empty:
        raise TypeError("baseline_points must be a nonempty pandas DataFrame")
    food_ids = _require_food_index(official_fcs.index, "official FCS2")
    if not official_fcs.index.is_unique:
        raise ValueError("official FCS2 food IDs must be unique")
    baseline_food_ids = _require_food_index(baseline_points.index, "baseline_points")
    if not baseline_points.index.is_unique:
        raise ValueError("baseline_points food IDs must be unique")
    if set(food_ids) != set(baseline_food_ids):
        raise ValueError("baseline_points must contain the exact official food IDs")
    official_values = tuple(
        _finite_number(value, f"official FCS2[{food_id}]")
        for food_id, value in official_fcs.items()
    )
    if any(not FCS_MIN <= value <= FCS_MAX for value in official_values):
        raise ValueError("official FCS2 must be between 1 and 100")

    if not isinstance(recomputed_domains, (tuple, list)) or not recomputed_domains:
        raise ValueError("recomputed_domains must name at least one domain")
    domains = tuple(recomputed_domains)
    _require_string_labels(domains, "recomputed domains")
    unknown = [domain for domain in domains if domain not in _DOMAIN_ATTRIBUTES]
    if unknown:
        raise ValueError(f"unknown recomputed domains: {unknown}")
    ordered_domains = tuple(domain for domain in _DOMAIN_ORDER if domain in domains)
    supplied_columns = tuple(baseline_points.columns)
    _require_string_labels(supplied_columns, "baseline attribute labels")
    expected_attributes = tuple(
        attribute
        for attribute in _ACTIVE_ATTRIBUTES
        if FCS2_RULES[attribute].domain in ordered_domains
    )
    missing = [attribute for attribute in expected_attributes if attribute not in supplied_columns]
    if missing:
        missing_domains = tuple(dict.fromkeys(FCS2_RULES[name].domain for name in missing))
        raise ValueError(
            "baseline_points requires exact active rule labels for every selected domain "
            f"{missing_domains}; missing: {missing}"
        )
    extra = [attribute for attribute in supplied_columns if attribute not in expected_attributes]
    if extra:
        residual_domains = sorted(
            {
                FCS2_RULES[attribute].domain
                for attribute in extra
                if attribute in FCS2_RULES and FCS2_RULES[attribute].active
            }
        )
        if residual_domains:
            raise ValueError(
                "baseline_points includes attributes from a residualized domain: "
                f"{residual_domains}"
            )
        raise ValueError(f"baseline_points contains unknown or inactive labels: {extra}")

    aligned = baseline_points.reindex(index=food_ids, columns=expected_attributes)
    point_rows: list[tuple[object, ...]] = []
    domain_rows: list[tuple[float, ...]] = []
    inverse_values = tuple(fcs_to_unscaled(score) for score in official_values)
    residual_values: list[float] = []
    for food_id, inverse in zip(food_ids, inverse_values):
        row = tuple(
            _point_value(aligned.loc[food_id, attribute], attribute, f"baseline_points[{food_id}]")
            for attribute in expected_attributes
        )
        contributions = aggregate_domains(dict(zip(expected_attributes, row)))
        point_rows.append(row)
        domain_rows.append(tuple(float(contributions[domain]) for domain in ordered_domains))
        residual_values.append(float(inverse - sum(contributions.values())))

    source_provenance = "content_fingerprint_bound"
    payload = _baseline_payload(
        food_ids,
        official_values,
        inverse_values,
        expected_attributes,
        tuple(point_rows),
        ordered_domains,
        tuple(domain_rows),
        ordered_domains,
        tuple(residual_values),
        source_provenance,
        _ANCHOR_INTERPRETATION,
    )
    return BaselineDecomposition(
        food_ids=food_ids,
        official_fcs_values=official_values,
        inverse_unscaled_values=inverse_values,
        attribute_labels=expected_attributes,
        baseline_attribute_point_values=tuple(point_rows),
        domain_labels=ordered_domains,
        baseline_domain_contribution_values=tuple(domain_rows),
        recomputed_domains=ordered_domains,
        fixed_residual_values=tuple(residual_values),
        source_provenance=source_provenance,
        anchor_interpretation=_ANCHOR_INTERPRETATION,
        fingerprint=_fingerprint(payload),
    )


def _records_frame(
    rows: tuple[tuple[object, ...], ...],
    *,
    index_names: tuple[str, ...],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    index_width = len(index_names)
    index = pd.MultiIndex.from_tuples(
        [tuple(row[:index_width]) for row in rows], names=index_names
    )
    return pd.DataFrame(
        [tuple(row[index_width:]) for row in rows], index=index, columns=columns
    )


_MEMBERSHIP_COLUMNS = (
    "domain",
    "baseline_points",
    "personalized_points",
    "calculated",
    "baseline_selected",
    "personalized_selected",
    "published_weight",
    "baseline_active_weight_denominator",
    "active_weight_denominator",
)
_ATTRIBUTE_AUDIT_COLUMNS = (
    "domain",
    "baseline_points",
    "personalized_points",
    "point_delta",
    "calibrated_target",
    "not_calculated",
)


@dataclass(frozen=True)
class PersonalizedDomainRecomposition:
    """Validated selected-domain contributions before final-score composition."""

    individual_food_pairs: tuple[tuple[str, str], ...]
    domain_labels: tuple[str, ...]
    domain_contribution_values: tuple[tuple[float, ...], ...]
    membership_rows: tuple[tuple[object, ...], ...]
    attribute_rows: tuple[tuple[object, ...], ...]
    baseline_fingerprint: str
    calibration_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_domain_recomposition(self)

    @property
    def domain_contributions(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.domain_contribution_values,
            index=pd.MultiIndex.from_tuples(
                self.individual_food_pairs, names=["individual_id", "food_id"]
            ),
            columns=self.domain_labels,
            dtype=float,
        )

    @property
    def membership_audit(self) -> pd.DataFrame:
        return _records_frame(
            self.membership_rows,
            index_names=("individual_id", "food_id", "attribute"),
            columns=_MEMBERSHIP_COLUMNS,
        )

    @property
    def attribute_audit(self) -> pd.DataFrame:
        return _records_frame(
            self.attribute_rows,
            index_names=("individual_id", "food_id", "attribute"),
            columns=_ATTRIBUTE_AUDIT_COLUMNS,
        )


def _recomposition_payload(value: PersonalizedDomainRecomposition) -> dict[str, object]:
    return {
        "attribute_rows": value.attribute_rows,
        "baseline_fingerprint": value.baseline_fingerprint,
        "calibration_fingerprint": value.calibration_fingerprint,
        "domain_contribution_values": value.domain_contribution_values,
        "domain_labels": value.domain_labels,
        "individual_food_pairs": value.individual_food_pairs,
        "membership_rows": value.membership_rows,
        "method_version": RECOMPOSITION_METHOD_VERSION,
    }


def _validate_domain_recomposition(value: PersonalizedDomainRecomposition) -> None:
    if not isinstance(value, PersonalizedDomainRecomposition):
        raise TypeError("recomputed must be a PersonalizedDomainRecomposition")
    if any(
        not isinstance(field, tuple)
        for field in (
            value.individual_food_pairs,
            value.domain_labels,
            value.domain_contribution_values,
            value.membership_rows,
            value.attribute_rows,
        )
    ):
        raise ValueError("domain recomposition data must use immutable tuples")
    if len(set(value.individual_food_pairs)) != len(value.individual_food_pairs):
        raise ValueError("domain recomposition individual-food pairs must be unique")
    for pair in value.individual_food_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("domain recomposition requires individual-food pairs")
        _require_string_labels((pair[0],), "recomposition individual IDs")
        _require_string_labels((pair[1],), "recomposition food IDs")
    _require_string_labels(value.domain_labels, "recomposition domain labels")
    if len(value.domain_contribution_values) != len(value.individual_food_pairs):
        raise ValueError("domain contribution row count is inconsistent")
    if any(len(row) != len(value.domain_labels) for row in value.domain_contribution_values):
        raise ValueError("domain contribution columns are inconsistent")
    if any(not isfinite(float(item)) for row in value.domain_contribution_values for item in row):
        raise ValueError("domain contributions must be finite")
    if not _is_sha256(value.baseline_fingerprint) or not _is_sha256(value.calibration_fingerprint):
        raise ValueError("recomposition provenance fingerprints are invalid")
    expected = _fingerprint(_recomposition_payload(value))
    if not _is_sha256(value.fingerprint) or value.fingerprint != expected:
        raise ValueError("domain recomposition fingerprint does not match its contents")


def _membership(
    values: Mapping[str, object], domain: str
) -> tuple[set[str], float]:
    calculated: list[tuple[str, float, float]] = []
    for attribute in _DOMAIN_ATTRIBUTES[domain]:
        value = values[attribute]
        if value is NOT_CALCULATED:
            continue
        calculated.append((attribute, float(value), float(FCS2_RULES[attribute].weight)))
    if domain in _TOP_K:
        selected = sorted(
            calculated, key=lambda item: (-abs(item[1]), _RULE_ORDER[item[0]])
        )[: _TOP_K[domain]]
    else:
        selected = calculated
    return {item[0] for item in selected}, sum(item[2] for item in selected)


def _validate_calibration(
    baseline: BaselineDecomposition,
    calibration: AttributeCalibrationResult,
) -> tuple[tuple[tuple[str, str], ...], str]:
    if not isinstance(calibration, AttributeCalibrationResult):
        raise TypeError("calibration must be an AttributeCalibrationResult from Task 3")
    points, deltas, diagnostics = calibration
    if not isinstance(points, pd.DataFrame) or not isinstance(deltas, pd.DataFrame):
        raise ValueError("calibration points and deltas must be DataFrames")
    if not isinstance(diagnostics, pd.DataFrame):
        raise ValueError("calibration diagnostics must be a DataFrame")
    if not isinstance(points.index, pd.MultiIndex) or points.index.nlevels != 2:
        raise ValueError("calibration points require an individual_id, food_id index")
    if tuple(points.index.names) != ("individual_id", "food_id"):
        raise ValueError("calibration point index provenance labels are invalid")
    if not points.index.is_unique or not points.index.equals(deltas.index):
        raise ValueError("calibration points and deltas require identical unique indices")
    if tuple(points.columns) != _ACTIVE_ATTRIBUTES or not points.columns.equals(deltas.columns):
        raise ValueError("calibration requires the exact active attribute labels")
    pairs = tuple(points.index)
    for individual_id, food_id in pairs:
        _require_string_labels((individual_id,), "calibration individual IDs")
        _require_string_labels((food_id,), "calibration food IDs")
    if set(points.index.get_level_values("food_id")) != set(baseline.food_ids):
        raise ValueError("calibration and decomposition food IDs do not align")
    individuals = tuple(dict.fromkeys(points.index.get_level_values("individual_id")))
    expected_pairs = {(individual, food) for individual in individuals for food in baseline.food_ids}
    if set(pairs) != expected_pairs:
        raise ValueError("calibration must contain every individual-food pair exactly once")
    if not isinstance(diagnostics.index, pd.MultiIndex) or tuple(diagnostics.index.names) != (
        "individual_id",
        "food_id",
        "attribute",
    ):
        raise ValueError("calibration diagnostic provenance labels are invalid")
    expected_diagnostic_index = pd.MultiIndex.from_tuples(
        [(*pair, attribute) for pair in pairs for attribute in _ACTIVE_ATTRIBUTES],
        names=["individual_id", "food_id", "attribute"],
    )
    if not diagnostics.index.equals(expected_diagnostic_index):
        raise ValueError("calibration diagnostics do not align with points")
    required_diagnostics = {
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
    }
    if set(diagnostics.columns) != required_diagnostics:
        raise ValueError("calibration diagnostics have an unexpected schema")

    selected_baseline = baseline.baseline_points
    payload_rows: list[tuple[object, ...]] = []
    for individual_id, food_id in pairs:
        for attribute in _ACTIVE_ATTRIBUTES:
            point = points.loc[(individual_id, food_id), attribute]
            delta = deltas.loc[(individual_id, food_id), attribute]
            diagnostic = diagnostics.loc[(individual_id, food_id, attribute)]
            rule = FCS2_RULES[attribute]
            lower = min(float(rule.low_points), float(rule.high_points))
            upper = max(float(rule.low_points), float(rule.high_points))
            mode = diagnostic["mode"]
            if not isinstance(mode, str) or mode not in ATTRIBUTE_POINT_FRACTION_MODES:
                raise ValueError("calibration mode provenance is not locked")
            fraction = float(ATTRIBUTE_POINT_FRACTION_MODES[mode])
            expected_lambda = fraction * (upper - lower)
            if not (
                np.isclose(float(diagnostic["lower_bound"]), lower, atol=1e-12, rtol=0.0)
                and np.isclose(float(diagnostic["upper_bound"]), upper, atol=1e-12, rtol=0.0)
                and np.isclose(float(diagnostic["fraction_cap"]), fraction, atol=1e-12, rtol=0.0)
                and np.isclose(float(diagnostic["lambda_points"]), expected_lambda, atol=1e-12, rtol=0.0)
            ):
                raise ValueError("calibration bound provenance is inconsistent")
            if bool(diagnostic["not_calculated"]):
                if point is not NOT_CALCULATED or delta is not NOT_CALCULATED:
                    raise ValueError("calibration NOT_CALCULATED values are inconsistent")
                if attribute in baseline.attribute_labels and selected_baseline.loc[food_id, attribute] is not NOT_CALCULATED:
                    raise ValueError("calibration baseline provenance is inconsistent")
                payload_rows.append((individual_id, food_id, attribute, NOT_CALCULATED, NOT_CALCULATED, mode))
                continue
            numeric_point = _point_value(point, attribute, "calibration points")
            numeric_delta = _finite_number(delta, "calibration point delta")
            numeric_baseline = _finite_number(
                diagnostic["baseline_points"], "calibration diagnostic baseline"
            )
            response = _finite_number(
                diagnostic["attribute_response"], "calibration attribute response"
            )
            raw_delta = expected_lambda * float(np.tanh(response / 2.0))
            expected_point = min(max(numeric_baseline + raw_delta, lower), upper)
            expected_delta = expected_point - numeric_baseline
            consistent = (
                np.isclose(float(diagnostic["raw_delta"]), raw_delta, atol=1e-12, rtol=0.0)
                and np.isclose(float(diagnostic["point_delta"]), expected_delta, atol=1e-12, rtol=0.0)
                and np.isclose(numeric_point, expected_point, atol=1e-12, rtol=0.0)
                and np.isclose(numeric_delta, expected_delta, atol=1e-12, rtol=0.0)
                and bool(diagnostic["clipped"]) == (expected_point != numeric_baseline + raw_delta)
            )
            if not consistent:
                raise ValueError("calibration values are inconsistent; possible tampering")
            domain = rule.domain
            if (
                attribute not in _PERSONALIZED_ATTRIBUTES
                and not np.isclose(numeric_delta, 0.0, atol=1e-12, rtol=0.0)
            ):
                raise ValueError(
                    f"nonpersonalized attribute {attribute} must stay at its baseline"
                )
            if attribute in baseline.attribute_labels:
                baseline_point = selected_baseline.loc[food_id, attribute]
                if baseline_point is NOT_CALCULATED or not np.isclose(
                    numeric_baseline, float(baseline_point), atol=1e-12, rtol=0.0
                ):
                    raise ValueError("calibration and decomposition baseline provenance do not align")
            elif not np.isclose(numeric_delta, 0.0, atol=1e-12, rtol=0.0):
                raise ValueError(
                    "calibration targets a residualized domain "
                    f"{domain}: attribute {attribute}"
                )
            payload_rows.append(
                (individual_id, food_id, attribute, numeric_point, numeric_delta, numeric_baseline, response, mode)
            )
    return pairs, _fingerprint({"rows": tuple(payload_rows), "type": "AttributeCalibrationResult"})


def recompute_personalized_domains(
    baseline: BaselineDecomposition,
    calibration: AttributeCalibrationResult,
) -> PersonalizedDomainRecomposition:
    """Recompute all selected domains, including dynamic top-k membership."""

    _validate_baseline_decomposition(baseline)
    pairs, calibration_fingerprint = _validate_calibration(baseline, calibration)
    baseline_points = baseline.baseline_points
    point_frame = calibration.points
    contribution_rows: list[tuple[float, ...]] = []
    membership_rows: list[tuple[object, ...]] = []
    attribute_rows: list[tuple[object, ...]] = []
    for individual_id, food_id in pairs:
        baseline_values = {
            attribute: baseline_points.loc[food_id, attribute]
            for attribute in baseline.attribute_labels
        }
        personalized_values = {
            attribute: point_frame.loc[(individual_id, food_id), attribute]
            for attribute in baseline.attribute_labels
        }
        contributions = aggregate_domains(personalized_values)
        contribution_rows.append(
            tuple(float(contributions[domain]) for domain in baseline.domain_labels)
        )
        for domain in baseline.domain_labels:
            domain_attributes = _DOMAIN_ATTRIBUTES[domain]
            baseline_selected, baseline_denominator = _membership(baseline_values, domain)
            personalized_selected, personalized_denominator = _membership(
                personalized_values, domain
            )
            for attribute in domain_attributes:
                baseline_point = baseline_values[attribute]
                personalized_point = personalized_values[attribute]
                calculated = personalized_point is not NOT_CALCULATED
                weight = float(FCS2_RULES[attribute].weight)
                membership_rows.append(
                    (
                        individual_id,
                        food_id,
                        attribute,
                        domain,
                        baseline_point,
                        personalized_point,
                        calculated,
                        attribute in baseline_selected,
                        attribute in personalized_selected,
                        weight,
                        float(baseline_denominator),
                        float(personalized_denominator),
                    )
                )
                if calculated:
                    delta = float(personalized_point) - float(baseline_point)
                    calibrated_target = not np.isclose(delta, 0.0, atol=1e-12, rtol=0.0)
                else:
                    delta = NOT_CALCULATED
                    calibrated_target = False
                attribute_rows.append(
                    (
                        individual_id,
                        food_id,
                        attribute,
                        domain,
                        baseline_point,
                        personalized_point,
                        delta,
                        calibrated_target,
                        not calculated,
                    )
                )
    result = object.__new__(PersonalizedDomainRecomposition)
    fields = {
        "individual_food_pairs": pairs,
        "domain_labels": baseline.domain_labels,
        "domain_contribution_values": tuple(contribution_rows),
        "membership_rows": tuple(membership_rows),
        "attribute_rows": tuple(attribute_rows),
        "baseline_fingerprint": baseline.fingerprint,
        "calibration_fingerprint": calibration_fingerprint,
    }
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, "fingerprint", "")
    object.__setattr__(result, "fingerprint", _fingerprint(_recomposition_payload(result)))
    _validate_domain_recomposition(result)
    return result


_DOMAIN_AUDIT_COLUMNS = (
    "baseline_contribution",
    "personalized_contribution",
    "domain_delta",
)
_COMPOSITION_DIAGNOSTIC_COLUMNS = (
    "official_fcs",
    "fixed_residual",
    "selected_domain_sum",
    "unscaled_raw",
    "unscaled_clipped",
    "native_truncated",
    "native_fcs",
    "uncapped_delta",
    "cap_mode",
    "deviation_cap",
    "deviation_cap_applied",
    "final_delta",
    "anchor_interpretation",
)


@dataclass(frozen=True)
class PersonalizedFCSResult:
    """Final capped scores plus complete native-domain and attribute audits."""

    individual_food_pairs: tuple[tuple[str, str], ...]
    score_values: tuple[float, ...]
    delta_values: tuple[float, ...]
    unscaled_raw_values: tuple[float, ...]
    domain_audit_rows: tuple[tuple[object, ...], ...]
    attribute_audit_rows: tuple[tuple[object, ...], ...]
    diagnostic_rows: tuple[tuple[object, ...], ...]
    baseline_fingerprint: str
    recomposition_fingerprint: str
    cap_mode: str
    fingerprint: str

    @property
    def scores(self) -> pd.Series:
        return pd.Series(
            self.score_values,
            index=pd.MultiIndex.from_tuples(
                self.individual_food_pairs, names=["individual_id", "food_id"]
            ),
            name="personalized_fcs",
        )

    @property
    def deltas(self) -> pd.Series:
        return pd.Series(
            self.delta_values,
            index=pd.MultiIndex.from_tuples(
                self.individual_food_pairs, names=["individual_id", "food_id"]
            ),
            name="personalized_delta",
        )

    @property
    def unscaled_raw(self) -> pd.Series:
        return pd.Series(
            self.unscaled_raw_values,
            index=pd.MultiIndex.from_tuples(
                self.individual_food_pairs, names=["individual_id", "food_id"]
            ),
            name="unscaled_raw",
        )

    @property
    def domain_audit(self) -> pd.DataFrame:
        return _records_frame(
            self.domain_audit_rows,
            index_names=("individual_id", "food_id", "domain"),
            columns=_DOMAIN_AUDIT_COLUMNS,
        )

    @property
    def attribute_audit(self) -> pd.DataFrame:
        return _records_frame(
            self.attribute_audit_rows,
            index_names=("individual_id", "food_id", "attribute"),
            columns=_ATTRIBUTE_AUDIT_COLUMNS,
        )

    @property
    def diagnostics(self) -> pd.DataFrame:
        return _records_frame(
            self.diagnostic_rows,
            index_names=("individual_id", "food_id"),
            columns=_COMPOSITION_DIAGNOSTIC_COLUMNS,
        )


def _final_payload(value: PersonalizedFCSResult) -> dict[str, object]:
    return {
        "attribute_audit_rows": value.attribute_audit_rows,
        "baseline_fingerprint": value.baseline_fingerprint,
        "cap_mode": value.cap_mode,
        "delta_values": value.delta_values,
        "diagnostic_rows": value.diagnostic_rows,
        "domain_audit_rows": value.domain_audit_rows,
        "individual_food_pairs": value.individual_food_pairs,
        "method_version": RECOMPOSITION_METHOD_VERSION,
        "recomposition_fingerprint": value.recomposition_fingerprint,
        "score_values": value.score_values,
        "unscaled_raw_values": value.unscaled_raw_values,
    }


def compose_personalized_fcs(
    baseline: BaselineDecomposition,
    recomputed: PersonalizedDomainRecomposition,
    *,
    cap_mode: str = "primary",
) -> PersonalizedFCSResult:
    """Compose native selected domains with the fixed residual and named final cap."""

    _validate_baseline_decomposition(baseline)
    _validate_domain_recomposition(recomputed)
    if recomputed.baseline_fingerprint != baseline.fingerprint:
        raise ValueError("recomposition provenance fingerprint does not match baseline")
    if recomputed.domain_labels != baseline.domain_labels:
        raise ValueError("recomposition domains do not match baseline provenance")
    if not isinstance(cap_mode, str) or cap_mode not in FINAL_DEVIATION_CAP_MODES:
        raise ValueError(
            "cap_mode must be one of the locked final deviation modes: "
            + ", ".join(FINAL_DEVIATION_CAP_MODES)
        )
    cap = FINAL_DEVIATION_CAP_MODES[cap_mode]
    contributions = recomputed.domain_contributions
    baseline_domains = baseline.baseline_domain_contributions
    official = baseline.official_fcs
    residual = baseline.fixed_residual
    score_values: list[float] = []
    delta_values: list[float] = []
    raw_values: list[float] = []
    domain_rows: list[tuple[object, ...]] = []
    diagnostic_rows: list[tuple[object, ...]] = []
    for individual_id, food_id in recomputed.individual_food_pairs:
        domain_sum = float(contributions.loc[(individual_id, food_id)].sum())
        unscaled_raw = float(residual.loc[food_id] + domain_sum)
        unscaled_clipped = min(max(unscaled_raw, UNSCALED_MIN), UNSCALED_MAX)
        native_fcs = float(unscaled_to_fcs(unscaled_raw))
        official_fcs = float(official.loc[food_id])
        uncapped_delta = native_fcs - official_fcs
        capped_delta = min(max(uncapped_delta, -cap), cap)
        final_score = min(max(official_fcs + capped_delta, FCS_MIN), FCS_MAX)
        final_delta = final_score - official_fcs
        raw_values.append(unscaled_raw)
        score_values.append(float(final_score))
        delta_values.append(float(final_delta))
        for domain in baseline.domain_labels:
            baseline_contribution = float(baseline_domains.loc[food_id, domain])
            personalized_contribution = float(
                contributions.loc[(individual_id, food_id), domain]
            )
            domain_rows.append(
                (
                    individual_id,
                    food_id,
                    domain,
                    baseline_contribution,
                    personalized_contribution,
                    personalized_contribution - baseline_contribution,
                )
            )
        diagnostic_rows.append(
            (
                individual_id,
                food_id,
                official_fcs,
                float(residual.loc[food_id]),
                domain_sum,
                unscaled_raw,
                unscaled_clipped,
                unscaled_raw != unscaled_clipped,
                native_fcs,
                uncapped_delta,
                cap_mode,
                cap,
                capped_delta != uncapped_delta,
                final_delta,
                baseline.anchor_interpretation,
            )
        )
    result = PersonalizedFCSResult(
        individual_food_pairs=recomputed.individual_food_pairs,
        score_values=tuple(score_values),
        delta_values=tuple(delta_values),
        unscaled_raw_values=tuple(raw_values),
        domain_audit_rows=tuple(domain_rows),
        attribute_audit_rows=recomputed.attribute_rows,
        diagnostic_rows=tuple(diagnostic_rows),
        baseline_fingerprint=baseline.fingerprint,
        recomposition_fingerprint=recomputed.fingerprint,
        cap_mode=cap_mode,
        fingerprint="0" * 64,
    )
    object.__setattr__(result, "fingerprint", _fingerprint(_final_payload(result)))
    return result
