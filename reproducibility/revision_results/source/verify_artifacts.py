"""Verify completeness, internal consistency, and hashes of synchronized experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_run_ids(config: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for domain in config["domains"]:
        for n_value in config["cardinality_caps"]:
            n = int(n_value)
            for offset_value in config["width_offsets"]:
                width = 2 * n + int(offset_value)
                for family in config["encoder_families"]:
                    for seed_value in config["seeds"]:
                        values.append(
                            f"{family}__{domain}__n{n}__m{width}__seed{int(seed_value)}"
                        )
    return sorted(values)


def close(first: float, second: float, tolerance: float = 2e-6) -> bool:
    return math.isclose(float(first), float(second), rel_tol=tolerance, abs_tol=tolerance)


def latex_key(value: Any) -> str:
    return str(value).replace("_", "").replace("-", "minus").replace(".", "p")


def verify_domain_points(domain: str, points: np.ndarray, tolerance: float = 2e-5) -> bool:
    if points.size == 0:
        return True
    if domain == "square":
        return bool(np.all(points >= -tolerance) and np.all(points <= 1.0 + tolerance))
    radius = np.linalg.norm(points, axis=-1)
    if domain == "disk":
        return bool(np.all(radius <= 1.0 + tolerance))
    if domain == "annulus":
        # The exact inner radius is checked by the caller against the resolved configuration.
        return bool(np.all(radius <= 1.0 + tolerance))
    return False


def domain_diameter_with_anchor(domain: str, anchor: np.ndarray) -> float:
    anchor64 = np.asarray(anchor, dtype=np.float64)
    if domain == "square":
        corners = np.asarray(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)))
        return max(math.sqrt(2.0), float(np.linalg.norm(corners - anchor64, axis=1).max()))
    if domain in {"disk", "annulus"}:
        return max(2.0, float(np.linalg.norm(anchor64)) + 1.0)
    raise ValueError(f"Unknown domain: {domain}")


def recompute_decoder_squared(
    predicted: np.ndarray,
    target: np.ndarray,
    predicted_sizes: np.ndarray,
    target_sizes: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Recompute exact assignment plus normalized squared cardinality discrepancy."""
    predicted64 = np.asarray(predicted, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    n = int(predicted64.shape[1])
    permutations = np.asarray(list(itertools.permutations(range(n))), dtype=np.int64)
    target_permuted = target64[:, permutations, :]
    difference = predicted64[:, None, :, :] - target_permuted
    geometric = np.sum(difference * difference, axis=(-1, -2)) / (n * radius * radius)
    minimum = geometric.min(axis=1)
    cardinality = (
        (np.asarray(predicted_sizes, dtype=np.float64) - np.asarray(target_sizes, dtype=np.float64))
        / float(n)
    ) ** 2
    return minimum + cardinality


def verify_prediction_archive(
    path: Path,
    record: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    n = int(record["n"])
    width = int(record["width"])
    test_count = int(config["splits"]["test"])
    anchor = np.asarray(config["anchor"], dtype=np.float32)
    required = {
        "target",
        "target_sizes",
        "predicted",
        "predicted_sizes",
        "squared_errors",
        "latent",
        "normalized_nearest",
        "local_sensitivity",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                return [f"{path}: missing arrays {missing}"]
            target = archive["target"]
            target_sizes = archive["target_sizes"]
            predicted = archive["predicted"]
            predicted_sizes = archive["predicted_sizes"]
            squared = archive["squared_errors"]
            latent = archive["latent"]
            nearest = archive["normalized_nearest"]
            sensitivity = archive["local_sensitivity"]
            expected_point_shape = (test_count, n, 2)
            if target.shape != expected_point_shape or predicted.shape != expected_point_shape:
                errors.append(f"{path}: point-array shape mismatch")
            if target_sizes.shape != (test_count,) or predicted_sizes.shape != (test_count,):
                errors.append(f"{path}: cardinality-array shape mismatch")
            if squared.shape != (test_count,) or nearest.shape != (test_count,) or sensitivity.shape != (test_count,):
                errors.append(f"{path}: per-example diagnostic shape mismatch")
            if latent.shape != (test_count, width):
                errors.append(f"{path}: latent shape mismatch")
            if not np.isfinite(target).all() or not np.isfinite(predicted).all() or not np.isfinite(latent).all():
                errors.append(f"{path}: non-finite target, prediction, or latent value")
            if not np.isfinite(squared).all() or np.any(squared < 0.0):
                errors.append(f"{path}: invalid squared error")
            if np.any(target_sizes < 0) or np.any(target_sizes > n):
                errors.append(f"{path}: invalid target cardinality")
            if np.any(predicted_sizes < 0) or np.any(predicted_sizes > n):
                errors.append(f"{path}: invalid predicted cardinality")
            for row, size_value in enumerate(target_sizes.tolist()):
                size = int(size_value)
                if not verify_domain_points(str(record["domain"]), target[row, :size]):
                    errors.append(f"{path}: target point outside domain at row {row}")
                    break
                if str(record["domain"]) == "annulus" and size:
                    inner = float(config["annulus_inner_radius"])
                    if np.any(np.linalg.norm(target[row, :size], axis=-1) < inner - 2e-5):
                        errors.append(f"{path}: target point inside annulus hole at row {row}")
                        break
                if size < n and not np.allclose(target[row, size:], anchor, rtol=0.0, atol=1e-6):
                    errors.append(f"{path}: target padding differs from anchor at row {row}")
                    break
            for row, size_value in enumerate(predicted_sizes.tolist()):
                size = int(size_value)
                if size < n and not np.allclose(predicted[row, size:], anchor, rtol=0.0, atol=1e-6):
                    errors.append(f"{path}: predicted padding differs from anchor at row {row}")
                    break
            if (
                target.shape == expected_point_shape
                and predicted.shape == expected_point_shape
                and target_sizes.shape == (test_count,)
                and predicted_sizes.shape == (test_count,)
                and squared.shape == (test_count,)
                and np.isfinite(target).all()
                and np.isfinite(predicted).all()
            ):
                radius = domain_diameter_with_anchor(str(record["domain"]), anchor)
                independently_squared = recompute_decoder_squared(
                    predicted, target, predicted_sizes, target_sizes, radius
                )
                if not np.allclose(squared, independently_squared, rtol=2e-5, atol=2e-6):
                    maximum = float(np.max(np.abs(squared - independently_squared)))
                    errors.append(
                        f"{path}: saved squared errors fail independent assignment recomputation "
                        f"(maximum difference {maximum})"
                    )
            recomputed_amrmse = float(np.sqrt(np.mean(squared, dtype=np.float64)))
            recomputed_accuracy = float(np.mean(predicted_sizes == target_sizes))
            if not close(recomputed_amrmse, record["test_amrmse"]):
                errors.append(f"{path}: record test_amrmse does not match per-example errors")
            if not close(recomputed_accuracy, record["test_cardinality_accuracy"]):
                errors.append(f"{path}: record cardinality accuracy does not match predictions")
            finite_nearest = int(np.isfinite(nearest).sum())
            if finite_nearest != int(record["normalized_nearest_valid_count"]):
                errors.append(f"{path}: normalized-neighbor valid count mismatch")
            finite_sensitivity = int(np.isfinite(sensitivity).sum())
            if finite_sensitivity != int(record["local_sensitivity_valid_count"]):
                errors.append(f"{path}: local-sensitivity valid count mismatch")
            for quantized in record["quantization"]:
                bits = int(quantized["bits"])
                key = f"quantized_{bits}_squared_errors"
                size_key = f"quantized_{bits}_predicted_sizes"
                prediction_key = f"quantized_{bits}_predicted"
                if (
                    key not in archive.files
                    or size_key not in archive.files
                    or prediction_key not in archive.files
                ):
                    errors.append(f"{path}: missing {key}, {size_key}, or {prediction_key}")
                    continue
                values = archive[key]
                quantized_sizes = archive[size_key]
                quantized_predicted = archive[prediction_key]
                if values.shape != (test_count,) or not np.isfinite(values).all() or np.any(values < 0.0):
                    errors.append(f"{path}: invalid {key}")
                    continue
                if quantized_sizes.shape != (test_count,):
                    errors.append(f"{path}: invalid {size_key}")
                    continue
                if quantized_predicted.shape != expected_point_shape or not np.isfinite(
                    quantized_predicted
                ).all():
                    errors.append(f"{path}: invalid {prediction_key}")
                    continue
                for row, size_value in enumerate(quantized_sizes.tolist()):
                    size = int(size_value)
                    if size < n and not np.allclose(
                        quantized_predicted[row, size:], anchor, rtol=0.0, atol=1e-6
                    ):
                        errors.append(
                            f"{path}: {prediction_key} padding differs from anchor at row {row}"
                        )
                        break
                independently_quantized = recompute_decoder_squared(
                    quantized_predicted,
                    target,
                    quantized_sizes,
                    target_sizes,
                    domain_diameter_with_anchor(str(record["domain"]), anchor),
                )
                if not np.allclose(values, independently_quantized, rtol=2e-5, atol=2e-6):
                    maximum = float(np.max(np.abs(values - independently_quantized)))
                    errors.append(
                        f"{path}: {key} fails independent assignment recomputation "
                        f"(maximum difference {maximum})"
                    )
                quantized_amrmse = float(np.sqrt(np.mean(values, dtype=np.float64)))
                if not close(quantized_amrmse, quantized["amrmse"]):
                    errors.append(f"{path}: {key} does not match record amrmse")
                quantized_accuracy = float(np.mean(quantized_sizes == target_sizes))
                if not close(quantized_accuracy, quantized["cardinality_accuracy"]):
                    errors.append(f"{path}: {size_key} does not match record cardinality accuracy")
    except (OSError, ValueError) as exc:
        errors.append(f"{path}: cannot read archive: {exc}")
    return errors


def verify_run_grid(root: Path, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    digest = config_digest(config)
    expected = expected_run_ids(config)
    expected_set = set(expected)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "runs").glob("*/record.json")):
        try:
            record = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        run_id = str(record.get("run_id", ""))
        if run_id in records:
            errors.append(f"Duplicate run_id {run_id}")
            continue
        records[run_id] = record
        if path.parent.name != run_id:
            errors.append(f"{path}: directory name disagrees with run_id")
        if record.get("config_sha256") != digest:
            errors.append(f"{path}: configuration digest mismatch")
        try:
            family = str(record["family"])
            domain = str(record["domain"])
            n = int(record["n"])
            width = int(record["width"])
            offset = int(record["width_offset"])
            seed = int(record["seed"])
            expected_from_fields = f"{family}__{domain}__n{n}__m{width}__seed{seed}"
            if run_id != expected_from_fields:
                errors.append(f"{path}: run_id disagrees with record fields")
            if family not in config["encoder_families"] or domain not in config["domains"]:
                errors.append(f"{path}: unconfigured family or domain")
            if n not in {int(value) for value in config["cardinality_caps"]}:
                errors.append(f"{path}: unconfigured cardinality cap")
            if offset not in {int(value) for value in config["width_offsets"]} or width != 2 * n + offset:
                errors.append(f"{path}: unconfigured or inconsistent width")
            if seed not in {int(value) for value in config["seeds"]}:
                errors.append(f"{path}: unconfigured seed")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}: invalid run identity fields: {exc}")
        if int(record.get("width", -1)) != 2 * int(record.get("n", 0)) + int(record.get("width_offset", 0)):
            errors.append(f"{path}: inconsistent width and width_offset")
        history = record.get("history", [])
        if len(history) != int(config["training"]["epochs"]):
            errors.append(f"{path}: incomplete training history")
        best_epoch = int(record.get("best_epoch", 0))
        if not 1 <= best_epoch <= len(history):
            errors.append(f"{path}: invalid best_epoch")
        checkpoint = path.parent / "best_checkpoint.pt"
        if config["artifacts"].get("save_checkpoints", True) and (
            not checkpoint.is_file() or checkpoint.stat().st_size == 0
        ):
            errors.append(f"{path.parent}: missing or empty checkpoint")
        prediction_path = path.parent / "test_predictions.npz"
        if config["artifacts"].get("save_per_example_predictions", True):
            if not prediction_path.is_file():
                errors.append(f"{path.parent}: missing test_predictions.npz")
            else:
                errors.extend(verify_prediction_archive(prediction_path, record, config))
    missing = sorted(expected_set - records.keys())
    unexpected = sorted(records.keys() - expected_set)
    if missing:
        errors.append(f"Missing {len(missing)} runs; first entries: {missing[:5]}")
    if unexpected:
        errors.append(f"Unexpected {len(unexpected)} runs; first entries: {unexpected[:5]}")
    index_path = root / "run_index.json"
    if not index_path.is_file():
        errors.append("run_index.json is missing")
    else:
        try:
            index_ids = sorted(str(item["run_id"]) for item in load_json(index_path))
            if index_ids != expected:
                errors.append("run_index.json does not match the expected grid")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"run_index.json is invalid: {exc}")
    return errors


def verify_analytic(root: Path, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    json_path = root / "analytic_checks.json"
    csv_path = root / "analytic_examples.csv"
    fixture_json_path = root / "analytic_fixture_checks.json"
    fixture_csv_path = root / "analytic_fixture_checks.csv"
    if not all(
        path.is_file() for path in (json_path, csv_path, fixture_json_path, fixture_csv_path)
    ):
        return ["One or more random-check or deterministic-fixture analytic artifacts are missing"]
    try:
        checks = load_json(json_path)
        expected_condition_keys = {
            (str(domain), int(n))
            for domain in config["domains"]
            for n in config["cardinality_caps"]
        }
        expected_conditions = len(expected_condition_keys)
        check_lookup: dict[tuple[str, int], dict[str, Any]] = {}
        for check in checks:
            key = (str(check.get("domain")), int(check.get("n", -1)))
            if key in check_lookup:
                errors.append(f"analytic_checks.json duplicates condition {key}")
            check_lookup[key] = check
        if set(check_lookup) != expected_condition_keys:
            errors.append("analytic_checks.json condition keys do not match the configured grid")
        if len(checks) != expected_conditions:
            errors.append("analytic_checks.json has the wrong condition count")
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected_examples = expected_conditions * int(config["splits"]["analytic_test"])
        if len(rows) != expected_examples:
            errors.append(
                f"analytic_examples.csv has {len(rows)} rows; expected {expected_examples}"
            )
        example_lookup: dict[tuple[str, int, int], dict[str, str]] = {}
        for row in rows:
            key = (str(row["domain"]), int(row["n"]), int(row["example_index"]))
            if key in example_lookup:
                errors.append(f"analytic_examples.csv duplicates example {key}")
            example_lookup[key] = row
        expected_example_keys = {
            (domain, n, index)
            for domain, n in expected_condition_keys
            for index in range(int(config["splits"]["analytic_test"]))
        }
        if set(example_lookup) != expected_example_keys:
            errors.append("analytic_examples.csv keys do not match the configured examples")
        for condition, check in check_lookup.items():
            condition_rows = [
                row
                for (domain, n, _), row in example_lookup.items()
                if (domain, n) == condition
            ]
            if int(check.get("sample_count", -1)) != len(condition_rows):
                errors.append(f"analytic sample_count mismatch for {condition}")
                continue
            tolerance = float(check["tolerance"])
            card_count = sum(int(row["cardinality_correct"]) for row in condition_rows)
            tolerance_count = sum(int(row["within_tolerance"]) for row in condition_rows)
            errors_squared = [float(row["normalized_anchor_error"]) ** 2 for row in condition_rows]
            if card_count != int(check["cardinality_correct_count"]):
                errors.append(f"analytic cardinality count mismatch for {condition}")
            if tolerance_count != int(check["within_tolerance_count"]):
                errors.append(f"analytic tolerance count mismatch for {condition}")
            if any(
                int(row["within_tolerance"])
                != int(float(row["normalized_anchor_error"]) <= tolerance)
                for row in condition_rows
            ):
                errors.append(f"analytic within_tolerance flags disagree with errors for {condition}")
            if not close(math.sqrt(float(np.mean(errors_squared))), check["amrmse"]):
                errors.append(f"analytic AMRMSE mismatch for {condition}")
            if not close(
                max(float(row["normalized_anchor_error"]) for row in condition_rows),
                check["maximum_anchor_matching_error"],
            ):
                errors.append(f"analytic maximum error mismatch for {condition}")
        fixtures = load_json(fixture_json_path)
        expected_fixture_count = 4 * expected_conditions
        if len(fixtures) != expected_fixture_count:
            errors.append(
                f"analytic_fixture_checks.json has {len(fixtures)} rows; expected "
                f"{expected_fixture_count}"
            )
        with fixture_csv_path.open("r", encoding="utf-8", newline="") as stream:
            fixture_csv_rows = list(csv.DictReader(stream))
        if len(fixture_csv_rows) != expected_fixture_count:
            errors.append(
                f"analytic_fixture_checks.csv has {len(fixture_csv_rows)} rows; expected "
                f"{expected_fixture_count}"
            )
        fixture_names = {
            "empty",
            "maximal_coincident_multiplicity",
            "two_support_maximal_cardinality",
            "distinct_boundary_maximal_cardinality",
        }
        expected_fixture_keys = {
            (domain, n, fixture)
            for domain, n in expected_condition_keys
            for fixture in fixture_names
        }
        fixture_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
        for fixture in fixtures:
            key = (str(fixture.get("domain")), int(fixture.get("n", -1)), str(fixture.get("fixture")))
            if key in fixture_lookup:
                errors.append(f"analytic_fixture_checks.json duplicates fixture {key}")
            fixture_lookup[key] = fixture
        fixture_csv_lookup: dict[tuple[str, int, str], dict[str, str]] = {}
        for fixture in fixture_csv_rows:
            key = (str(fixture["domain"]), int(fixture["n"]), str(fixture["fixture"]))
            if key in fixture_csv_lookup:
                errors.append(f"analytic_fixture_checks.csv duplicates fixture {key}")
            fixture_csv_lookup[key] = fixture
        if set(fixture_lookup) != expected_fixture_keys:
            errors.append("analytic fixture JSON keys do not match the declared suite")
        if set(fixture_csv_lookup) != expected_fixture_keys:
            errors.append("analytic fixture CSV keys do not match the declared suite")
        for key in sorted(set(fixture_lookup) & set(fixture_csv_lookup)):
            json_row = fixture_lookup[key]
            csv_row = fixture_csv_lookup[key]
            for field in ("true_size", "decoded_size", "cardinality_correct", "within_tolerance"):
                if int(json_row[field]) != int(csv_row[field]):
                    errors.append(f"fixture JSON/CSV {field} mismatch for {key}")
            for field in (
                "normalized_anchor_error",
                "normalized_padded_root_error",
                "maximum_normalized_polynomial_residual",
                "tolerance",
            ):
                if not close(json_row[field], csv_row[field]):
                    errors.append(f"fixture JSON/CSV {field} mismatch for {key}")
            if int(json_row["within_tolerance"]) != int(
                float(json_row["normalized_anchor_error"]) <= float(json_row["tolerance"])
            ):
                errors.append(f"fixture tolerance flag disagrees with error for {key}")
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        errors.append(f"Cannot read analytic artifacts: {exc}")
    return errors


def verify_aggregate(root: Path, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    aggregate = root / "aggregate"
    required = (
        "summary.csv",
        "pooled_by_width_offset.csv",
        "quantization_summary.csv",
        "analytic_checks.csv",
        "analytic_fixture_checks.csv",
        "aggregate_metadata.json",
        "manuscript_macros.tex",
    )
    for name in required:
        path = aggregate / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty aggregate/{name}")
    expected_condition_rows = (
        len(config["encoder_families"])
        * len(config["domains"])
        * len(config["cardinality_caps"])
        * len(config["width_offsets"])
    )
    expected_pooled_rows = len(config["encoder_families"]) * len(config["width_offsets"])
    expected_quantization_rows = expected_condition_rows * len(config["diagnostics"]["quantization_bits"])
    expected_analytic_rows = len(config["domains"]) * len(config["cardinality_caps"])
    expected_fixture_rows = 4 * expected_analytic_rows
    expected_csv_rows = {
        "summary.csv": expected_condition_rows,
        "pooled_by_width_offset.csv": expected_pooled_rows,
        "quantization_summary.csv": expected_quantization_rows,
        "analytic_checks.csv": expected_analytic_rows,
        "analytic_fixture_checks.csv": expected_fixture_rows,
    }
    if len(config["encoder_families"]) == 2:
        expected_csv_rows["paired_family_differences.csv"] = (
            len(config["domains"])
            * len(config["cardinality_caps"])
            * len(config["width_offsets"])
        )
    for name, expected_count in expected_csv_rows.items():
        path = aggregate / name
        if not path.is_file():
            errors.append(f"Missing aggregate/{name}")
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                actual_count = sum(1 for _ in csv.DictReader(stream))
            if actual_count != expected_count:
                errors.append(
                    f"aggregate/{name} has {actual_count} rows; expected {expected_count}"
                )
        except (OSError, csv.Error) as exc:
            errors.append(f"Cannot parse aggregate/{name}: {exc}")
    metadata_path = aggregate / "aggregate_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = load_json(metadata_path)
            expected_count = len(expected_run_ids(config))
            if int(metadata.get("actual_learned_run_count", -1)) != expected_count:
                errors.append("aggregate metadata has the wrong learned-run count")
            if metadata.get("config_sha256") != config_digest(config):
                errors.append("aggregate metadata configuration digest mismatch")
            expected_fixtures = 4 * len(config["domains"]) * len(config["cardinality_caps"])
            if int(metadata.get("fixture_count", -1)) != expected_fixtures:
                errors.append("aggregate metadata has the wrong deterministic-fixture count")
            snapshotted_aggregator = root / "source" / "aggregate_results.py"
            if snapshotted_aggregator.is_file() and metadata.get("aggregator_sha256") != sha256_file(
                snapshotted_aggregator
            ):
                errors.append("aggregate metadata aggregator hash mismatch")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"Invalid aggregate metadata: {exc}")
    macros_path = aggregate / "manuscript_macros.tex"
    if macros_path.is_file():
        macro_text = macros_path.read_text(encoding="utf-8")
        required_macro_keys = {
            "NN:run-count",
            "NN:analytic-example-count",
            "NN:analytic-within-tolerance-count",
            "NN:analytic-cardinality-correct-count",
            "NN:analytic-maximum-anchor-matching-error",
            "NN:fixture-count",
            "NN:fixture-within-tolerance-count",
            "NN:fixture-cardinality-correct-count",
            "NN:fixture-maximum-anchor-matching-error",
        }
        for family in config["encoder_families"]:
            family_key = latex_key(family)
            for offset in config["width_offsets"]:
                prefix = f"NN:pooled:{family_key}:offset{latex_key(offset)}:testamrmse"
                required_macro_keys.update(
                    {f"{prefix}mean", f"{prefix}ci95low", f"{prefix}ci95high"}
                )
        reporting = config["reporting"]
        primary = reporting["primary_slice"]
        domain = latex_key(primary["domain"])
        n = int(primary["n"])
        width = 2 * n + int(primary["width_offset"])
        for family in config["encoder_families"]:
            family_key = latex_key(family)
            condition_prefix = f"NN:condition:{family_key}:{domain}:n{n}:m{width}:"
            for metric in (
                "testamrmse",
                "testcardinalityaccuracy",
                "normalizednearestp05",
                "localsensitivitymedian",
            ):
                required_macro_keys.update(
                    {
                        f"{condition_prefix}{metric}mean",
                        f"{condition_prefix}{metric}ci95low",
                        f"{condition_prefix}{metric}ci95high",
                    }
                )
            for bits in reporting["displayed_quantization_bits"]:
                quantization_prefix = (
                    f"NN:quantization:{family_key}:{domain}:n{n}:m{width}:b{int(bits)}:"
                )
                for metric in ("amrmse", "pairedamrmsechange"):
                    required_macro_keys.update(
                        {
                            f"{quantization_prefix}{metric}mean",
                            f"{quantization_prefix}{metric}ci95low",
                            f"{quantization_prefix}{metric}ci95high",
                        }
                    )
        for key in sorted(required_macro_keys):
            if key not in macro_text:
                errors.append(f"manuscript_macros.tex is missing {key}")
    return errors


def artifact_files(root: Path) -> list[Path]:
    values: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "manifest.sha256":
            continue
        if path.name.startswith(".") and ".tmp" in path.name:
            continue
        values.append(path)
    return sorted(values, key=lambda path: path.relative_to(root).as_posix())


def write_manifest(root: Path) -> None:
    path = root / "manifest.sha256"
    temporary = root / ".manifest.sha256.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for artifact in artifact_files(root):
            relative = artifact.relative_to(root).as_posix()
            stream.write(f"{sha256_file(artifact)}  {relative}\n")
    os.replace(temporary, path)


def verify_manifest(root: Path) -> list[str]:
    path = root / "manifest.sha256"
    if not path.is_file():
        return ["manifest.sha256 is missing; run with --write-manifest after aggregation"]
    errors: list[str] = []
    listed: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = line.rstrip("\n")
            if "  " not in value:
                errors.append(f"manifest line {line_number} is malformed")
                continue
            digest, relative = value.split("  ", 1)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                errors.append(f"manifest line {line_number} has an invalid SHA-256 digest")
                continue
            if relative in listed:
                errors.append(f"manifest contains duplicate path {relative}")
            listed[relative] = digest
    actual = {path.relative_to(root).as_posix(): path for path in artifact_files(root)}
    missing = sorted(set(actual) - listed.keys())
    extra = sorted(listed.keys() - set(actual))
    if missing:
        errors.append(f"manifest omits {len(missing)} files; first entries: {missing[:5]}")
    if extra:
        errors.append(f"manifest lists {len(extra)} absent files; first entries: {extra[:5]}")
    for relative in sorted(set(actual) & listed.keys()):
        if sha256_file(actual[relative]) != listed[relative]:
            errors.append(f"SHA-256 mismatch: {relative}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="After structural verification, create or replace manifest.sha256.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    required_root_files = (
        "resolved_config.json",
        "environment.json",
        "provenance.json",
        "run_index.json",
    )
    for name in required_root_files:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty {name}")
    config_path = root / "resolved_config.json"
    if config_path.is_file():
        try:
            config = load_json(config_path)
            errors.extend(verify_run_grid(root, config))
            errors.extend(verify_analytic(root, config))
            errors.extend(verify_aggregate(root, config))
            source_hash_path = root / "source" / "source_hashes.json"
            if not source_hash_path.is_file():
                errors.append("source/source_hashes.json is missing")
            else:
                source_hashes = load_json(source_hash_path)
                for name, digest in source_hashes.items():
                    source_path = root / "source" / name
                    if not source_path.is_file() or sha256_file(source_path) != digest:
                        errors.append(f"Source snapshot mismatch: source/{name}")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"Cannot validate resolved configuration: {exc}")
    temporary_files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name.startswith(".") and ".tmp" in path.name
    ]
    if temporary_files:
        errors.append(f"Temporary files remain: {temporary_files[:5]}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"Artifact verification failed with {len(errors)} error(s).")
    if args.write_manifest:
        write_manifest(root)
    manifest_errors = verify_manifest(root)
    if manifest_errors:
        for error in manifest_errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"Manifest verification failed with {len(manifest_errors)} error(s).")
    print(
        f"Verified {len(expected_run_ids(config))} learned runs and "
        f"{len(artifact_files(root))} hashed artifacts under {root}"
    )


if __name__ == "__main__":
    main()
