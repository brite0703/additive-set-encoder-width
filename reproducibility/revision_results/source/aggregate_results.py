"""Deterministically aggregate completed revision experiments.

This program reads only immutable per-run records and the resolved configuration.  It refuses an
incomplete or duplicated grid, writes machine-readable summaries, and emits LaTeX lookup macros so
that manuscript values need not be transcribed by hand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.2621571627409915,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

SUMMARY_METRICS = (
    "test_amrmse",
    "test_cardinality_accuracy",
    "best_validation_amrmse",
    "latent_scale",
    "normalized_nearest_p05",
    "normalized_nearest_median",
    "local_sensitivity_median",
    "local_sensitivity_p95",
    "best_epoch",
    "duration_seconds",
)


def validate_reporting(config: dict[str, Any]) -> None:
    reporting = config.get("reporting", {})
    primary = reporting.get("primary_slice", {})
    if set(primary) != {"domain", "n", "width_offset"}:
        raise ValueError("The resolved configuration lacks a complete reporting.primary_slice.")
    if primary["domain"] not in config["domains"]:
        raise ValueError("The reporting domain is absent from the resolved grid.")
    if int(primary["n"]) not in {int(value) for value in config["cardinality_caps"]}:
        raise ValueError("The reporting cap is absent from the resolved grid.")
    if int(primary["width_offset"]) not in {int(value) for value in config["width_offsets"]}:
        raise ValueError("The reporting width offset is absent from the resolved grid.")
    displayed_bits = {int(value) for value in reporting.get("displayed_quantization_bits", [])}
    if not displayed_bits or not displayed_bits.issubset(
        {int(value) for value in config["diagnostics"]["quantization_bits"]}
    ):
        raise ValueError("Displayed quantization depths are not a configured nonempty subset.")
    if reporting.get("pooled_rule") != "equal_weight_domain_cap_within_seed":
        raise ValueError("Unsupported prespecified pooling rule.")
    if reporting.get("interval_method") != "two_sided_student_t_across_seeds":
        raise ValueError("Unsupported prespecified interval method.")
    if float(reporting.get("confidence_level", 0.0)) != 0.95:
        raise ValueError("The prespecified confidence level must be 0.95.")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


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
                        seed = int(seed_value)
                        values.append(f"{family}__{domain}__n{n}__m{width}__seed{seed}")
    return sorted(values)


def finite_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def read_records(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    digest = config_digest(config)
    records: dict[str, dict[str, Any]] = {}
    required = {
        "schema_version",
        "run_id",
        "config_sha256",
        "family",
        "domain",
        "n",
        "width",
        "width_offset",
        "seed",
        "test_amrmse",
        "test_cardinality_accuracy",
        "duration_seconds",
        "quantization",
    }
    for path in sorted((root / "runs").glob("*/record.json")):
        record = load_json(path)
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        run_id = str(record["run_id"])
        if run_id in records:
            raise ValueError(f"Duplicate run_id: {run_id}")
        if path.parent.name != run_id:
            raise ValueError(f"Run directory and run_id disagree: {path}")
        if int(record["schema_version"]) != 1:
            raise ValueError(f"Unsupported record schema in {path}")
        if record["config_sha256"] != digest:
            raise ValueError(f"Configuration digest mismatch in {path}")
        family = str(record["family"])
        domain = str(record["domain"])
        n = int(record["n"])
        width = int(record["width"])
        offset = int(record["width_offset"])
        seed = int(record["seed"])
        expected_from_fields = f"{family}__{domain}__n{n}__m{width}__seed{seed}"
        if run_id != expected_from_fields:
            raise ValueError(f"run_id disagrees with record fields in {path}")
        if family not in config["encoder_families"] or domain not in config["domains"]:
            raise ValueError(f"Unconfigured family or domain in {path}")
        if n not in {int(value) for value in config["cardinality_caps"]}:
            raise ValueError(f"Unconfigured cardinality cap in {path}")
        if offset not in {int(value) for value in config["width_offsets"]} or width != 2 * n + offset:
            raise ValueError(f"Unconfigured or inconsistent width in {path}")
        if seed not in {int(value) for value in config["seeds"]}:
            raise ValueError(f"Unconfigured seed in {path}")
        for metric in ("test_amrmse", "test_cardinality_accuracy", "duration_seconds"):
            finite_number(record[metric], f"{run_id}.{metric}")
        records[run_id] = record
    expected = expected_run_ids(config)
    missing_runs = sorted(set(expected) - records.keys())
    unexpected_runs = sorted(records.keys() - set(expected))
    if missing_runs or unexpected_runs:
        raise ValueError(
            f"Run grid is incomplete or contaminated: missing={len(missing_runs)}, "
            f"unexpected={len(unexpected_runs)}. First missing={missing_runs[:3]}, "
            f"first unexpected={unexpected_runs[:3]}"
        )
    index_path = root / "run_index.json"
    if not index_path.exists():
        raise ValueError("run_index.json is missing")
    index_ids = sorted(str(item["run_id"]) for item in load_json(index_path))
    if index_ids != expected:
        raise ValueError("run_index.json does not exactly match the expected run grid")
    return [records[key] for key in expected]


def sample_summary(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    if not all(math.isfinite(value) for value in clean):
        raise ValueError("Cannot aggregate non-finite values")
    count = len(clean)
    mean = sum(clean) / count
    if count == 1:
        return {"count": 1, "mean": mean, "sd": None, "ci95_low": None, "ci95_high": None}
    variance = sum((value - mean) ** 2 for value in clean) / (count - 1)
    sd = math.sqrt(variance)
    critical = T_975.get(count - 1, 1.96)
    half_width = critical * sd / math.sqrt(count)
    return {
        "count": count,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize_groups(
    records: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in group_fields)].append(record)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        group = grouped[key]
        row = dict(zip(group_fields, key))
        row["run_count"] = len(group)
        row["seeds"] = ";".join(str(value) for value in sorted({int(item["seed"]) for item in group}))
        for metric in metrics:
            statistics = sample_summary(item.get(metric) for item in group)
            for statistic, value in statistics.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return rows


def pooled_seed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["family"]), int(record["width_offset"]), int(record["seed"]))].append(record)
    pooled: list[dict[str, Any]] = []
    for (family, offset, seed), group in sorted(grouped.items()):
        row: dict[str, Any] = {"family": family, "width_offset": offset, "seed": seed}
        row["condition_count"] = len(group)
        for metric in SUMMARY_METRICS:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            row[metric] = sum(values) / len(values) if values else None
        pooled.append(row)
    return pooled


def paired_family_records(records: list[dict[str, Any]], families: list[str]) -> list[dict[str, Any]]:
    if len(families) != 2:
        return []
    first, second = families
    lookup = {
        (record["family"], record["domain"], record["n"], record["width"], record["seed"]): record
        for record in records
    }
    differences: list[dict[str, Any]] = []
    for record in records:
        if record["family"] != first:
            continue
        key = (second, record["domain"], record["n"], record["width"], record["seed"])
        counterpart = lookup.get(key)
        if counterpart is None:
            raise ValueError(f"Missing paired family run for {key}")
        row: dict[str, Any] = {
            "family_first": first,
            "family_second": second,
            "difference_direction": f"{second}-minus-{first}",
            "domain": record["domain"],
            "n": record["n"],
            "width": record["width"],
            "width_offset": record["width_offset"],
            "seed": record["seed"],
        }
        for metric in ("test_amrmse", "test_cardinality_accuracy", "local_sensitivity_median"):
            if record.get(metric) is None or counterpart.get(metric) is None:
                row[metric] = None
            else:
                row[metric] = float(counterpart[metric]) - float(record[metric])
        differences.append(row)
    return differences


def flatten_quantization(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for result in record["quantization"]:
            row = {
                "family": record["family"],
                "domain": record["domain"],
                "n": record["n"],
                "width": record["width"],
                "width_offset": record["width_offset"],
                "seed": record["seed"],
            }
            row.update(result)
            rows.append(row)
    return rows


def csv_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def latex_number(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) < 1e-4 or abs(number) >= 1e4:
        mantissa, exponent = f"{number:.4e}".split("e")
        return rf"\ensuremath{{{mantissa}\times 10^{{{int(exponent)}}}}}"
    return f"{number:.6f}"


def latex_key(value: Any) -> str:
    return str(value).replace("_", "").replace("-", "minus").replace(".", "p")


def make_latex_macros(
    summary_rows: list[dict[str, Any]],
    pooled_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    quantization_rows: list[dict[str, Any]],
    analytic_rows: list[dict[str, Any]],
    fixture_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    lines = [
        "% Generated deterministically by aggregate_results.py; do not edit by hand.",
        r"\providecommand{\NNResult}[1]{\csname NN:#1\endcsname}",
    ]

    def define(key: str, value: Any) -> None:
        lines.append(rf"\expandafter\def\csname NN:{key}\endcsname{{{latex_number(value)}}}")

    define("run-count", metadata["actual_learned_run_count"])
    define("seed-count", metadata["seed_count"])
    define("analytic-example-count", metadata["analytic_example_count"])
    define("analytic-within-tolerance-count", metadata["analytic_within_tolerance_count"])
    define("analytic-cardinality-correct-count", metadata["analytic_cardinality_correct_count"])
    define("analytic-maximum-anchor-matching-error", metadata["analytic_maximum_anchor_matching_error"])
    define("fixture-count", metadata["fixture_count"])
    define("fixture-within-tolerance-count", metadata["fixture_within_tolerance_count"])
    define("fixture-cardinality-correct-count", metadata["fixture_cardinality_correct_count"])
    define("fixture-maximum-anchor-matching-error", metadata["fixture_maximum_anchor_matching_error"])
    define("total-compute-hours", metadata["summed_run_hours"])
    for row in summary_rows:
        prefix = ":".join(
            (
                "condition",
                latex_key(row["family"]),
                latex_key(row["domain"]),
                f"n{row['n']}",
                f"m{row['width']}",
            )
        )
        for key, value in row.items():
            if key in {"family", "domain", "n", "width", "width_offset", "seeds"}:
                continue
            define(f"{prefix}:{latex_key(key)}", value)
    for row in pooled_rows:
        prefix = ":".join(
            ("pooled", latex_key(row["family"]), f"offset{latex_key(row['width_offset'])}")
        )
        for key, value in row.items():
            if key in {"family", "width_offset", "seeds"}:
                continue
            define(f"{prefix}:{latex_key(key)}", value)
    for row in paired_rows:
        prefix = ":".join(
            (
                "pairedfamily",
                latex_key(row["difference_direction"]),
                latex_key(row["domain"]),
                f"n{row['n']}",
                f"m{row['width']}",
            )
        )
        for key, value in row.items():
            if key in {
                "family_first",
                "family_second",
                "difference_direction",
                "domain",
                "n",
                "width",
                "width_offset",
                "seeds",
            }:
                continue
            define(f"{prefix}:{latex_key(key)}", value)
    for row in quantization_rows:
        prefix = ":".join(
            (
                "quantization",
                latex_key(row["family"]),
                latex_key(row["domain"]),
                f"n{row['n']}",
                f"m{row['width']}",
                f"b{row['bits']}",
            )
        )
        for key, value in row.items():
            if key in {
                "family",
                "domain",
                "n",
                "width",
                "width_offset",
                "bits",
                "seeds",
            }:
                continue
            define(f"{prefix}:{latex_key(key)}", value)
    for row in analytic_rows:
        prefix = ":".join(("analytic", latex_key(row["domain"]), f"n{row['n']}"))
        for key, value in row.items():
            if key in {"domain", "n"}:
                continue
            define(f"{prefix}:{latex_key(key)}", value)
    for row in fixture_rows:
        prefix = ":".join(
            (
                "fixture",
                latex_key(row["domain"]),
                f"n{row['n']}",
                latex_key(row["fixture"]),
            )
        )
        for key, value in row.items():
            if key in {"domain", "n", "fixture"}:
                continue
            define(f"{prefix}:{latex_key(key)}", value)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Completed raw results directory")
    parser.add_argument("--output", type=Path, required=True, help="Aggregate output directory")
    args = parser.parse_args()

    root = args.input.resolve()
    output = args.output.resolve()
    expected_output = root / "aggregate"
    if output != expected_output:
        raise SystemExit(
            f"Aggregate output must be inside the results root at {expected_output}; received {output}."
        )
    current_aggregator_hash = sha256_file(Path(__file__).resolve())
    snapshotted_aggregator = root / "source" / "aggregate_results.py"
    if not snapshotted_aggregator.is_file():
        raise SystemExit("The run does not contain source/aggregate_results.py.")
    if sha256_file(snapshotted_aggregator) != current_aggregator_hash:
        raise SystemExit(
            "The executing aggregator differs from the source snapshotted by the runner; refusing "
            "untraceable aggregation."
        )
    config = load_json(root / "resolved_config.json")
    validate_reporting(config)
    records = read_records(root, config)

    condition_rows = summarize_groups(
        records,
        ("family", "domain", "n", "width", "width_offset"),
        SUMMARY_METRICS,
    )
    pooled_records = pooled_seed_records(records)
    expected_conditions_per_seed = len(config["domains"]) * len(config["cardinality_caps"])
    if any(int(row["condition_count"]) != expected_conditions_per_seed for row in pooled_records):
        raise ValueError("A pooled seed/offset cell does not contain every domain-cardinality condition")
    pooled_rows = summarize_groups(
        pooled_records,
        ("family", "width_offset"),
        SUMMARY_METRICS,
    )
    paired_records = paired_family_records(records, [str(value) for value in config["encoder_families"]])
    paired_rows = (
        summarize_groups(
            paired_records,
            (
                "family_first",
                "family_second",
                "difference_direction",
                "domain",
                "n",
                "width",
                "width_offset",
            ),
            ("test_amrmse", "test_cardinality_accuracy", "local_sensitivity_median"),
        )
        if paired_records
        else []
    )
    quantization_records = flatten_quantization(records)
    quantization_rows = summarize_groups(
        quantization_records,
        ("family", "domain", "n", "width", "width_offset", "bits"),
        (
            "amrmse",
            "cardinality_accuracy",
            "paired_amrmse_change",
            "validation_clipping_rate",
            "test_clipping_rate",
        ),
    )
    seed_count = len(config["seeds"])
    for row in pooled_rows:
        if int(row["run_count"]) != seed_count or int(row["test_amrmse_count"]) != seed_count:
            raise ValueError("A manuscript-displayed pooled AMRMSE cell lacks a complete seed set.")
    primary = config["reporting"]["primary_slice"]
    primary_width = 2 * int(primary["n"]) + int(primary["width_offset"])
    primary_conditions = [
        row
        for row in condition_rows
        if str(row["domain"]) == str(primary["domain"])
        and int(row["n"]) == int(primary["n"])
        and int(row["width"]) == primary_width
    ]
    if len(primary_conditions) != len(config["encoder_families"]):
        raise ValueError("The prespecified manuscript slice is incomplete.")
    for row in primary_conditions:
        if int(row["run_count"]) != seed_count:
            raise ValueError("A manuscript-displayed condition lacks a complete seed set.")
        for metric in (
            "test_amrmse",
            "test_cardinality_accuracy",
            "normalized_nearest_p05",
            "local_sensitivity_median",
        ):
            if int(row[f"{metric}_count"]) != seed_count:
                raise ValueError(
                    f"The manuscript-displayed metric {metric} is undefined for one or more seeds."
                )
    displayed_bits = {int(value) for value in config["reporting"]["displayed_quantization_bits"]}
    primary_quantization = [
        row
        for row in quantization_rows
        if str(row["domain"]) == str(primary["domain"])
        and int(row["n"]) == int(primary["n"])
        and int(row["width"]) == primary_width
        and int(row["bits"]) in displayed_bits
    ]
    expected_quantization_cells = len(config["encoder_families"]) * len(displayed_bits)
    if len(primary_quantization) != expected_quantization_cells:
        raise ValueError("The prespecified manuscript quantization slice is incomplete.")
    for row in primary_quantization:
        if int(row["run_count"]) != seed_count:
            raise ValueError("A manuscript-displayed quantization cell lacks a complete seed set.")
        for metric in ("amrmse", "paired_amrmse_change"):
            if int(row[f"{metric}_count"]) != seed_count:
                raise ValueError(
                    f"The manuscript-displayed quantization metric {metric} is incomplete."
                )
    analytic_rows = load_json(root / "analytic_checks.json")
    fixture_rows = load_json(root / "analytic_fixture_checks.json")
    analytic_example_count = sum(int(row["sample_count"]) for row in analytic_rows)
    analytic_within_tolerance_count = sum(
        int(row["within_tolerance_count"]) for row in analytic_rows
    )
    analytic_cardinality_correct_count = sum(
        int(row["cardinality_correct_count"]) for row in analytic_rows
    )
    analytic_maximum_anchor_matching_error = max(
        float(row["maximum_anchor_matching_error"]) for row in analytic_rows
    )
    fixture_count = len(fixture_rows)
    fixture_within_tolerance_count = sum(int(row["within_tolerance"]) for row in fixture_rows)
    fixture_cardinality_correct_count = sum(int(row["cardinality_correct"]) for row in fixture_rows)
    fixture_maximum_anchor_matching_error = max(
        float(row["normalized_anchor_error"]) for row in fixture_rows
    )
    metadata = {
        "schema_version": 1,
        "config_sha256": config_digest(config),
        "aggregator_sha256": current_aggregator_hash,
        "expected_learned_run_count": len(expected_run_ids(config)),
        "actual_learned_run_count": len(records),
        "seed_count": len(config["seeds"]),
        "domain_count": len(config["domains"]),
        "cardinality_cap_count": len(config["cardinality_caps"]),
        "width_offset_count": len(config["width_offsets"]),
        "encoder_family_count": len(config["encoder_families"]),
        "analytic_condition_count": len(analytic_rows),
        "analytic_example_count": analytic_example_count,
        "analytic_within_tolerance_count": analytic_within_tolerance_count,
        "analytic_cardinality_correct_count": analytic_cardinality_correct_count,
        "analytic_maximum_anchor_matching_error": analytic_maximum_anchor_matching_error,
        "fixture_count": fixture_count,
        "fixture_within_tolerance_count": fixture_within_tolerance_count,
        "fixture_cardinality_correct_count": fixture_cardinality_correct_count,
        "fixture_maximum_anchor_matching_error": fixture_maximum_anchor_matching_error,
        "summed_run_hours": sum(float(record["duration_seconds"]) for record in records) / 3600.0,
        "pooling_rule": (
            "For each family, width offset, and seed, condition metrics are averaged equally over "
            "the Cartesian product of domains and cardinality caps; uncertainty is then computed "
            "across independent seeds."
        ),
        "confidence_interval": "Two-sided 95% Student-t interval across seed-level values.",
    }

    write_csv(output / "summary.csv", csv_fields(condition_rows), condition_rows)
    write_csv(output / "pooled_by_width_offset.csv", csv_fields(pooled_rows), pooled_rows)
    if paired_rows:
        write_csv(output / "paired_family_differences.csv", csv_fields(paired_rows), paired_rows)
    write_csv(output / "quantization_summary.csv", csv_fields(quantization_rows), quantization_rows)
    write_csv(output / "analytic_checks.csv", csv_fields(analytic_rows), analytic_rows)
    write_csv(output / "analytic_fixture_checks.csv", csv_fields(fixture_rows), fixture_rows)
    write_json(output / "aggregate_metadata.json", metadata)
    write_text(
        output / "manuscript_macros.tex",
        make_latex_macros(
            condition_rows,
            pooled_rows,
            paired_rows,
            quantization_rows,
            analytic_rows,
            fixture_rows,
            metadata,
        ),
    )
    print(
        f"Aggregated {len(records)} learned runs, {analytic_example_count} random analytic examples, "
        f"and {fixture_count} deterministic fixtures into {output}"
    )


if __name__ == "__main__":
    main()
