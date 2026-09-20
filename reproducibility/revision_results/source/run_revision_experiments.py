"""Reviewer-corrected experiments for NEUNET-D-26-03059.

The script intentionally contains no lexicographic target ordering and no random-pair collision test.
All multiset losses use exact assignment after padding with an out-of-domain anchor.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import itertools
import io
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Required by CUDA for deterministic matrix multiplication.  This must be set
# before a CUDA context is created on the remote workstation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised on the remote workstation
    raise SystemExit("PyTorch is required. Install the pinned environment from requirements.txt.") from exc


@dataclass(frozen=True)
class Split:
    points: np.ndarray
    mask: np.ndarray
    targets: np.ndarray
    sizes: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def stable_seed(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def domain_diameter_with_anchor(domain: str, anchor: np.ndarray) -> float:
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (2,):
        raise ValueError("The planar anchor must contain exactly two coordinates.")
    if domain == "square":
        corners = np.asarray(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)))
        return max(math.sqrt(2.0), float(np.linalg.norm(corners - anchor, axis=1).max()))
    if domain in {"disk", "annulus"}:
        return max(2.0, float(np.linalg.norm(anchor)) + 1.0)
    raise ValueError(f"Unknown domain: {domain}")


def domain_anchor_distance(domain: str, anchor: np.ndarray, inner: float) -> float:
    anchor = np.asarray(anchor, dtype=np.float64)
    if domain == "square":
        displacement = np.maximum(np.maximum(-anchor, anchor - 1.0), 0.0)
        return float(np.linalg.norm(displacement))
    norm = float(np.linalg.norm(anchor))
    if domain == "disk":
        return max(norm - 1.0, 0.0)
    if domain == "annulus":
        if norm < inner:
            return inner - norm
        if norm > 1.0:
            return norm - 1.0
        return 0.0
    raise ValueError(f"Unknown domain: {domain}")


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "runtime",
        "sampling",
        "seeds",
        "domains",
        "cardinality_caps",
        "width_offsets",
        "encoder_families",
        "anchor",
        "annulus_inner_radius",
        "splits",
        "training",
        "diagnostics",
        "reporting",
        "artifacts",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Configuration is missing required keys: {missing}")
    if int(config["schema_version"]) != 1:
        raise ValueError("Unsupported configuration schema_version; expected 1.")
    runtime = config["runtime"]
    if len(runtime.get("python_major_minor", [])) != 2:
        raise ValueError("runtime.python_major_minor must contain [major, minor].")
    if not runtime.get("torch") or not runtime.get("numpy"):
        raise ValueError("runtime must declare exact torch and numpy versions.")
    expected_sampling = {
        "cardinality": "discrete_uniform_on_0_to_n",
        "square": "coordinatewise_uniform",
        "disk": "area_uniform",
        "annulus": "area_uniform",
    }
    if config["sampling"] != expected_sampling:
        raise ValueError(f"sampling must equal the implemented declaration: {expected_sampling}")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be a nonempty list of unique integers.")
    if any(int(n) < 1 for n in config["cardinality_caps"]):
        raise ValueError("Every cardinality cap must be positive.")
    if any(2 * int(n) + int(offset) < 1 for n in config["cardinality_caps"] for offset in config["width_offsets"]):
        raise ValueError("Every requested latent width must be positive.")
    supported_domains = {"square", "disk", "annulus"}
    unknown_domains = sorted(set(config["domains"]) - supported_domains)
    if unknown_domains:
        raise ValueError(f"Unsupported domains: {unknown_domains}")
    supported_families = {"mlp", "theorem_projection"}
    unknown_families = sorted(set(config["encoder_families"]) - supported_families)
    if unknown_families:
        raise ValueError(f"Unsupported encoder families: {unknown_families}")
    inner = float(config["annulus_inner_radius"])
    if not 0.0 < inner < 1.0:
        raise ValueError("annulus_inner_radius must lie strictly between zero and one.")
    anchor = np.asarray(config["anchor"], dtype=np.float64)
    if anchor.shape != (2,) or not np.isfinite(anchor).all():
        raise ValueError("anchor must be a finite planar coordinate pair.")
    if "theorem_projection" in config["encoder_families"] and anchor[1] != 0.0:
        raise ValueError("theorem_projection currently requires a real complex-plane anchor.")
    for domain in config["domains"]:
        tau = domain_anchor_distance(domain, anchor, inner)
        if tau <= 0.0:
            raise ValueError(f"The anchor {anchor.tolist()} is not outside domain {domain!r}.")
    for name in ("train", "validation", "test", "analytic_test"):
        if int(config["splits"][name]) < 1:
            raise ValueError(f"splits.{name} must be positive.")
    training = config["training"]
    for name in ("epochs", "batch_size", "hidden_width", "hidden_depth"):
        if int(training[name]) < 1:
            raise ValueError(f"training.{name} must be positive.")
    for name in ("learning_rate", "cardinality_loss_weight"):
        if float(training[name]) <= 0.0:
            raise ValueError(f"training.{name} must be positive.")
    if training.get("theorem_projection_mode") != "learned_all_widths":
        raise ValueError(
            "training.theorem_projection_mode must be 'learned_all_widths' for a comparable width sweep."
        )
    if training.get("strict_determinism", True) is not True:
        raise ValueError("training.strict_determinism must be true for the revision run.")
    diagnostics = config["diagnostics"]
    recovery_tolerance = float(
        diagnostics.get("normalized_recovery_tolerance", diagnostics.get("root_tolerance", 0.0))
    )
    if recovery_tolerance <= 0.0:
        raise ValueError("diagnostics.normalized_recovery_tolerance must be positive.")
    if float(diagnostics["relative_perturbation"]) <= 0.0:
        raise ValueError("diagnostics.relative_perturbation must be positive.")
    if int(diagnostics["perturbations_per_example"]) < 1:
        raise ValueError("diagnostics.perturbations_per_example must be positive.")
    if not 0.0 < float(diagnostics["nearest_code_quantile"]) < 1.0:
        raise ValueError("diagnostics.nearest_code_quantile must lie strictly between zero and one.")
    if diagnostics.get("quantile_method") != "linear":
        raise ValueError("diagnostics.quantile_method must be 'linear'.")
    if float(diagnostics["standardized_clip"]) <= 0.0:
        raise ValueError("diagnostics.standardized_clip must be positive.")
    if int(diagnostics.get("quantization_scale_ddof", -1)) != 0:
        raise ValueError("diagnostics.quantization_scale_ddof must be zero.")
    if float(diagnostics.get("quantization_scale_floor_relative", 0.0)) <= 0.0:
        raise ValueError("diagnostics.quantization_scale_floor_relative must be positive.")
    if any(int(bits) < 2 for bits in diagnostics["quantization_bits"]):
        raise ValueError("Every quantization bit depth must be at least two.")
    reporting = config["reporting"]
    primary = reporting.get("primary_slice", {})
    if set(primary) != {"domain", "n", "width_offset"}:
        raise ValueError("reporting.primary_slice must declare domain, n, and width_offset.")
    if primary["domain"] not in config["domains"]:
        raise ValueError("reporting.primary_slice.domain is not in the configured domains.")
    if int(primary["n"]) not in {int(value) for value in config["cardinality_caps"]}:
        raise ValueError("reporting.primary_slice.n is not in the configured cardinality caps.")
    if int(primary["width_offset"]) not in {int(value) for value in config["width_offsets"]}:
        raise ValueError("reporting.primary_slice.width_offset is not configured.")
    displayed_bits = [int(value) for value in reporting.get("displayed_quantization_bits", [])]
    if not displayed_bits or not set(displayed_bits).issubset(
        {int(value) for value in diagnostics["quantization_bits"]}
    ):
        raise ValueError("reporting.displayed_quantization_bits must be a nonempty configured subset.")
    if reporting.get("pooled_rule") != "equal_weight_domain_cap_within_seed":
        raise ValueError("Unsupported reporting.pooled_rule.")
    if reporting.get("interval_method") != "two_sided_student_t_across_seeds":
        raise ValueError("Unsupported reporting.interval_method.")
    if float(reporting.get("confidence_level", 0.0)) != 0.95:
        raise ValueError("reporting.confidence_level must be 0.95 for the prespecified analysis.")


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    effective["domains"] = effective["domains"][:1]
    effective["cardinality_caps"] = effective["cardinality_caps"][:1]
    effective["width_offsets"] = [0]
    effective["encoder_families"] = effective["encoder_families"][:1]
    effective["reporting"]["primary_slice"] = {
        "domain": effective["domains"][0],
        "n": int(effective["cardinality_caps"][0]),
        "width_offset": 0,
    }
    effective["seeds"] = effective["seeds"][:1]
    effective["splits"].update({"train": 64, "validation": 32, "test": 32, "analytic_test": 32})
    effective["training"]["epochs"] = min(2, int(effective["training"]["epochs"]))
    effective["diagnostics"]["perturbations_per_example"] = min(
        4, int(effective["diagnostics"]["perturbations_per_example"])
    )
    effective["smoke_mode"] = True
    return effective


def sample_domain_points(domain: str, count: int, rng: np.random.Generator, inner: float) -> np.ndarray:
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    if domain == "square":
        return rng.uniform(0.0, 1.0, size=(count, 2)).astype(np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=count)
    if domain == "disk":
        radius = np.sqrt(rng.uniform(0.0, 1.0, size=count))
    elif domain == "annulus":
        radius = np.sqrt(rng.uniform(inner * inner, 1.0, size=count))
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=1).astype(np.float32)


def make_split(
    domain: str,
    n: int,
    count: int,
    seed: int,
    anchor: np.ndarray,
    inner: float,
) -> Split:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(0, n + 1, size=count, endpoint=False, dtype=np.int64)
    points = np.zeros((count, n, 2), dtype=np.float32)
    mask = np.zeros((count, n), dtype=np.float32)
    targets = np.broadcast_to(anchor.astype(np.float32), (count, n, 2)).copy()
    for row, size in enumerate(sizes.tolist()):
        sampled = sample_domain_points(domain, size, rng, inner)
        points[row, :size] = sampled
        mask[row, :size] = 1.0
        targets[row, :size] = sampled
    return Split(points=points, mask=mask, targets=targets, sizes=sizes)


def all_permutations(n: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long, device=device)


def assignment_squared(
    predicted: torch.Tensor,
    target: torch.Tensor,
    permutations: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    # predicted/target: [batch,n,2]; result: minimum normalized squared distance per example.
    target_permuted = target[:, permutations, :]  # [batch,permutations,n,2]
    difference = predicted[:, None, :, :] - target_permuted
    costs = difference.square().sum(dim=(-1, -2)) / (predicted.shape[1] * radius * radius)
    return costs.min(dim=1).values


def cardinality_discrepancy_squared(
    predicted_sizes: torch.Tensor,
    target_sizes: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Normalized squared cardinality term used in every reported decoder distance."""
    return ((predicted_sizes.to(torch.float32) - target_sizes.to(torch.float32)) / float(n)).square()


def postprocess_predictions(points: torch.Tensor, logits: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = logits.argmax(dim=1)
    result = anchor.view(1, 1, 2).expand_as(points).clone()
    distances = (points - anchor.view(1, 1, 2)).square().sum(dim=-1)
    order = distances.argsort(dim=1, descending=True)
    for row in range(points.shape[0]):
        size = int(sizes[row].item())
        if size:
            chosen = order[row, :size]
            result[row, :size] = points[row, chosen]
    return result, sizes


class PointMLPSum(nn.Module):
    def __init__(self, output_width: int, hidden_width: int, hidden_depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_width = 2
        for _ in range(hidden_depth):
            layers.extend((nn.Linear(input_width, hidden_width), nn.ReLU()))
            input_width = hidden_width
        layers.append(nn.Linear(input_width, output_width))
        self.point_network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.point_network(points)
        return (features * mask[..., None]).sum(dim=1)


class TheoremProjectionSum(nn.Module):
    def __init__(
        self,
        n: int,
        output_width: int,
        anchor: tuple[float, float],
    ) -> None:
        super().__init__()
        if anchor[1] != 0.0:
            raise ValueError("This implementation currently expects a real complex-plane anchor.")
        self.n = n
        self.anchor_real = float(anchor[0])
        native_width = 2 * n
        self.projection = nn.Linear(native_width, output_width, bias=False)

    def point_features(self, points: torch.Tensor) -> torch.Tensor:
        real = points[..., 0]
        imag = points[..., 1]
        power_real = real
        power_imag = imag
        blocks: list[torch.Tensor] = []
        for degree in range(1, self.n + 1):
            blocks.append(torch.stack((power_real - self.anchor_real**degree, power_imag), dim=-1))
            next_real = power_real * real - power_imag * imag
            next_imag = power_real * imag + power_imag * real
            power_real, power_imag = next_real, next_imag
        return torch.cat(blocks, dim=-1)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        summed = (self.point_features(points) * mask[..., None]).sum(dim=1)
        return self.projection(summed)


class Decoder(nn.Module):
    def __init__(self, latent_width: int, n: int, hidden_width: int, hidden_depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_width = latent_width
        for _ in range(hidden_depth):
            layers.extend((nn.Linear(input_width, hidden_width), nn.ReLU()))
            input_width = hidden_width
        layers.append(nn.Linear(input_width, 2 * n + n + 1))
        self.network = nn.Sequential(*layers)
        self.n = n

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(latent)
        return output[:, : 2 * self.n].reshape(-1, self.n, 2), output[:, 2 * self.n :]


class ReconstructionModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(points, mask)
        predicted, logits = self.decoder(latent)
        return latent, predicted, logits


def to_tensor(split: Split, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(split.points).to(device),
        torch.from_numpy(split.mask).to(device),
        torch.from_numpy(split.targets).to(device),
        torch.from_numpy(split.sizes).to(device),
    )


@torch.no_grad()
def evaluate(
    model: ReconstructionModel,
    split: Split,
    batch_size: int,
    permutations: torch.Tensor,
    radius: float,
    anchor: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    points, mask, targets, sizes = to_tensor(split, device)
    all_latent: list[torch.Tensor] = []
    all_predicted: list[torch.Tensor] = []
    all_predicted_sizes: list[torch.Tensor] = []
    all_squared: list[torch.Tensor] = []
    for start in range(0, points.shape[0], batch_size):
        stop = min(start + batch_size, points.shape[0])
        latent, raw_predicted, logits = model(points[start:stop], mask[start:stop])
        predicted, predicted_sizes = postprocess_predictions(raw_predicted, logits, anchor)
        squared = assignment_squared(predicted, targets[start:stop], permutations, radius)
        squared = squared + cardinality_discrepancy_squared(
            predicted_sizes, sizes[start:stop], predicted.shape[1]
        )
        all_latent.append(latent.cpu())
        all_predicted.append(predicted.cpu())
        all_predicted_sizes.append(predicted_sizes.cpu())
        all_squared.append(squared.cpu())
    latent = torch.cat(all_latent)
    predicted = torch.cat(all_predicted)
    predicted_sizes = torch.cat(all_predicted_sizes)
    squared = torch.cat(all_squared)
    if not torch.isfinite(latent).all() or not torch.isfinite(predicted).all() or not torch.isfinite(squared).all():
        raise FloatingPointError("Non-finite value encountered during model evaluation.")
    return {
        "amrmse": float(squared.mean().sqrt().item()),
        "cardinality_accuracy": float((predicted_sizes == sizes.cpu()).float().mean().item()),
        "latent": latent.numpy(),
        "predicted": predicted.numpy(),
        "predicted_sizes": predicted_sizes.numpy(),
        "squared_errors": squared.numpy(),
    }


@torch.no_grad()
def evaluate_latent_codes(
    decoder: Decoder,
    latent_codes: np.ndarray,
    targets: np.ndarray,
    true_sizes: np.ndarray,
    batch_size: int,
    permutations: torch.Tensor,
    radius: float,
    anchor: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    decoder.eval()
    latent = torch.from_numpy(np.asarray(latent_codes, dtype=np.float32)).to(device)
    target = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
    target_sizes_tensor = torch.from_numpy(np.asarray(true_sizes, dtype=np.int64)).to(device)
    predicted_parts: list[torch.Tensor] = []
    size_parts: list[torch.Tensor] = []
    squared_parts: list[torch.Tensor] = []
    for start in range(0, latent.shape[0], batch_size):
        stop = min(start + batch_size, latent.shape[0])
        raw_predicted, logits = decoder(latent[start:stop])
        predicted, predicted_sizes = postprocess_predictions(raw_predicted, logits, anchor)
        squared = assignment_squared(predicted, target[start:stop], permutations, radius)
        squared = squared + cardinality_discrepancy_squared(
            predicted_sizes, target_sizes_tensor[start:stop], predicted.shape[1]
        )
        predicted_parts.append(predicted.cpu())
        size_parts.append(predicted_sizes.cpu())
        squared_parts.append(squared.cpu())
    predicted = torch.cat(predicted_parts).numpy()
    predicted_sizes = torch.cat(size_parts).numpy()
    squared = torch.cat(squared_parts).numpy()
    if not np.isfinite(predicted).all() or not np.isfinite(squared).all():
        raise FloatingPointError("Non-finite value encountered while decoding perturbed latents.")
    return {
        "amrmse": float(np.sqrt(np.mean(squared, dtype=np.float64))),
        "cardinality_accuracy": float(np.mean(predicted_sizes == true_sizes)),
        "predicted": predicted,
        "predicted_sizes": predicted_sizes,
        "squared_errors": squared,
    }


def multiset_fingerprints(split: Split) -> np.ndarray:
    """Exact identity keys used only to exclude duplicate inputs from neighbor diagnostics."""
    fingerprints: list[bytes] = []
    for row, size_value in enumerate(split.sizes.tolist()):
        size = int(size_value)
        point_values = np.asarray(split.points[row, :size], dtype="<f4")
        if size:
            order = np.lexsort((point_values[:, 1], point_values[:, 0]))
            point_values = point_values[order]
        payload = size.to_bytes(4, "little", signed=False) + point_values.tobytes(order="C")
        fingerprints.append(hashlib.sha256(payload).digest())
    return np.asarray(fingerprints, dtype="S32")


def normalized_nearest_neighbor_distances(
    latent: np.ndarray,
    split: Split,
    latent_scale: float,
    chunk_size: int = 128,
) -> np.ndarray:
    if not math.isfinite(latent_scale) or latent_scale <= 0.0:
        return np.full(latent.shape[0], np.nan, dtype=np.float64)
    keys = multiset_fingerprints(split)
    nearest = np.full(latent.shape[0], np.inf, dtype=np.float64)
    latent64 = np.asarray(latent, dtype=np.float64)
    for start in range(0, latent64.shape[0], chunk_size):
        stop = min(start + chunk_size, latent64.shape[0])
        block = latent64[start:stop]
        distances = np.linalg.norm(block[:, None, :] - latent64[None, :, :], axis=-1)
        same_input = keys[start:stop, None] == keys[None, :]
        distances[same_input] = np.inf
        nearest[start:stop] = distances.min(axis=1) / latent_scale
    nearest[~np.isfinite(nearest)] = np.nan
    return nearest


@torch.no_grad()
def local_decoder_sensitivity(
    decoder: Decoder,
    latent_codes: np.ndarray,
    clean_predictions: np.ndarray,
    clean_predicted_sizes: np.ndarray,
    latent_scale: float,
    relative_perturbation: float,
    perturbations_per_example: int,
    batch_size: int,
    permutations: torch.Tensor,
    radius: float,
    anchor: torch.Tensor,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    if not math.isfinite(latent_scale) or latent_scale <= 0.0:
        return np.full(latent_codes.shape[0], np.nan, dtype=np.float64)
    decoder.eval()
    rng = np.random.default_rng(seed)
    sensitivities = np.empty((latent_codes.shape[0], perturbations_per_example), dtype=np.float64)
    for start in range(0, latent_codes.shape[0], batch_size):
        stop = min(start + batch_size, latent_codes.shape[0])
        clean_latent = np.asarray(latent_codes[start:stop], dtype=np.float32)
        clean_padded = torch.from_numpy(np.asarray(clean_predictions[start:stop], dtype=np.float32)).to(device)
        clean_sizes = torch.from_numpy(
            np.asarray(clean_predicted_sizes[start:stop], dtype=np.int64)
        ).to(device)
        for repetition in range(perturbations_per_example):
            directions = rng.normal(size=clean_latent.shape)
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / np.maximum(norms, np.finfo(np.float64).tiny)
            perturbed = clean_latent.astype(np.float64) + (
                relative_perturbation * latent_scale * directions
            )
            latent_tensor = torch.from_numpy(perturbed.astype(np.float32)).to(device)
            raw_predicted, logits = decoder(latent_tensor)
            predicted, predicted_sizes = postprocess_predictions(raw_predicted, logits, anchor)
            squared = assignment_squared(predicted, clean_padded, permutations, radius)
            squared = squared + cardinality_discrepancy_squared(
                predicted_sizes, clean_sizes, predicted.shape[1]
            )
            sensitivities[start:stop, repetition] = (
                squared.detach().cpu().numpy().astype(np.float64) ** 0.5
            ) / relative_perturbation
    return np.median(sensitivities, axis=1)


def quantize_from_validation(
    validation_latent: np.ndarray,
    test_latent: np.ndarray,
    bits: int,
    standardized_clip: float,
    scale_ddof: int,
    scale_floor_relative: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    validation64 = np.asarray(validation_latent, dtype=np.float64)
    test64 = np.asarray(test_latent, dtype=np.float64)
    mean = validation64.mean(axis=0)
    raw_scale = validation64.std(axis=0, ddof=scale_ddof)
    largest_scale = float(raw_scale.max(initial=0.0))
    scale_floor = max(largest_scale * scale_floor_relative, np.finfo(np.float64).eps)
    floored = raw_scale < scale_floor
    scale = np.maximum(raw_scale, scale_floor)
    standardized_validation = (validation64 - mean) / scale
    standardized_test = (test64 - mean) / scale
    positive_integer_limit = 2 ** (int(bits) - 1) - 1
    step = standardized_clip / positive_integer_limit
    clipped = np.clip(standardized_test, -standardized_clip, standardized_clip)
    quantized_integer = np.clip(
        np.round(clipped / step), -positive_integer_limit, positive_integer_limit
    )
    quantized_standardized = quantized_integer * step
    quantized = quantized_standardized * scale + mean
    metadata = {
        "bits": int(bits),
        "standardized_clip": float(standardized_clip),
        "standardized_step": float(step),
        "quantization_scheme": "signed_symmetric_midtread",
        "usable_level_count": int(2 * positive_integer_limit + 1),
        "validation_clipping_rate": float(
            np.mean(np.abs(standardized_validation) > standardized_clip)
        ),
        "test_clipping_rate": float(np.mean(np.abs(standardized_test) > standardized_clip)),
        "floored_coordinate_count": int(floored.sum()),
        "coordinate_count": int(raw_scale.size),
        "scale_floor": float(scale_floor),
        "scale_ddof": int(scale_ddof),
        "scale_floor_relative": float(scale_floor_relative),
    }
    return quantized.astype(np.float32), metadata


def train_one_run(
    config: dict[str, Any],
    family: str,
    domain: str,
    n: int,
    width: int,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    splits = config["splits"]
    anchor_np = np.asarray(config["anchor"], dtype=np.float32)
    anchor = torch.from_numpy(anchor_np).to(device)
    inner = float(config["annulus_inner_radius"])
    dataset_seed = stable_seed("dataset", seed, domain, n)
    train_split = make_split(domain, n, int(splits["train"]), stable_seed(dataset_seed, "train"), anchor_np, inner)
    validation_split = make_split(domain, n, int(splits["validation"]), stable_seed(dataset_seed, "validation"), anchor_np, inner)
    test_split = make_split(domain, n, int(splits["test"]), stable_seed(dataset_seed, "test"), anchor_np, inner)

    model_seed = stable_seed("model", seed, family, domain, n, width)
    set_determinism(model_seed)
    hidden = int(training["hidden_width"])
    depth = int(training["hidden_depth"])
    if family == "mlp":
        encoder: nn.Module = PointMLPSum(width, hidden, depth)
    elif family == "theorem_projection":
        encoder = TheoremProjectionSum(
            n,
            width,
            tuple(config["anchor"]),
        )
    else:
        raise ValueError(f"Unknown encoder family: {family}")
    model = ReconstructionModel(encoder, Decoder(width, n, hidden, depth)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    permutations = all_permutations(n, device)
    radius = domain_diameter_with_anchor(domain, anchor_np)
    train_points, train_mask, train_targets, train_sizes = to_tensor(train_split, device)
    batch_size = int(training["batch_size"])
    card_weight = float(training["cardinality_loss_weight"])

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_validation = math.inf
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(int(training["epochs"])):
        model.train()
        epoch_generator = torch.Generator(device="cpu")
        epoch_generator.manual_seed(stable_seed("shuffle", seed, domain, n, epoch))
        order = torch.randperm(train_points.shape[0], generator=epoch_generator).to(device)
        running_loss = 0.0
        for start in range(0, order.numel(), batch_size):
            indices = order[start : start + batch_size]
            _, predicted, logits = model(train_points[indices], train_mask[indices])
            matching = assignment_squared(predicted, train_targets[indices], permutations, radius)
            loss = matching.mean() + card_weight * F.cross_entropy(logits, train_sizes[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss for family={family}, domain={domain}, n={n}, "
                    f"width={width}, seed={seed}, epoch={epoch + 1}."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * indices.numel()
        validation = evaluate(model, validation_split, batch_size, permutations, radius, anchor, device)
        record = {
            "epoch": float(epoch + 1),
            "training_loss": running_loss / train_points.shape[0],
            "validation_amrmse": float(validation["amrmse"]),
            "validation_cardinality_accuracy": float(validation["cardinality_accuracy"]),
        }
        history.append(record)
        if record["validation_amrmse"] < best_validation:
            best_validation = record["validation_amrmse"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    test = evaluate(model, test_split, batch_size, permutations, radius, anchor, device)
    validation = evaluate(model, validation_split, batch_size, permutations, radius, anchor, device)

    centered = validation["latent"].astype(np.float64) - validation["latent"].mean(
        axis=0, keepdims=True, dtype=np.float64
    )
    latent_scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1), dtype=np.float64)))
    nearest = normalized_nearest_neighbor_distances(test["latent"], test_split, latent_scale)
    finite_nearest = nearest[np.isfinite(nearest)]

    diagnostics = config["diagnostics"]
    nearest_quantile = float(diagnostics["nearest_code_quantile"])
    quantile_method = str(diagnostics["quantile_method"])
    relative_perturbation = float(diagnostics["relative_perturbation"])
    perturbation_count = int(diagnostics["perturbations_per_example"])
    sensitivity = local_decoder_sensitivity(
        model.decoder,
        test["latent"],
        test["predicted"],
        test["predicted_sizes"],
        latent_scale,
        relative_perturbation,
        perturbation_count,
        batch_size,
        permutations,
        radius,
        anchor,
        device,
        stable_seed("local_sensitivity", seed, family, domain, n, width),
    )
    finite_sensitivity = sensitivity[np.isfinite(sensitivity)]

    quantization_records: list[dict[str, Any]] = []
    quantization_errors: dict[int, np.ndarray] = {}
    quantization_sizes: dict[int, np.ndarray] = {}
    quantization_predictions: dict[int, np.ndarray] = {}
    standardized_clip = float(diagnostics["standardized_clip"])
    scale_ddof = int(diagnostics["quantization_scale_ddof"])
    scale_floor_relative = float(diagnostics["quantization_scale_floor_relative"])
    for bit_value in diagnostics["quantization_bits"]:
        bits = int(bit_value)
        quantized_latent, quantization_metadata = quantize_from_validation(
            validation["latent"],
            test["latent"],
            bits,
            standardized_clip,
            scale_ddof,
            scale_floor_relative,
        )
        quantized_evaluation = evaluate_latent_codes(
            model.decoder,
            quantized_latent,
            test_split.targets,
            test_split.sizes,
            batch_size,
            permutations,
            radius,
            anchor,
            device,
        )
        quantization_metadata.update(
            {
                "amrmse": quantized_evaluation["amrmse"],
                "cardinality_accuracy": quantized_evaluation["cardinality_accuracy"],
                "paired_amrmse_change": quantized_evaluation["amrmse"] - test["amrmse"],
            }
        )
        quantization_records.append(quantization_metadata)
        quantization_errors[bits] = quantized_evaluation["squared_errors"]
        quantization_sizes[bits] = quantized_evaluation["predicted_sizes"]
        quantization_predictions[bits] = quantized_evaluation["predicted"]

    run_id = f"{family}__{domain}__n{n}__m{width}__seed{seed}"
    run_dir = output / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if config["artifacts"].get("save_checkpoints", True):
        checkpoint_path = run_dir / "best_checkpoint.pt"
        temporary_checkpoint = run_dir / ".best_checkpoint.tmp.pt"
        torch.save({"state_dict": best_state, "config": config, "run_id": run_id}, temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)
    if config["artifacts"].get("save_per_example_predictions", True):
        prediction_path = run_dir / "test_predictions.npz"
        temporary_prediction_path = run_dir / ".test_predictions.tmp.npz"
        quantization_arrays = {
            f"quantized_{bits}_squared_errors": errors for bits, errors in quantization_errors.items()
        }
        quantization_arrays.update(
            {
                f"quantized_{bits}_predicted_sizes": sizes
                for bits, sizes in quantization_sizes.items()
            }
        )
        quantization_arrays.update(
            {
                f"quantized_{bits}_predicted": predicted
                for bits, predicted in quantization_predictions.items()
            }
        )
        np.savez_compressed(
            temporary_prediction_path,
            target=test_split.targets,
            target_sizes=test_split.sizes,
            predicted=test["predicted"],
            predicted_sizes=test["predicted_sizes"],
            squared_errors=test["squared_errors"],
            latent=test["latent"],
            normalized_nearest=nearest,
            local_sensitivity=sensitivity,
            **quantization_arrays,
        )
        os.replace(temporary_prediction_path, prediction_path)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "config_sha256": config_digest(config),
        "family": family,
        "domain": domain,
        "n": n,
        "width": width,
        "width_offset": width - 2 * n,
        "seed": seed,
        "dataset_seed": dataset_seed,
        "model_seed": model_seed,
        "best_epoch": best_epoch,
        "best_validation_amrmse": best_validation,
        "test_amrmse": test["amrmse"],
        "test_cardinality_accuracy": test["cardinality_accuracy"],
        "latent_scale": latent_scale,
        "normalized_nearest_valid_count": int(finite_nearest.size),
        "normalized_nearest_p05": finite_or_none(
            np.quantile(finite_nearest, nearest_quantile, method=quantile_method)
        )
        if finite_nearest.size
        else None,
        "normalized_nearest_median": finite_or_none(
            np.quantile(finite_nearest, 0.5, method=quantile_method)
        )
        if finite_nearest.size
        else None,
        "relative_perturbation": relative_perturbation,
        "perturbations_per_example": perturbation_count,
        "local_sensitivity_valid_count": int(finite_sensitivity.size),
        "local_sensitivity_median": finite_or_none(np.median(finite_sensitivity))
        if finite_sensitivity.size
        else None,
        "local_sensitivity_p95": finite_or_none(np.quantile(finite_sensitivity, 0.95))
        if finite_sensitivity.size
        else None,
        "quantization": quantization_records,
        "duration_seconds": time.time() - started,
        "history": history,
    }
    write_json(run_dir / "record.json", record)
    return record


def newton_root_decode(
    code: np.ndarray,
    n: int,
    anchor: complex,
    tau: float,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    moments = np.empty(n, dtype=np.complex128)
    for degree in range(1, n + 1):
        moments[degree - 1] = n * anchor**degree + code[degree - 1]
    elementary = np.zeros(n + 1, dtype=np.complex128)
    elementary[0] = 1.0
    for degree in range(1, n + 1):
        total = 0.0j
        for index in range(1, degree + 1):
            total += ((-1) ** (index - 1)) * elementary[degree - index] * moments[index - 1]
        elementary[degree] = total / degree
    coefficients = np.asarray([((-1) ** degree) * elementary[degree] for degree in range(n + 1)])
    roots = np.roots(coefficients)
    is_padding = np.abs(roots - anchor) < tau / 2.0
    data_roots = roots[~is_padding]
    decoded = np.stack((data_roots.real, data_roots.imag), axis=1) if data_roots.size else np.empty((0, 2))
    return decoded, int(data_roots.size), roots, coefficients


def encode_theorem_np(points: np.ndarray, n: int, anchor: complex) -> np.ndarray:
    values = points[:, 0].astype(np.float64) + 1j * points[:, 1].astype(np.float64)
    return np.asarray([np.sum(values**degree - anchor**degree) for degree in range(1, n + 1)])


def matching_squared_np(first: np.ndarray, second: np.ndarray, radius: float) -> float:
    n = first.shape[0]
    best = math.inf
    for permutation in itertools.permutations(range(n)):
        difference = first - second[np.asarray(permutation)]
        best = min(best, float(np.sum(difference * difference)))
    return best / (n * radius * radius)


def maximum_normalized_polynomial_residual(coefficients: np.ndarray, roots: np.ndarray) -> float:
    if roots.size == 0:
        return 0.0
    degree = coefficients.size - 1
    powers = np.arange(degree, -1, -1)
    residuals: list[float] = []
    for root in roots:
        numerator = abs(np.polyval(coefficients, root))
        denominator = float(np.sum(np.abs(coefficients) * (abs(root) ** powers)))
        residuals.append(float(numerator / max(denominator, np.finfo(np.float64).tiny)))
    return max(residuals)


def run_analytic_checks(config: dict[str, Any], output: Path, smoke: bool) -> list[dict[str, Any]]:
    anchor_xy = np.asarray(config["anchor"], dtype=np.float64)
    anchor = complex(anchor_xy[0], anchor_xy[1])
    inner = float(config["annulus_inner_radius"])
    sample_count = 32 if smoke else int(config["splits"]["analytic_test"])
    tolerance = float(
        config["diagnostics"].get(
            "normalized_recovery_tolerance",
            config["diagnostics"].get("root_tolerance", 1e-6),
        )
    )
    records: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    domains = config["domains"]
    caps = config["cardinality_caps"]
    for domain in domains:
        radius = domain_diameter_with_anchor(domain, anchor_xy)
        tau = domain_anchor_distance(domain, anchor_xy, inner)
        for n in caps:
            split = make_split(
                domain,
                int(n),
                sample_count,
                stable_seed("analytic", domain, n),
                anchor_xy.astype(np.float32),
                inner,
            )
            squared: list[float] = []
            padded_root_squared: list[float] = []
            residuals: list[float] = []
            cardinality_ok = 0
            within_tolerance = 0
            for row, size in enumerate(split.sizes.tolist()):
                data = split.points[row, :size]
                code = encode_theorem_np(data, int(n), anchor)
                decoded, decoded_size, roots, coefficients = newton_root_decode(code, int(n), anchor, tau)
                predicted_padded = np.broadcast_to(anchor_xy, (int(n), 2)).copy()
                predicted_padded[:decoded_size] = decoded
                error_squared = matching_squared_np(
                    predicted_padded, split.targets[row].astype(np.float64), radius
                ) + ((decoded_size - int(size)) / float(n)) ** 2
                root_coordinates = np.stack((roots.real, roots.imag), axis=1)
                padded_error_squared = matching_squared_np(
                    root_coordinates, split.targets[row].astype(np.float64), radius
                )
                residual = maximum_normalized_polynomial_residual(coefficients, roots)
                squared.append(error_squared)
                padded_root_squared.append(padded_error_squared)
                residuals.append(residual)
                cardinality_ok += int(decoded_size == size)
                within_tolerance += int(math.sqrt(error_squared) <= tolerance)
                example_rows.append(
                    {
                        "domain": domain,
                        "n": int(n),
                        "example_index": row,
                        "true_size": int(size),
                        "decoded_size": decoded_size,
                        "cardinality_correct": int(decoded_size == size),
                        "within_tolerance": int(math.sqrt(error_squared) <= tolerance),
                        "normalized_anchor_error": math.sqrt(error_squared),
                        "normalized_padded_root_error": math.sqrt(padded_error_squared),
                        "maximum_normalized_polynomial_residual": residual,
                    }
                )
            records.append(
                {
                    "domain": domain,
                    "n": int(n),
                    "sample_count": sample_count,
                    "within_tolerance_count": within_tolerance,
                    "cardinality_correct_count": cardinality_ok,
                    "amrmse": float(math.sqrt(float(np.mean(squared)))),
                    "maximum_anchor_matching_error": float(math.sqrt(max(squared))),
                    "padded_root_amrmse": float(math.sqrt(float(np.mean(padded_root_squared)))),
                    "maximum_padded_root_error": float(math.sqrt(max(padded_root_squared))),
                    "maximum_normalized_polynomial_residual": float(max(residuals)),
                    "anchor_distance": tau,
                    "normalization_radius": radius,
                    "tolerance": tolerance,
                }
            )
    write_json(output / "analytic_checks.json", records)
    write_csv(
        output / "analytic_examples.csv",
        [
            "domain",
            "n",
            "example_index",
            "true_size",
            "decoded_size",
            "cardinality_correct",
            "within_tolerance",
            "normalized_anchor_error",
            "normalized_padded_root_error",
            "maximum_normalized_polynomial_residual",
        ],
        example_rows,
    )
    return records


def deterministic_fixture_points(domain: str, n: int) -> list[tuple[str, np.ndarray]]:
    if domain == "square":
        first = np.asarray((0.37, 0.61), dtype=np.float64)
        second = np.asarray((0.83, 0.29), dtype=np.float64)
        parameter = np.linspace(0.0, 1.0, num=n, dtype=np.float64)
        distinct = np.stack((parameter, np.zeros_like(parameter)), axis=1)
    elif domain in {"disk", "annulus"}:
        first = np.asarray((math.cos(0.37), math.sin(0.37)), dtype=np.float64)
        second = np.asarray((math.cos(2.11), math.sin(2.11)), dtype=np.float64)
        angles = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n
        distinct = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    repeated = np.repeat(first[None, :], repeats=n, axis=0)
    mixed = np.concatenate((np.repeat(first[None, :], repeats=n - 1, axis=0), second[None, :]), axis=0)
    return [
        ("empty", np.empty((0, 2), dtype=np.float64)),
        ("maximal_coincident_multiplicity", repeated),
        ("two_support_maximal_cardinality", mixed),
        ("distinct_boundary_maximal_cardinality", distinct),
    ]


def run_analytic_fixture_checks(config: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    anchor_xy = np.asarray(config["anchor"], dtype=np.float64)
    anchor = complex(anchor_xy[0], anchor_xy[1])
    inner = float(config["annulus_inner_radius"])
    tolerance = float(config["diagnostics"]["normalized_recovery_tolerance"])
    records: list[dict[str, Any]] = []
    for domain in config["domains"]:
        radius = domain_diameter_with_anchor(domain, anchor_xy)
        tau = domain_anchor_distance(domain, anchor_xy, inner)
        for n_value in config["cardinality_caps"]:
            n = int(n_value)
            for fixture_name, points in deterministic_fixture_points(domain, n):
                true_size = int(points.shape[0])
                target = np.broadcast_to(anchor_xy, (n, 2)).copy()
                target[:true_size] = points
                code = encode_theorem_np(points, n, anchor)
                decoded, decoded_size, roots, coefficients = newton_root_decode(code, n, anchor, tau)
                predicted = np.broadcast_to(anchor_xy, (n, 2)).copy()
                predicted[:decoded_size] = decoded
                error_squared = matching_squared_np(predicted, target, radius) + (
                    (decoded_size - true_size) / float(n)
                ) ** 2
                root_coordinates = np.stack((roots.real, roots.imag), axis=1)
                padded_root_error_squared = matching_squared_np(root_coordinates, target, radius)
                residual = maximum_normalized_polynomial_residual(coefficients, roots)
                records.append(
                    {
                        "domain": domain,
                        "n": n,
                        "fixture": fixture_name,
                        "true_size": true_size,
                        "decoded_size": decoded_size,
                        "cardinality_correct": int(decoded_size == true_size),
                        "within_tolerance": int(math.sqrt(error_squared) <= tolerance),
                        "normalized_anchor_error": math.sqrt(error_squared),
                        "normalized_padded_root_error": math.sqrt(padded_root_error_squared),
                        "maximum_normalized_polynomial_residual": residual,
                        "tolerance": tolerance,
                    }
                )
    write_json(output / "analytic_fixture_checks.json", records)
    write_csv(
        output / "analytic_fixture_checks.csv",
        [
            "domain",
            "n",
            "fixture",
            "true_size",
            "decoded_size",
            "cardinality_correct",
            "within_tolerance",
            "normalized_anchor_error",
            "normalized_padded_root_error",
            "maximum_normalized_polynomial_residual",
            "tolerance",
        ],
        records,
    )
    return records


def capture_environment(output: Path, device: torch.device) -> None:
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True
    ).stdout.splitlines()
    numpy_configuration = io.StringIO()
    with contextlib.redirect_stdout(numpy_configuration):
        np.show_config()
    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_smi = (
        subprocess.run(
            [
                nvidia_smi_path,
                "--query-gpu=name,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if nvidia_smi_path
        else None
    )
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "numpy_configuration": numpy_configuration.getvalue().splitlines(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "nvidia_smi": nvidia_smi.stdout.splitlines()
        if nvidia_smi is not None and nvidia_smi.returncode == 0
        else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "packages": packages,
    }
    path = output / "environment.json"
    if path.exists():
        existing = load_json(path)
        if existing != environment:
            raise RuntimeError(
                "The Python, package, platform, or accelerator environment changed; refusing to mix "
                "resumed runs. Use a new output directory."
            )
    else:
        write_json(path, environment)


def snapshot_package(output: Path, original_config_path: Path) -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    source_dir = output / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "run_revision_experiments.py",
        "aggregate_results.py",
        "verify_artifacts.py",
        "requirements.txt",
        "README.md",
        "RUN_SMOKE.cmd",
        "RUN_FULL.cmd",
        "RUN_FINALIZE_AND_SYNC.cmd",
    )
    copied: dict[str, str] = {}
    for name in names:
        source = package_dir / name
        if source.exists():
            destination = source_dir / name
            shutil.copy2(source, destination)
            copied[name] = sha256_file(destination)
    config_destination = source_dir / "submitted_config.json"
    shutil.copy2(original_config_path, config_destination)
    copied[config_destination.name] = sha256_file(config_destination)
    write_json(source_dir / "source_hashes.json", copied)
    return copied


def existing_run_records(output: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    runs_dir = output / "runs"
    if not runs_dir.exists():
        return records
    for path in sorted(runs_dir.glob("*/record.json")):
        record = load_json(path)
        run_id = str(record.get("run_id", ""))
        if not run_id:
            raise RuntimeError(f"Missing run_id in {path}")
        if run_id in records:
            raise RuntimeError(f"Duplicate run_id {run_id!r}")
        records[run_id] = record
    return records


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return device


def validate_runtime(config: dict[str, Any], device: torch.device, smoke: bool) -> None:
    runtime = config["runtime"]
    expected_python = tuple(int(value) for value in runtime["python_major_minor"])
    if sys.version_info[:2] != expected_python:
        raise SystemExit(
            f"Python {expected_python[0]}.{expected_python[1]} is required; found "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )
    actual_torch = str(torch.__version__).split("+", 1)[0]
    if actual_torch != str(runtime["torch"]):
        raise SystemExit(f"PyTorch {runtime['torch']} is required; found {torch.__version__}.")
    if str(np.__version__) != str(runtime["numpy"]):
        raise SystemExit(f"NumPy {runtime['numpy']} is required; found {np.__version__}.")
    if runtime.get("require_cuda_for_full_run", True) and not smoke and device.type != "cuda":
        raise SystemExit("The full revision grid requires a CUDA device under the declared protocol.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a compatible interrupted run; records with the matching configuration digest "
            "are skipped and the complete artifact is verified after aggregation."
        ),
    )
    args = parser.parse_args()

    submitted_config = load_json(args.config)
    validate_config(submitted_config)
    config = smoke_config(submitted_config) if args.smoke else submitted_config
    validate_config(config)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(
            f"Output directory is not empty: {output}. Use a new directory or --resume; files are never deleted."
        )
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and any(output.iterdir()) and not (output / "resolved_config.json").is_file():
        raise SystemExit("Cannot resume a nonempty directory without resolved_config.json.")
    digest = config_digest(config)
    resolved_config_path = output / "resolved_config.json"
    if resolved_config_path.exists():
        existing_config = load_json(resolved_config_path)
        if config_digest(existing_config) != digest:
            raise SystemExit("The existing output uses a different resolved configuration; refusing to resume.")
    provenance_path = output / "provenance.json"
    runner_hash = sha256_file(Path(__file__).resolve())
    previous_provenance: dict[str, Any] | None = None
    if provenance_path.exists():
        previous_provenance = load_json(provenance_path)
        if previous_provenance.get("runner_sha256") != runner_hash:
            raise SystemExit("The experiment runner changed since the interrupted run; refusing to mix results.")
    device = choose_device(args.device)
    validate_runtime(config, device, args.smoke)
    torch.use_deterministic_algorithms(True)
    write_json(resolved_config_path, config)
    source_hashes = snapshot_package(output, args.config.resolve())
    write_json(
        provenance_path,
        {
            "schema_version": 1,
            "initial_argv": previous_provenance.get("initial_argv")
            if previous_provenance
            else [str(item) for item in sys.argv],
            "last_argv": [str(item) for item in sys.argv],
            "invocation_count": int(previous_provenance.get("invocation_count", 0)) + 1
            if previous_provenance
            else 1,
            "config_sha256": digest,
            "runner_sha256": runner_hash,
            "source_hashes": source_hashes,
            "smoke_mode": bool(args.smoke),
        },
    )
    capture_environment(output, device)
    run_analytic_checks(config, output, args.smoke)
    run_analytic_fixture_checks(config, output)

    domains: Iterable[str] = config["domains"]
    caps: Iterable[int] = config["cardinality_caps"]
    families: Iterable[str] = config["encoder_families"]
    seeds: Iterable[int] = config["seeds"]
    offsets: Iterable[int] = config["width_offsets"]
    existing = existing_run_records(output) if args.resume else {}
    for run_id, record in existing.items():
        if record.get("config_sha256") != digest:
            raise SystemExit(f"Existing record {run_id!r} has an incompatible configuration hash.")
    records: dict[str, dict[str, Any]] = dict(existing)
    completed_now = 0
    total_expected = (
        len(config["domains"])
        * len(config["cardinality_caps"])
        * len(config["width_offsets"])
        * len(config["encoder_families"])
        * len(config["seeds"])
    )
    for domain in domains:
        for n in caps:
            for offset in offsets:
                width = 2 * int(n) + int(offset)
                for family in families:
                    for seed in seeds:
                        run_id = f"{family}__{domain}__n{int(n)}__m{width}__seed{int(seed)}"
                        if run_id in records:
                            continue
                        print(
                            f"[{len(records) + 1}/{total_expected}] starting {run_id}",
                            flush=True,
                        )
                        records[run_id] = train_one_run(
                            config, family, domain, int(n), width, int(seed), output, device
                        )
                        completed_now += 1
                        write_json(
                            output / "run_index.json",
                            [records[key] for key in sorted(records)],
                        )
                        print(
                            f"[{len(records)}/{total_expected}] completed {run_id} in "
                            f"{records[run_id]['duration_seconds'] / 60.0:.1f} min",
                            flush=True,
                        )
    write_json(output / "run_index.json", [records[key] for key in sorted(records)])
    print(
        f"Available learned runs: {len(records)}; completed in this invocation: {completed_now}; "
        f"device: {device}; results: {output}"
    )


if __name__ == "__main__":
    main()
