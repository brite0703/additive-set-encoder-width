# Neural Networks revision experiment package

This package regenerates the numerical evidence for manuscript `NEUNET-D-26-03059` under the
reviewer-corrected protocol. No numerical value from the submitted manuscript is an input to these
programs. Do not update the manuscript until a complete remote run, deterministic aggregation, and
artifact verification have all succeeded.

## Corrected estimands

1. Training minimizes exact optimal assignment to an anchor-padded multiset. It never selects a
   lexicographic target ordering.
2. Evaluation uses normalized, multiplicity-aware anchor matching plus the explicit squared
   cardinality discrepancy `((s - s-hat) / n)^2`. This term guarantees positive cost for every
   cardinality error even though raw neural coordinates are unconstrained. The anchor `(2, 0)` is
   outside every configured planar domain, and empty-empty is defined as zero. At inference, the
   cardinality head supplies `s-hat`; the `s-hat` candidate coordinates farthest from the anchor are
   retained, and the rest are replaced by exact anchor copies before matching.
3. No finite random-pair collision rate is computed. Exact injectivity remains a theorem, not an
   empirical estimand. Nearest-code separation excludes identical input multisets and is normalized by
   a validation-derived latent scale.
4. Decoder perturbations have fixed norm relative to that validation latent scale. Quantization uses
   validation means and coordinate scales plus a prespecified standardized clipping range and a
   signed symmetric midtread grid (which represents standardized zero exactly); validation and test clipping rates
   are retained. Coordinate scales use the validation population standard deviation (`ddof=0`) and
   a floor equal to the larger of `1e-6` times the largest coordinate scale and float64 machine
   epsilon. Reported nearest-code quantiles use NumPy's linear interpolation rule.
5. The exact theorem encoder is assessed separately by the constructive Newton-root check. In the
   learned width grid, the theorem-feature family uses a learned linear projection at *every* width,
   including `2n`, so the width comparison does not introduce a special identity layer only at the
   theorem width.

## Declared scope

The full configuration contains

```text
3 domains x 4 cardinality caps x 4 width offsets x 2 encoder families x 10 seeds
= 960 learned training runs.
```

Before execution, the `reporting` block fixes the displayed matched-width slice as
`domain=disk`, `n=4`, and `width_offset=0`, fixes four bits as the displayed quantization depth,
and declares equal-weight domain-cap pooling within seed followed by two-sided 95% Student-t
intervals across seeds.

Each seed is paired across encoder families and widths by using the same generated train, validation,
and test multisets for a fixed domain and cardinality cap. The constructive check contains
`3 x 4 x 1024 = 12,288` random sampled multisets. It also contains 48 deterministic fixtures: empty,
maximally coincident, two-support maximal-cardinality, and distinct boundary maximal-cardinality
multisets for every domain-cap pair. The fixture results are reported separately because random
continuous sampling almost surely does not test exact multiplicity. The aggregate metadata records the
summed run-hours after execution; no runtime estimate is presented as an observed value before the
remote run.

This is a large grid. Run the smoke test first, inspect its artifacts, and reserve adequate remote GPU
time and storage. Interrupted full runs can be resumed without overwriting completed records.

## Clean remote environment

Use a clean Python 3.11 environment. Install the CUDA build of PyTorch 2.9 appropriate for the
remote driver using the official PyTorch wheel index, then install the remaining pinned packages:

```text
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url <OFFICIAL_PYTORCH_CUDA_INDEX>
python -m pip install -r requirements.txt
```

`requirements.txt` intentionally does not contain `torch`; this prevents the second command from
replacing the explicitly selected CUDA wheel. The runner refuses a PyTorch version other than 2.9.0.
Verify `torch.version.cuda` and `torch.cuda.is_available()` before starting the grid.

Record the exact index URL and driver version in the run log. The runner captures Python, platform,
GPU, CUDA, cuDNN, PyTorch, the complete `pip freeze`, the resolved configuration, command line, source
snapshots, and SHA-256 source hashes. Strict deterministic algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` are enabled; a deterministic-kernel error must be resolved rather
than suppressed.

## Smoke test

From this directory:

```text
python run_revision_experiments.py --config config_revision.json --output smoke_results --smoke
python aggregate_results.py --input smoke_results --output smoke_results/aggregate
python verify_artifacts.py --root smoke_results --write-manifest
python verify_artifacts.py --root smoke_results
```

On the Windows remote workstation, `RUN_SMOKE.cmd` performs these four commands after checking the
fixed Python, NumPy, PyTorch, CUDA, and GPU environment.

Smoke mode writes its reduced grid into `resolved_config.json` (one domain, one cap, theorem-width
offset, one family, one seed, two epochs, and small splits). Smoke outputs must never be merged with the
full results.

## Full run and safe resume

Run from the synced source package, but write the 960-run working output to a remote-local NVMe/SSD
directory. Do not use a live Drive/Dropbox/OneDrive-synchronized directory for checkpoints, compressed
arrays, or the repeatedly replaced run index: synchronization latency and partial-file propagation can
substantially slow the grid and complicate recovery. `<LOCAL_FAST_ROOT>` below is a placeholder (for
example, a dedicated local scratch directory on the remote workstation), not a prescribed drive
letter. On Windows, keep this local root short enough to avoid legacy path-length limits.

Start a new full output directory:

```text
python run_revision_experiments.py --config config_revision.json --output <LOCAL_FAST_ROOT>/revision_results
```

`RUN_FULL.cmd` prompts the operator for this remote-local NVMe/SSD root and records the selected full
result path in `remote_local_run_root.txt`; it has no hard-coded user-profile or synchronized-storage
default. `RUN_FINALIZE_AND_SYNC.cmd` reads that path, aggregates and verifies the local result, copies
the complete artifact into this package, and verifies the synchronized copy.

If the process is interrupted, rerun the same source and configuration with:

```text
python run_revision_experiments.py --config config_revision.json --output <LOCAL_FAST_ROOT>/revision_results --resume
```

Resume is refused if the resolved configuration or runner hash differs. The runner never deletes an
existing output directory and writes JSON, checkpoints, prediction archives, and the run index using
replace-on-completion semantics.

After all 960 records exist:

```text
python aggregate_results.py --input <LOCAL_FAST_ROOT>/revision_results --output <LOCAL_FAST_ROOT>/revision_results/aggregate
python verify_artifacts.py --root <LOCAL_FAST_ROOT>/revision_results --write-manifest
python verify_artifacts.py --root <LOCAL_FAST_ROOT>/revision_results
```

The second verification call checks the newly written manifest rather than merely creating it.

## Deterministic result products

`aggregate_results.py` refuses incomplete or contaminated grids and writes:

- `aggregate/summary.csv`: one row per family, domain, cap, and width; mean, sample standard
  deviation, and two-sided 95% Student-t interval across the ten seeds for every declared metric;
- `aggregate/pooled_by_width_offset.csv`: equal-weight domain-cap averages computed within each seed,
  followed by uncertainty across seeds; this avoids treating the 12 conditions as independent
  replicates;
- `aggregate/paired_family_differences.csv`: seed-paired differences, with the direction recorded in
  the file;
- `aggregate/quantization_summary.csv`, `aggregate/analytic_checks.csv`, and
  `aggregate/analytic_fixture_checks.csv`;
- `aggregate/aggregate_metadata.json`: grid dimensions, run count, analytic sample count, pooling rule,
  and summed compute-hours;
- `aggregate/manuscript_macros.tex`: deterministic lookup macros generated from the CSV values.

The LaTeX fragment defines `\NNResult{key}`. For example, a condition-level value is addressed by a
key of the form

```text
condition:mlp:disk:n4:m8:testamrmsemean
```

and a pooled value by

```text
pooled:mlp:offset0:testamrmsemean
```

and a quantization result by a key such as

```text
quantization:mlp:disk:n4:m8:b4:pairedamrmsechangemean
```

The exact available keys are the generated macro definitions; manuscript values must be inserted from
this fragment or from the CSV files, never by retyping console output.

## Synchronization and integrity

Only after the local verification succeeds, copy the complete `revision_results/` directory into the
synced revision package. Synchronize `runs/`, `source/`, `aggregate/`, environment
and provenance records, per-example archives, checkpoints, and `manifest.sha256`. Do not copy only
figures or summary tables, and do not continue writing to the remote-local source after starting the
copy. After synchronization completes, run verification against the copied directory:

```text
python verify_artifacts.py --root <SYNCED_COPY>/revision_results
```

Verification checks the complete expected grid, configuration hashes, run histories, checkpoint and
prediction presence, array shapes and finite values, recomputed AMRMSE and cardinality accuracy,
analytic row counts, aggregate metadata, source snapshots, absence of incomplete temporary files, and
the SHA-256 hash of every artifact.

## Interpretation boundary

The finite-sample diagnostics check an implementation of the constructive decoder and compare learned
reconstruction under a specified protocol. They do not prove global injectivity, certify absence of
collisions, or establish a trainability phase transition at width `2n`. A zero observed failure count
must be reported as `0/N failures under the stated sampling distribution and tolerance`, not as exact
recovery on the whole multiset space.
