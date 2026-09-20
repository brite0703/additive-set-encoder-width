# Reproducing the revised Neural Networks results

Manuscript: NEUNET-D-26-03059. This guide accompanies the first revision of the manuscript and its response to reviewers. The verified numerical results and their immutable source snapshots are unchanged.

The distribution file is `additive_set_encoder_width_reproducibility.zip`. Extract it and open its `additive_set_encoder_width_reproducibility` folder. The paths below also apply inside that folder. Its `README.md` provides the entry points, and `package_manifest.sha256` covers the packaged files. The original result manifest remains inside the result directory.

The complete result directory is `reproducibility/revision_results`, relative to this guide. It contains 960 learned runs, prediction arrays, checkpoints, source snapshots, configuration, aggregates, and `manifest.sha256`. The reference file `NN_verified_results.tex` is an exact copy of `reproducibility/revision_results/aggregate/manuscript_macros.tex`. The current manuscript source embeds these complete numerical definitions verbatim and compiles without a separate numerical file. The reference copy remains in the reproducibility distribution for auditing; it is not a manuscript submission dependency.

The obsolete sorted-target, zero-padding protocol and duplicate archives were removed during the 20 September 2026 cleanup. The canonical result directory and its source snapshots are unchanged. Use the source snapshots inside that directory for the reviewer-corrected protocol.

## Verify the supplied results

Open a terminal in `reproducibility/revision_results`. Use Python with NumPy installed. Verification reads the existing files and checks their manifest; it does not require PyTorch or a GPU.

```text
python source/verify_artifacts.py --root .
```

The expected successful result is verification of 960 learned runs and 2,906 hashed artifacts. Keep the contents of `revision_results` unchanged, including its source snapshots, so that subsequent manifest checks remain meaningful.

## Recompute the tables

Make a separate working copy of `reproducibility/revision_results` and open a terminal in that copy. The canonical aggregator requires its output to be the `aggregate` directory immediately inside its input root, so regeneration should use this working copy:

```text
python source/aggregate_results.py --input . --output aggregate
python source/verify_artifacts.py --root .
```

Compare the generated CSV files and `manuscript_macros.tex` with the preserved original copy's `aggregate` directory. A byte-identical regeneration also passes the original manifest. Differences on another software environment should be inspected before replacing any reported value. The reporting program is `source/aggregate_results.py` inside the canonical result directory.

The 20 September audit independently recomputed 10,240 aggregate statistics from the stored records, with maximum absolute difference `7.11e-15`. The manuscript's numerical fragment is byte-identical to the canonical aggregate. Small floating-point differences across environments can cause a regenerated copy to fail the original manifest even when its reported table values agree. The untouched supplied result directory passed its manifest check. Retain it rather than replacing its manifest with one for the working copy.

## Regenerate experiments

The archived configuration is named `source/submitted_config.json`. The historical README and Windows launchers use its original name, `config_revision.json`; that original filename is not present inside the source snapshot. The following commands use the supplied filename directly.

Use a Python 3.11 environment with NumPy 2.3.5 and CUDA-enabled PyTorch 2.9.0. The recorded execution used Python 3.11.9, PyTorch 2.9.0+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 3060 Laptop GPU. Install the PyTorch build appropriate to the available driver, then install the archived requirements:

```text
python -m pip install -r source/requirements.txt
```

Start with a smoke run in a new directory:

```text
python source/run_revision_experiments.py --config source/submitted_config.json --output ../rerun_smoke --smoke
python source/aggregate_results.py --input ../rerun_smoke --output ../rerun_smoke/aggregate
python source/verify_artifacts.py --root ../rerun_smoke --write-manifest
python source/verify_artifacts.py --root ../rerun_smoke
```

For the complete experiment, choose a fresh output directory on a local SSD. The example below uses a sibling directory; substitute an absolute local path if the result directory is on synchronized storage.

```text
python source/run_revision_experiments.py --config source/submitted_config.json --output ../rerun_full
python source/aggregate_results.py --input ../rerun_full --output ../rerun_full/aggregate
python source/verify_artifacts.py --root ../rerun_full --write-manifest
python source/verify_artifacts.py --root ../rerun_full
```

Use `--resume` only when continuing that newly generated output with the same runner and configuration. Smoke and full results have separate output directories. The archived Windows launchers preserve the original workstation setup; the commands above are the entry points for the archived snapshot.

## Scope of verification

The 20 September 2026 audit reran the artifact verifier for all 960 runs and 2,906 hashed artifacts, independently recomputed 10,240 aggregate statistics, and checked 1,966,080 clean and quantized per-example assignment errors across all 960 prediction archives by subset dynamic programming. It also recomputed nearest-code distances in the displayed disk slice and checked the pairing of test inputs across all widths and families. The verification program and results are in `audit/`. These checks support the stored evidence. They do not repeat training or checkpoint inference, independently establish historical execution provenance, or certify identical numerical results across different hardware and software environments.
