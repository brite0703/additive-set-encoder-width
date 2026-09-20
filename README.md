# Exact Planar Width Thresholds for Continuous Additive Deep Sets

Reproducibility materials for the manuscript by Jih-Jeng Huang and Chin-Yi Chen (NEUNET-D-26-03059).

## Complete package

Download **`additive_set_encoder_width_reproducibility.zip`** from the [final revision release](https://github.com/brite0703/additive-set-encoder-width/releases/tag/final-revision). This is the complete package, including the code, configuration, all 960 learned runs, checkpoints, per-example predictions, constructive decoder checks, aggregate tables, environment records, and reproduction instructions.

The ZIP is 243,872,494 bytes. Its SHA-256 checksum is:

```text
452fada89f91f1b0f6f9284029ad1d7b91e458de77a710c720793c0b0606fa05
```

Extract the archive and open a terminal in its `additive_set_encoder_width_reproducibility` folder. Then run:

```text
python -m pip install -r reproducibility/revision_results/source/requirements.txt
python reproducibility/revision_results/source/verify_artifacts.py --root reproducibility/revision_results
```

The expected result is verification of **960 learned runs and 2,906 hashed artifacts**. The result directory contains 2,907 files, including its manifest. These checks require Python and NumPy; they do not require a GPU or execute the saved checkpoints.

Use [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for table regeneration, smoke testing, and the complete training grid. The archived configuration is `source/submitted_config.json` relative to the result directory. The historical README and Windows launchers inside the source snapshot retain an earlier filename; use the current reproduction guide for commands.

The repository makes the source snapshot available for browsing. The complete numerical evidence is in the attached release ZIP. GitHub's automatically generated source-code archives contain only the repository files and do not contain the full experiment results.

## Verification and scope

The release archive was extracted into a fresh directory and passed the supplied artifact verifier. All archived payloads were also checked against their SHA-256 hashes. The manuscript's numerical fragment agrees byte for byte with the generated aggregate.

The recorded training environment was Python 3.11.9, NumPy 2.3.5, PyTorch 2.9.0+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 3060 Laptop GPU. Packaging and verification did not repeat training or checkpoint inference. Identical numerical results across different hardware and software are not guaranteed.

## Citation

Huang, J.-J., and Chen, C.-Y. (2026). *Reproducibility materials for Exact Planar Width Thresholds for Continuous Additive Deep Sets* [Data set and software]. GitHub, final revision release. https://github.com/brite0703/additive-set-encoder-width/releases/tag/final-revision
