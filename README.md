# PepCompass

## Introduction

PepCompass is a research library for peptide optimisation with latent-space
geometry, composable candidate-generation steps, decision filters and
biological oracles.

### Research Context

The repository contains implementations and migrated model code used to study
latent-space locality and peptide optimisation. Scientific equivalence of
migrated strategies must be validated against their source implementations and
associated research material before reported results are treated as reproduced.

## Installation

### Library 

PepCompass uses [uv](https://docs.astral.sh/uv/) for environment management.
Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create one supported environment:

```bash
# without torch
uv sync 

# CPU
uv sync --extra cpu

# CUDA 11.8
uv sync --extra cu118

# CUDA 12.6
uv sync --extra cu126

# CUDA 12.8
uv sync --extra cu128
```

### Models 
To be able to run experiments you need to download models weights. 

We provide helper scripts for downloading those models. If model is already present, script will not download it again. 

#### BattleAMP

```bash
# HydraAMP from article (peptides of length equal to 25)
bash assets/scripts/downloads/hydraAMP/download_hydramp_model.sh 
```

> Each script installs weights below
> `src/pep_compass/autoencoder/strategies/hydramp/models/<model>`.

#### APEX Model


```bash
# model: default (8-pathogen article set)
assets/scripts/downloads/apex/download_apex_models_default.sh  

# model: full (34-pathogen set, ~1 GB)
assets/scripts/downloads/apex/download_apex_models_full.sh     
```

> Each script installs weights below
> `src/pep_compass/optimization/components/oracles/strategies/apex/models/<model>`.

## Run

`pep-compass` is the console entry point (`src/pep_compass/runtime/cli.py`).
Verify a configuration without loading any model:

```bash
uv run --extra cu118 pep-compass dry-run \
  assets/experiments/configs/validation/optimization_flow_smoke.yaml
```

Then execute a small bounded run with real models and no persisted output:

```bash
uv run --extra cu118 pep-compass test-run \
  assets/experiments/configs/composable_example.yaml \
  --device cpu
```

And a full run, persisting results under `experiment.output_directory`:

```bash
uv run --extra cu118 pep-compass run \
  assets/experiments/configs/composable_example.yaml
```

Every command and flag (`--device`, `--run-index`, backends, Slurm) is
documented in [User Guide § Commands](docs/user-guide.md#commands).

## Guides

- [User Guide](docs/user-guide.md) — configure and execute an experiment.
- [Analysis Guide](docs/analysis-guide.md) — inspect persisted results with
  `analysis.reader`.
- [Technical Architecture](docs/technical-architecture.md) — inspect internal
  component and data flow.
- [Developer Guide](docs/developer-guide.md) — extend and validate the package.

## Documentation

The complete documentation entry page is [docs/README.md](docs/README.md).

- [User Guide](docs/user-guide.md)
- [Technical Architecture](docs/technical-architecture.md)
- [Developer Guide](docs/developer-guide.md)
- [Analysis Guide](docs/analysis-guide.md)
- [Architecture Decisions](docs/architecture-decisions.md)

## Citation

If you use our model cite [this article](https://www.researchgate.net/publication/396142806_PepCompass_Navigating_peptide_embedding_spaces_using_Riemannian_Geometry)