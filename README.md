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
bash assets/scripts/downloads/pep_compass/hydramp/download_hydramp_model.sh
```

> Each script installs weights below
> `src/pep_compass/autoencoder/strategies/hydramp/models/<model>`.

#### APEX Model


```bash
# model: default (8-pathogen article set)
assets/scripts/downloads/pep_compass/apex/download_apex_models_default.sh

# model: full (34-pathogen set, ~1 GB)
assets/scripts/downloads/pep_compass/apex/download_apex_models_full.sh
```

> Each script installs weights below
> `src/pep_compass/steps/oracles/strategies/apex/models/<model>`.

## Python API

```python
from pep_compass import Pipeline
from pep_compass.autoencoder import HydrAMP
import pep_compass.steps as ps  # ps - PipelineSteps

pipeline = Pipeline(
    [ps.filters.CandidateSubset(count=3), ps.oracles.Hydrophobicity()],
    autoencoder=HydrAMP(jacobian_mode="approx"),
)
pipeline.describe_component(ps.filters.CandidateSubset)
pipeline.show()
compiled = pipeline.build()
result = pipeline.run(["FLYKWWIRIGRLKL"], seed=7)
print(result.reader.get_archive())
result.close()
```

See [the direct API tutorial](docs/artefacts/wrapper_new.md) and
[the executable notebook](docs/tutorial/wrapper_tutorial.ipynb).

## Run

```bash
uv run --extra cpu pep-compass plan pipeline=simple
uv run --extra cpu pep-compass run pipeline=simple ++runner.execution.seed=7
uv run --extra cpu pep-compass run --multirun pipeline=simple ++runner.execution.seed=7,8
```

The Hydra CLI lives in `src/pep_compass/orchestrator/cli.py` and executes the public
Python wrapper. See the [orchestrator guide](docs/artefacts/orchestrator.md) for YAML
composition, independent workers, Slurm templates and migration from old commands.

## Guides

- [User Guide](docs/user-guide.md) — configure and execute an experiment.
- [Analysis Guide](docs/artefacts/analysis-guide.md) — inspect persisted results with
  `analysis.reader`.
- [Technical Architecture](docs/technical-architecture.md) — inspect internal
  component and data flow.
- [Developer Guide](docs/developer-guide.md) — extend and validate the package.

## Documentation

The complete documentation entry page is [docs/README.md](docs/README.md).

- [User Guide](docs/user-guide.md)
- [Technical Architecture](docs/technical-architecture.md)
- [Developer Guide](docs/developer-guide.md)
- [Analysis Guide](docs/artefacts/analysis-guide.md)
- [Architecture Decisions](docs/architecture-decisions.md)

## Citation

If you use our model cite [this article](https://www.researchgate.net/publication/396142806_PepCompass_Navigating_peptide_embedding_spaces_using_Riemannian_Geometry)
