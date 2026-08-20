# Architecture Decisions

This document records *why* PepCompass is split the way it is, what was
deliberately deferred, and the currently known open issues. It condenses the
implementation discussion that drove the `core`/`optimization`/`runtime`/
`autoencoder`/`data`/`analysis` restructuring into decisions and current
status — read [Technical Architecture](technical-architecture.md) for how
the result actually works.

## Package split rationale

The starting problem was four responsibilities mixed into two packages:
describing/validating an experiment, building an executable optimisation
program, running one optimisation and a whole experiment plan, and reading
persisted results back for analysis. The chosen split makes each
responsibility a package with a one-directional dependency:

```text
runtime  ──►  core  ──►  optimization  ──►  data, autoencoder
                                 ▲
analysis ──► data (result_schema, dataset)   (never → runtime/core/optimization)
```

- **`optimization`** is the core execution module. It defines all operators
  used by the optimisation loop, including walkers, mutation generators,
  filters, oracles and the execution graph. It runs from plain Python with a
  manually constructed `PepCompassPipeline`, without `core`, YAML or the
  CLI — so each operator can be tested and used independently of the
  configuration layer.
- **`core`** builds and validates only. `PipelineBuilder` turns a parsed
  `PipelineSpecification` plus an already-constructed autoencoder into a
  `PepCompassPipeline`. It does not read YAML, does not know about the CLI,
  and does not persist anything.
- **`runtime`** is everything from a configuration file to persisted
  results: loading, validation, input/grid/plan materialisation, workflow
  selection, backend execution, and output writing. It is the layer that
  changes when you add a new way to *run* PepCompass (a backend, a
  workflow), not a new way to *compute*.
- **`autoencoder`** (not `models` — see below) and **`data`** are shared,
  dependency-light foundations used by both `optimization` and `runtime`.
- **`analysis`** only ever depends on `data` (the result schema and the
  logical dataset view) and its own `reader`. It cannot import `runtime` or
  `optimization` — it consumes what they produced, after the fact.

## Naming decisions

### `autoencoder`, not `models`

A generic top-level `models` package was rejected: it would recreate the kind
of undifferentiated bucket this restructuring was meant to remove. This does
not prohibit model-specific directories containing checkpoint files, such as
`autoencoder/strategies/hydramp/models/`. The top-level package is named for
what it concretely provides, the autoencoder contract, its geometry
operations and its registered implementations, rather than for arbitrary
future model types.

### `method` vs `model`

`autoencoder.method` selects the *implementation* (`hydramp`), while
`autoencoder.model` selects a *named checkpoint variant* of that
implementation (`article_25`). The implementation factory is registered with
`AutoencoderRegistry.register_method`; each checkpoint variant is registered
with `AutoencoderRegistry.register_model` as a model descriptor. At build time,
`AutoencoderFactory` resolves the descriptor, merges its parameters with
explicit overrides, and invokes the selected method factory. This two-axis
split means that a future fine-tuned checkpoint can be added as another model
descriptor without writing a new method strategy. See
`autoencoder/registry.py`/`factory.py` and [Developer Guide](developer-guide.md#adding-an-autoencoder).

### `core` builds; it does not run

An earlier proposal collected CLI, planning, backends and the engine under
one `core`. That was rejected because it made `core` mean "everything
executable," which is exactly the ambiguity this restructuring set out to
remove. `core` is deliberately narrow: specification → validated,
constructed `PepCompassPipeline`, nothing else. Everything about *running*
that pipeline (once, or as a planned experiment) belongs to `optimization`
(for the single run) or `runtime` (for the experiment).

### Latent geometry folded into `autoencoder`

The former `core/latent_geometry.py` (Jacobian SVD / tangent-space
decomposition) was moved into `autoencoder/geometry.py`. It is not a
pipeline-construction concern — it is an operation derived directly from the
decoder Jacobian, used by SORBES and MUTANG, and belongs next to the
autoencoder contract that produces that Jacobian.

## Resource control

Two independent, complementary mechanisms were designed and are now
implemented as `optimization/stability_estimation/`:

- **Static estimation** (`estimation.py`) — per-node candidate-count and
  byte-size bounds computed from declared cardinality behaviour, run during
  `dry-run` before any model is constructed. A node whose output cannot be
  bounded from configuration alone (a data-dependent filter's pass-through
  rate, for example) is reported as an explicit unknown rather than a
  guessed number, so a false sense of precision is never presented.
- **Runtime monitoring** (`monitoring.py`) — actual process RSS and CUDA
  allocator sampling at step boundaries (not inside per-operation tensor
  code, to avoid forcing GPU synchronisation mid-operation), written to
  `tracking/stability.csv`.

See [Technical Architecture](technical-architecture.md#stability-estimation)
for the implemented mechanism.

## PoGS

PoGS takes a different input shape than the composable pipeline and was
therefore deliberately **not** inserted into the existing `Loop`/`Flow`
graph. 

The agreed integration shape is a second, independent
`RuntimeWorkflow` implementation (`runtime/workflows/base.py`'s
`RuntimeWorkflow` protocol, the same contract `ComposableWorkflow`
implements), selected the same way at the runtime boundary rather than
branching inside the engine.

**Current status**: - not implemented #TODO PoGS idea 
