# Technical Architecture

This document describes how PepCompass constructs and executes an
optimisation pipeline.

> Remarks:
> - Public configuration belongs to the [User Guide](user-guide.md).
> - Modification procedures belong to the [Developer Guide](developer-guide.md).
> - Why the packages below are split this way belongs to
>   [Architecture Decisions](architecture-decisions.md).

## Table of Contents

- [Technical Architecture](#technical-architecture)
  - [Table of Contents](#table-of-contents)
  - [System Architecture](#system-architecture)
    - [Core](#core)
    - [Autoencoder](#autoencoder)
    - [Data](#data)
    - [Optimisation Engine](#optimisation-engine)
    - [Runtime](#runtime)
    - [Tracking and Analysis](#tracking-and-analysis)
  - [Runtime Construction](#runtime-construction)
  - [Pipeline Elements](#pipeline-elements)
    - [Candidate Data Model](#candidate-data-model)
    - [Step Contract](#step-contract)
    - [Optimisation Context and State](#optimisation-context-and-state)
  - [Execution Graph](#execution-graph)
    - [Flow](#flow)
    - [Loop](#loop)
    - [Parallel and Batch Merging](#parallel-and-batch-merging)
    - [Local Enumeration](#local-enumeration)
  - [Runtime Implementation Notes](#runtime-implementation-notes)
  - [Stability Estimation](#stability-estimation)
  - [Tracking](#tracking)

## System Architecture

PepCompass isolates *construction* (validating configuration and wiring
components into an executable graph) from the *components* themselves
(walkers, mutation generators, filters, oracles, interchangeable strategy
families behind abstract contracts) and from *running experiments*
(planning, backends, persistence). This lets the same components be combined
into different pipelines, and lets the pipeline itself be built and run from
plain Python without YAML, CLI or persistence.

### Core

`pep_compass.core` (`specification.py`, `builder.py`, `validation.py`,
`estimation.py`) is the composition root. `PipelineBuilder.build(...)`
turns a `PipelineSpecification` plus an already-constructed autoencoder into
one executable `PepCompassPipeline`. Core validates structure and component
parameters and can run the static stability estimator; it does not implement
walking, mutation, filtering or scoring algorithms itself, and it does not
know about YAML, the CLI, or where results are written.

### Autoencoder

`pep_compass.autoencoder` (`base.py`, `registry.py`, `factory.py`,
`geometry.py`, `strategies/hydramp/`) owns the autoencoder contract, its
strategy registry, and geometric operations derived from the decoder
Jacobian (`field_derivative`, tangent-space decomposition consumed by SORBES
and MUTANG). `AutoencoderFactory.build(method, model, device, **parameters)`
separates the *implementation* (`method`, e.g. `hydramp`) from the *named
checkpoint* (`model`, e.g. `article_25`) — see
[Architecture Decisions](architecture-decisions.md#autoencoder-not-models).

### Data

`pep_compass.data` holds shared data types used across the runtime:
`optimization.py` (`CandidateBatch` and batch fields, used during execution),
`result_schema.py` (the versioned on-disk result format) and `dataset.py` (the
logical, lazy view of loaded results consumed by `analysis`).

### Optimisation Engine

`pep_compass.optimization` defines the executable graph and the science:

- `engine/execution/` — `Step`, `OptimizationContext`, `OptimizationState`,
  `OptimizationResult`: generic, method-agnostic execution machinery.
- `engine/operations/` — `Flow`, `Loop`, `Parallel` (+ merge policies),
  `LocalEnumeration`: composite control-flow operations.
- `components/{walkers,mutation_generators,filters,oracles}/` — independent
  strategy families, each with a `manager.py` registry and a `strategies/`
  package. `filters/` further splits into `direct/` (self-contained
  transformations) and `ranked/` (`ScoreFunction` + `SelectionRule`
  composition).
- `stability_estimation/` — static memory bounds and a runtime memory
  monitor (see [Stability Estimation](#stability-estimation)).
- `pipeline.py` — `PepCompassPipeline`, the public executable object; it can
  be constructed and run manually (no `core`, YAML or CLI required), or
  obtained from `core.builder.PipelineBuilder`.

### Runtime

`pep_compass.runtime` turns a configuration file into persisted results: it
loads and validates configuration (`configuration/`), materialises input
tasks, grid variants and a flat execution plan (`planning/`), selects a
`RuntimeWorkflow` that builds a `PepCompassPipeline` (`workflows/`), executes
selected plan entries on the configured backend (`backends/`), and writes
results and tracking output (`output/`). See
[Runtime Construction](#runtime-construction) and the
[User Guide](user-guide.md#commands) for the CLI surface.

### Tracking and Analysis

Tracking observes step execution and writes runtime events; it is not stored
in `CandidateBatch` and cannot change optimisation fields, candidate count or
step order. `pep_compass.analysis` reads persisted results after execution
and does not participate in candidate transformations — see the
[Analysis Guide](analysis-guide.md).

## Runtime Construction

```text
configuration file (YAML/JSON)
       │
       ▼
load_runtime_configuration ──► validate_runtime_configuration
       │                              (experiment/execution/tracking/autoencoder shape)
       ▼
materialize_execution_plan ──► input tasks × grid variants → flat ExecutionPlan
       │
       ▼
RuntimeWorkflow.build_pipeline
       │
       ├── parse_pipeline_specification
       │
       └── PipelineBuilder.build
             ├── validate_pipeline_specification
             ├── validate_registered_components
             ├── construct the Step tree
             └── configure limits and stability estimation
       │
       ▼
PepCompassPipeline.run(sequences, seed=...) ──► OptimizationResult
       │
       ▼
ResultWriter ──► variants/<variant_id>/runs/<run_id>/{result.json, candidates.csv, ...}
```

`ComposableWorkflow` (`runtime/workflows/composable.py`) is the only
implemented `RuntimeWorkflow` today: it builds one autoencoder once, wraps a
`PipelineBuilder`, and parses+builds a fresh pipeline per plan entry. `dry-run`
runs configuration and component validation plus static estimation without
constructing an autoencoder or executing anything. `test-run` builds and
executes a real pipeline from a copied configuration with numeric limits
lowered (`runtime/test_run.py`) and result persistence disabled. It also
reduces local-enumeration trajectories to one and removes a time-based
`walk_time` bound when present, because the test policy caps fixed iteration
counts instead.

## Pipeline Elements

### Candidate Data Model

`Candidate` contains `sequence` and `latent_origin` — the concrete latent
point that generated the sequence, never replaced by the mean of re-encoding
that sequence.

`CandidateBatch` (`data/optimization.py`) stores sequences as one tuple,
latent origins as a tensor with batch dimension first, and algorithm data as
named `fields`. It is immutable; every transformation returns another batch.
Field types:

- `TensorField` — candidate-aligned tensor;
- `ObjectField` — candidate-aligned Python objects;
- `SharedField` — one value shared by the whole batch;
- `OptionalField` — adds a validity mask when merged branches do not all
  provide the same field.

`select(indices)` applies one index set to sequences, latent origins and
every field, so a filter cannot silently misalign a score or geometry
column. `repeat_from_parents(parent_indices)` duplicates every parent-aligned
column for one-to-many expansion; a mutation generator then replaces
sequences with `with_sequences` while every child retains its parent's exact
latent origin. `CandidateBatch.concatenate` combines branch outputs in branch
order; a field present in only a subset of branches becomes optional.

### Step Contract

Every node in the executable graph — leaf component or composite — inherits
`Step` (`engine/execution/step.py`) and implements only `_execute(batch,
context) -> CandidateBatch`. `precompute(context)` prepares state reusable
for the entire run; `prepare_iteration(batch, context)` refreshes state that
depends on the current batch. Subclasses must not override `__call__`: it
enters a scoped context, runs `prepare_iteration`, samples memory before and
after, drives the tracker's `begin_step`/`end_step`/`fail_step`, times the
call, and only then delegates to `_execute`. Tracking is therefore automatic
— a strategy implementation never calls the tracker itself.

### Optimisation Context and State

`OptimizationContext` (`engine/execution/context.py`) carries the
autoencoder, tracker, execution `scope`, seed, NumPy `rng`, `state` and
stability monitor through the tree. `enter_step`/`enter_iteration` extend the
scope without changing the seed or RNG. `enter_branch(name, index)` derives
an **independently seeded** context (`seed + index + 1`, fresh
`np.random.default_rng`), so parallel branches are reproducible and
independent of completion order; every other context copy shares one
`OptimizationState`.

`OptimizationState` (`engine/execution/state.py`) holds, run-scoped:

- **oracle observations** — indexed by objective and sequence, so later
  selection reads observation history instead of requiring every batch to
  retain a score column;
- **trust-region state** — per objective, centre sequence/position, best
  score, radius, success/failure counters;
- **resource counters** — evaluated sequences and generated candidates,
  attached to tracking records before and after each step;
- **stop requests** — set when a limit is reached; `Loop` tests it (and
  whether the batch is empty) before another iteration.

## Execution Graph

### Flow

`Flow(steps)` passes the output of each child to the next; it does not
interpret fields or insert missing operations. A configuration `steps:` list
becomes one `Flow`.

### Loop

`Loop(body, iterations)` runs a `Flow` body a fixed number of times per
nested `iteration[N]` scope, stopping early on a global stop request or an
empty batch.

### Parallel and Batch Merging

`Parallel(branches, execution, merger)` gives every branch the same
immutable input batch. `sequential` (default) runs branches in order;
`concurrent` uses one worker thread per branch — branch completion order
never affects merged candidate order, since outputs are read back in
configuration order and merged by branch index. `ConcatenateMerger` is the
only implemented policy; `interleave`, `select_best` and `weighted_sample`
are declared (`MergeMethod` literal) but raise `NotImplementedError` (see
[Architecture Decisions](architecture-decisions.md#known-issues)).
Both execution modes share one `OptimizationState`; concurrent branches can
therefore update counters, observations and trust regions concurrently and
are not transactionally isolated.

### Local Enumeration

`LocalEnumeration` (`engine/operations/local_enumeration.py`) composes one
`walker` and one `mutation_generator` into a bounded trajectory-exploration
operation used by the reference LE-BO pipeline, keeping the continuing SORBES
trajectory point separate from the MUTANG candidate pool it emits — MUTANG
candidates do not automatically become the next SORBES step's input.

For every input seed, `trajectories` independent branches walk from that
seed. At every step: the walker advances one position; if
`include_walk_points` is true, the decoded walk point itself is emitted
(minus transient walker-only fields); the mutation generator expands that
point and the configured `filters` (a small `Flow` of `filter` steps,
typically distance/plausibility constraints scoped to
`local_enumeration.center_sequence`) run immediately after, and that
filtered batch is also emitted. Only the walk point continues to the next
iteration — mutations are collected, not walked from. Global deduplication,
selection, oracle evaluation and Bayesian optimisation remain ordinary steps
placed *after* `LocalEnumeration`, not inside it.

Termination is exactly one of `iterations` (fixed step count) or `walk_time`
(accumulated adjusted SORBES time per trajectory). `trajectory_execution:
batched` (default) advances active trajectories as one batched tensor call
inside an isolated forked Torch RNG stream (`torch.random.fork_rng`), removing
finished trajectories from the batch under a `walk_time` bound;
`trajectory_execution: sequential` runs one
trajectory at a time as a correctness reference and per-trajectory tracking
identity.

`local_enumerations.csv` (see [Tracking](#tracking)) records each execution's
input checkpoint path, input count, output count and a SHA-256 digest of the
output sequences. The checkpoint shard stores the exact input sequences,
latents, lineage identifiers and RNG states; the output list is not duplicated
in the CSV table. `analysis.reader.RunReplay.local_enumeration_input` and
`.verify_local_enumeration` (see [Analysis Guide](analysis-guide.md))
reconstruct and verify one execution's output against that digest.

## Runtime Implementation Notes

- **Encoder-decoder reuse**: one autoencoder instance is built once per
  `ComposableWorkflow` and reused across every plan entry it builds a
  pipeline for; each run still gets an independent `OptimizationContext`,
  tracker and candidate data.
- **Oracle batching**: a `BlackBoxOracle` splits evaluated sequences by
  `evaluation_batch_size`, never calls the underlying model on an empty
  batch, and always attaches a correctly-shaped (possibly empty) score field
  so batch schema stays valid.
- **Candidate-growth control**: MUTANG reports generated-candidate counts to
  `OptimizationState`; mutation-choice strategies additionally bound their
  local Cartesian product through `maximum_candidates`. Neither mechanism
  changes the requirement that every step returns one aligned batch.
- **Failure semantics**: the tracker records a failed step and the exception
  propagates; `runtime.runner` persists a failed `result.json` status, and
  `execution.continue_on_error` decides whether later, independent plan
  entries still execute.
- **Backends**: `local`, `subprocess` (isolated local worker processes) and
  `slurm` (writes an array script, does not submit) — see the
  [User Guide](user-guide.md#run).

## Stability Estimation

Two independent, complementary mechanisms live in
`optimization/stability_estimation/`:

- **Static estimation** (`estimation.py`, driven by `core.estimation` during
  `dry-run`): computes conservative upper candidate-count and latent-memory
  bounds per graph node from declared cardinality behaviour, without executing
  anything. A node whose output depends on data (e.g. a data-dependent
  filter pass-through rate) is reported as an explicit unknown rather than a
  guessed number.
- **Runtime monitoring** (`monitoring.py`, `StabilityMonitor` /
  `NullStabilityMonitor`, enabled by `tracking.monitor_stability`): samples
  process RSS and CUDA allocator statistics at step boundaries
  (`step.before:<path>`/`step.after:<path>`, plus `pipeline.input`/
  `pipeline.output`), not inside per-operation tensor code, so sampling does
  not force GPU synchronisation mid-operation. Snapshots are written to
  `tracking/stability.csv`.

Field-lifetime pruning (dropping batch fields no later step needs) and
explicit candidate-viability selectors are discussed as future extensions to
this mechanism but are **not implemented** — see
[Architecture Decisions](architecture-decisions.md#deferred-work).

## Tracking

`CSVStepTracker` (`runtime/output/tracking.py`) writes, per run, into
`tracking/`:

- `steps.csv` — every tracked step's timing, candidate-count and
  oracle-call deltas, tree depth, path, and nested loop/branch indices.
- `candidates.csv` — per-step candidate rows, written according to
  `tracking.candidate_snapshots`: `none`, `oracle`, or `all`.
- `trajectory_points.csv` + `checkpoints/trajectory/*.pt` — SORBES walker
  metadata and latent checkpoint shards (written for `SorbesWalker` steps at
  `level` `normal`/`all`).
- `local_enumerations.csv` + `checkpoints/local_enumeration/*.pt` —
  `LocalEnumeration` boundary metadata and input/RNG checkpoint shards (see
  [Local Enumeration](#local-enumeration)).
- `run.log` — run-scoped package log written by `RuntimeRunner`.
- `replay_manifest.json` — schema version, run/variant id, resolved seed,
  SHA-256 of the resolved configuration, Python/Torch/CUDA versions,
  determinism flag, and git revision/dirty state — enough to judge whether a
  later reconstruction attempt ran under matching conditions.

`ExecutionScope` (`optimization/tracking/core.py`) carries the hierarchical path,
nested loop indices, and parallel branch names/indices; entering a step,
iteration, or branch returns a new immutable scope, so tracking identifies
the complete branch hierarchy without adding fields to candidate batches.
`tracking.level` (`short`/`normal`/`all`) and `max_depth` bound which step
summaries and replay checkpoints are written without changing which steps
execute. `candidate_snapshots` independently controls candidate-row retention;
`store_latents`/`store_fields` control the optional columns in retained rows,
and `field_names` can restrict serialized fields. Serialising SORBES
tangent-space matrices for every retained candidate can dominate output size —
enable `store_fields` only when those values are needed for analysis.

See [User Guide](user-guide.md#output-files) for the complete per-run
directory layout, and the [Analysis Guide](analysis-guide.md) for reading
this layout back.
