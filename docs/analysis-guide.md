# Analysis Guide

`pep_compass.analysis` reads results persisted by `runtime` (see
[User Guide](user-guide.md#output-files)) and computes derived tables and
plots. It is a **separate, read-only consumer**: it never imports
`runtime`, `core`, or the execution engine, and it cannot change what a run
produced; it only interprets it after the fact.

> If you are looking for how a pipeline runs, see
> [Technical Architecture](technical-architecture.md).

> Remark:
> To reduce memory use, the reader stores only seeds and reconstructs analyses
> from those seeds.

## `analysis.reader`

`analysis.reader` discovers, selects and lazily reads tracking output
without loading whole tables into memory unless asked to. The available result
schema is illustrated in
[`assets/experiments/schema_example_notebook_for_analysis.ipynb`](../assets/experiments/schema_example_notebook_for_analysis.ipynb).

### `ExperimentReader`

```python
from pep_compass.analysis.reader import ExperimentReader

reader = ExperimentReader("experiments/results/reference_lebo/run")
reader.experiments   # discovered experiments, tables not loaded yet
reader.runs           # every discovered ExperimentRun (run/grid/tracking-path identity + metadata)
reader.dataset         # ExperimentDataset: schema_version, runs, lazy per-table handles
reader.cached_analyses()
```

`ExperimentReader(path)` accepts a collection root, an experiment directory,
or a single tracking-run directory. The supported result layout is the
versioned `variants/*/runs/*/result.json` layout written by the current
runtime. Discovery does not load tracking tables. The reader also opens a
`.pep_compass_analysis.sqlite` cache next to the opened root (see
[`MetricsStore`](#metricsstore)).

### `ExperimentSelection`

```python
selection = reader.select(seeds=[1234], peptides=["FLYKWWIRIGRLKL"])
selection = selection.rows(candidate_index=0)   # deferred row filter

selection.count_rows("candidates")
selection.paths("candidates")
selection.collect("candidates")                 # materialize
for chunk in selection.scan("trajectory_points", chunk_size=1000):
    ...
selection.specification()                        # stable cache key: run ids, paths, filters, file fingerprints
```

`select(...)` filters *runs* by manifest/result metadata (`experiments`,
`methods`, `grid_ids`, `peptides`, `seeds`, `parameters`); it is immutable
and chainable. `.rows(**filters)` adds a deferred *row*-level equality/
membership filter, applied only to tables whose source columns contain the
filter key, and only when a table is actually scanned. Logical table names
are registered in `analysis.reader.data_schemas.registry.SCHEMAS` (`steps`,
`candidates`, `trajectory_points`, `local_enumerations`, `stability`, plus
BO-loop tables not produced by the composable-pipeline reference runs).

### `RunReplay`

```python
replay = reader.replay("run_00000")
replay.result                 # terminal run metadata
replay.resolved_configuration # exact configuration this run used
replay.final_candidates       # sequences + every oracle.*.score column
replay.final_latents           # (N, D) tensor
replay.trajectory_points, replay.trajectory_latents
points, latents = replay.trajectory(trajectory_id)

sequences, latents = replay.local_enumeration_input(execution_id)
verification = replay.verify_local_enumeration(execution_id, reconstructed_sequences)
verification.matches
```

`reader.replay(run_id)` resolves one unambiguous run and exposes its final
results, exact resolved configuration, and tracking checkpoints, including
the ones needed to reconstruct one `local_enumeration` execution's exact
input (see
[Technical Architecture](technical-architecture.md#local-enumeration)).
`local_enumeration_input`/`verify_local_enumeration` only return the stored
input and compare a *reconstruction* you supply against the stored count and
SHA-256 digest — they do not recompute the pipeline themselves. Producing a
real reconstruction means rebuilding the same `walker`/`mutation_generator`/
`filters` `Step`s from `resolved_configuration["pipeline"]` through
`core.builder.PipelineBuilder` and running them against that input; see
`tests/analysis/reader/test_reader.py::test_reader_validates_local_enumeration_replay_checkpoint`
for the pattern (there built from mock components).

### `MetricsStore`

```python
reader.metrics.get_or_compute_metrics("apex", sequences, evaluator, metric_version="1")
reader.metrics.put_analysis("my_analysis", frame, metadata={}, selection=selection.specification(), parameters={})
reader.metrics.get_analysis("my_analysis", selection=selection.specification(), parameters={})
reader.cached_analyses()   # == reader.metrics.list_analyses()
```

`reader.metrics` is a `MetricsStore` backed by
`<root>/.pep_compass_analysis.sqlite`, with two independent caches: per-
sequence metric values (`put_metrics`/`get_metrics`/`get_or_compute_metrics`,
useful for caching an expensive oracle call across notebooks) and exact
aggregate analysis results keyed by a stable hash of the selection and
parameters (`put_analysis`/`get_analysis`/`list_analyses`/`load_analysis`).

## `analysis_types`, `resampling`, `visualization`

- **`analysis_types/locality`** exposes `LocalityAnalysis` with the maintained
  `latent_jump` and `sorbes_trajectory_profile` analyses. Both consume an
  `ExperimentSelection`, stream the required tracking tables and return an
  `AnalysisResult` (`data`, `metadata`, `diagnostics`).
- **`resampling`** (`bootstrap.py`: `ClusterSampler`, `RowSampler`,
  `BootstrapEngine`) provides resampling utilities for uncertainty estimates
  over analysis outputs.
- **`visualization`** (`locality.py`: `LocalityVisualizer`, `theme.py`:
  `PlotTheme`) renders `AnalysisResult` tables produced by `analysis_types`.
- **`experiment.py`** (`ExperimentAnalysis`) is a thin composition facade that
  exposes the locality analysis and its `LocalityVisualizer` through one
  selection.