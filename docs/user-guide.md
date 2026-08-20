# User Guide

This guide documents the `pep-compass` command-line interface and the public
YAML configuration for PepCompass experiments. 

## Commands

`pep-compass` is installed as a console script (`pyproject.toml`
`[project.scripts]`) and exposes three commands, all defined in
[`runtime/cli.py`](../src/pep_compass/runtime/cli.py). 

Every command takes one
configuration path plus a common set of flags. Run them from the repository
root so relative input and output paths resolve as expected.

```bash
uv run --extra cu118 pep-compass <command> <configuration.yaml> [flags]
```

Use `--extra cpu` instead of `--extra cu118` on a machine without CUDA.

### Common flags

Available on all three commands:

```text
configuration              required path to a YAML or JSON configuration file
--working-directory PATH   base for relative input/output paths (default: cwd)
--run-index INDEX          restrict execution to one plan index; may be repeated
--device DEVICE            override autoencoder.device (for example cpu or cuda)
```

### commands

#### `dry-run`

Validates the configuration and every registered component it references,
then prints a static candidate-count and memory estimate — **without loading any model**. 

> Use it before running anything expensive.

```bash
uv run --extra cu118 pep-compass dry-run \
  assets/experiments/configs/composable_example.yaml
```

For every materialised grid variant, `dry-run` prints per-node input/output/
peak candidate bounds (or `UNKNOWN` where the bound cannot be determined
statically), plus any structural or parameter diagnostics, then lists the
plan entries (`run_id`, `variant_id`, `task_id`, resolved `seed`) it would
execute.

#### `test-run`

Executes a bounded pipeline with **real models** and **no result persistence** — 
the fast way to confirm a configuration actually runs before
committing GPU time to a full experiment.

```bash
uv run --extra cu118 pep-compass test-run \
  assets/experiments/configs/composable_example.yaml \
  --device cpu
```

```text
--tasks N              number of plan entries to execute (default: 2)
--iterations N          cap every loop.iterations at this value (default: 1)
--max-candidates N      cap mutation_generator.parameters.maximum_candidates (default: 2)
--max-oracle-calls N    cap pipeline.limits.oracle_calls (default: 2)
```

`test-run` lowers numeric limits in place (`runtime/test_run.py`); it does
**not** change pipeline structure — every configured step still executes,
just with a small candidate pool. `local_enumeration.walk_time` is dropped
entirely for the duration of the test run (a time-based bound cannot be
capped the same way an iteration count can); `generated_candidates` is left
untouched, because it is a stop condition, not a per-parent cap, and lowering
it can short-circuit a local enumeration after its first mutation call. Output
is disabled (`experiment.output_directory` is forced to `None`) and execution
is forced to `backend: local, max_workers: 1`. The command prints, per
executed run: step timings and candidate-count deltas, memory snapshots
(process RSS and CUDA allocator stats where applicable), and the final
candidate list with scores (or a note that no oracle score field was
produced).

#### `run`

Executes the full plan through the configured (or overridden) backend and
persists results.

```bash
uv run --extra cu118 pep-compass run \
  assets/experiments/configs/composable_example.yaml
```

```text
--backend local|subprocess|slurm   override execution.backend
--max-workers INTEGER              override execution.max_workers
--resume                           skip plan entries with a completed result.json
--continue-on-error                keep executing remaining entries after a failure
--slurm-script PATH                output script path, required for --backend slurm
```

`local` executes selected runs in the current process and reuses one
constructed autoencoder across all of them. `subprocess` runs each selected
plan entry as an isolated **local** worker process (`python -m
pep_compass.runtime.cli run ... --worker`, one per entry, up to
`max_workers` concurrently) — it does not distribute work to another machine.
`slurm` **writes** a Slurm array script to `--slurm-script` and exits; it does
not call `sbatch`. Submit the generated script yourself:

```bash
uv run --extra cu118 pep-compass run \
  assets/experiments/configs/reference_lebo.yaml \
  --backend slurm \
  --slurm-script experiments/jobs/reference_lebo.sh

sbatch experiments/jobs/reference_lebo.sh
```

The script contains one `#SBATCH --array=` entry per selected plan index,
`#SBATCH --chdir=<working-directory>`, and any `job_name`/`partition`/`time`/
`gres`/`cpus_per_task`/`mem` settings from `execution.slurm`.

#### PoGS

<!-- #TODO: PoGS integration -->
PoGS is a declared but **unimplemented** workflow boundary. It will take a
different input shape than the composable pipeline above, so it is not
inserted into the existing loop — it is planned as a second, independent
workflow selected the same way `ComposableWorkflow` is selected today (see
`runtime.workflows.base.RuntimeWorkflow`). Today,
[`runtime/workflows/pogs.py`](../src/pep_compass/runtime/workflows/pogs.py)
raises `NotImplementedError` unconditionally and no `pep-compass` command
selects it. Once implemented, document its command here (a `pogs-run`
subcommand, or a `workflow:` selector shared with `run`/`test-run`) and its
own configuration section. See
[Architecture Decisions](architecture-decisions.md#pogs) for the current
integration plan.

## Configuration

A configuration file has five top-level, sibling sections — `tracking` and
`execution` are **not** nested under `experiment`
(`runtime/configuration/loading.py`):

```yaml
experiment:
  name: example
  seed: 1234
  seed_scope: run # or task
  input:
    sequences: [FLYKWWIRIGRLKL]
  output_directory: experiments/results/example
  grid: {}

autoencoder:
  method: hydramp
  model: article_25
  device: cpu # or cuda
  parameters:
    jacobian_mode: approx
    jacobian_eps: 0.001
    field_eps: 0.001

tracking:
  level: normal
  max_depth: null
  store_latents: false
  store_fields: false
  field_names: null
  candidate_snapshots: none
  monitor_stability: true

execution:
  backend: local
  max_workers: 1
  resume: false
  continue_on_error: false
  slurm: {}

pipeline:
  limits:
    oracle_calls: 10
    generated_candidates: 100000
  steps: [...] # see Parameter Reference
```

### `experiment`

`name` identifies the experiment. `seed` is the base seed; `seed_scope`
controls how it is derived per plan entry (see
[Reproducibility](#reproducibility)). `input` configures exactly one input
form:

```yaml
input:
  sequences: [FLYKWWIRIGRLKL, ACDEFGHIK]
  repetitions: 2
```

```yaml
input:
  csv:
    path: assets/peptides_data/peptides.csv
    sequence_column: sequence
    repetitions_column: repetitions
```

`csv.path` may be absolute or relative to `--working-directory`. The sequence
column is required; a row missing the repetitions column defaults to one run.
`output_directory` is optional — omitting it disables persisted results
(matches what `test-run` forces). `grid` maps dotted paths rooted at
`pipeline` to lists of values; the runner materialises their Cartesian
product (see [Grid](#grid)).

#### Reproducibility

`experiment.seed` plus a stable offset determines every run's resolved seed
(`runtime/planning/plan.py`). With `seed_scope: run` (default), every plan
entry — every combination of input task and grid variant — gets a distinct
seed. With `seed_scope: task`, the same input task keeps the same seed across
grid variants, which isolates the effect of a grid parameter from the effect
of randomness. The resolved seed is recorded in `resolved_config.json` for
every run. `repetitions` produces independent runs; it does not duplicate
candidates inside one optimisation batch.

#### Grid

```yaml
experiment:
  grid:
    pipeline.steps.0.loop.iterations: [1, 2]
```

The dotted path must already exist as a leaf in the base `pipeline` mapping —
a grid path is a substitution, not an insertion (`runtime/planning/
variants.py`). Keep the base value equal to the first grid value so the
configuration stays readable if the grid is later removed.

### `autoencoder`

`method` selects the implementation; `model` selects a named model/checkpoint
variant of that implementation. The only built-in `method` today is
`hydramp`, with model variant `article_25`. Separating `method` from `model`
lets a future fine-tuned checkpoint be added as a new `model` name without a
new strategy registration. `jacobian_eps` and `field_eps` are required;
`jacobian_mode` accepts `strict` or `approx`.

### `tracking`

`level` accepts `short` (oracle step summaries only),
`normal` (adds step summaries, sizes, timings and counters — the default), or
`all` (adds summaries for every enabled step). `max_depth` limits collection
by execution-tree depth without changing which steps execute.
`candidate_snapshots` controls candidate-row retention independently: `none`
(default), `oracle` (oracle steps only), or `all` (every enabled step).
`store_latents` serialises candidate latent origins into retained rows.
`store_fields` serialises algorithm fields; `field_names`, when set, limits
those fields to the listed names. `monitor_stability` enables the runtime
memory monitor (see
[Technical Architecture](technical-architecture.md#stability-estimation)).

### `execution`

`backend` accepts `local`, `subprocess` or `slurm` (see [Commands](#run)).
`max_workers` must be a positive integer. `slurm` accepts `job_name`,
`partition`, `time`, `gres`, `cpus_per_task` and `mem`.

### Output files

Each completed run directory contains:

```text
variants/<variant_id>/runs/<run_id>/
  result.json            status, input identity, final candidate count, objective summary
  resolved_config.json   exact resolved configuration used for this run
  candidates.csv         final sequences and every oracle.<name>.score column
  fields.jsonl           all non-latent candidate fields, one JSON object per row
  latent_origins.pt       final candidate latent origins
  tracking/
    steps.csv
    candidates.csv
    trajectory_points.csv
    local_enumerations.csv
    replay_manifest.json
    stability.csv
    run.log
    checkpoints/
      trajectory/
      local_enumeration/
```

The writer always uses `variants/<variant_id>/runs/<run_id>/`, including for a
configuration with no grid. `result.json` reports `null` objective, best
sequence and best score for an experiment without an oracle, and a completed
result with zero candidates is valid when filters remove the entire pool before
evaluation.
See [Technical Architecture](technical-architecture.md#tracking) for what
each tracking file contains, and the
[Analysis Guide](analysis-guide.md) for reading this layout back
programmatically.

## Parameter Reference

### Pipeline limits

```yaml
pipeline:
  limits:
    oracle_calls: 1400 # or null to disable
    generated_candidates: 1000000 # or null to disable
```

`oracle_calls` bounds evaluated sequences; an oracle truncates its input to
the remaining budget. `generated_candidates` counts candidates reported by
mutation generators and requests termination once the threshold is reached.

### Pipeline steps

Every entry under `pipeline.steps` (and every nested `steps` list) is a
single-key mapping. Seven operation keys exist
(`runtime/configuration/pipeline.py`):

- **`walker`**, **`mutation_generator`**, **`filter`**, **`oracle`** — leaf
  components: `{method: <name>, parameters: {...}}`.
- **`loop`** — `{iterations: <int>, steps: [...]}`. Repeats its body a fixed
  number of times; stops early if the global stop state is requested or the
  candidate batch becomes empty.
- **`parallel`** — either replicated or explicit branches:

  ```yaml
  - parallel:
      replicas: 10
      execution: sequential # or concurrent
      merge: concatenate
      steps: [...] # shared by every replica, named replica_000..replica_009
  ```

  ```yaml
  - parallel:
      execution: sequential
      merge: concatenate
      branches:
        - name: conservative
          steps: [...]
        - name: exploratory
          steps: [...]
  ```

  `replicas + steps` and `branches` are mutually exclusive. `execution`
  accepts `sequential` (default) or `concurrent` (one worker thread per
  branch). `merge` accepts only `concatenate` today; `interleave`,
  `select_best` and `weighted_sample` are declared but raise
  `NotImplementedError` (see
  [Architecture Decisions](architecture-decisions.md#known-issues)).

- **`local_enumeration`** — composes one walker and one mutation generator
  into a bounded trajectory-exploration operation, used by the reference
  LE-BO pipeline:

  ```yaml
  - local_enumeration:
      walker:
        method: sorbes
        parameters: {...}
      mutation_generator:
        method: mutang
        parameters: {...}
      filters: # local filters applied after every mutation batch
        - method: levenshtein
          parameters: {maximum_distance: 4, reference_field: local_enumeration.center_sequence}
      trajectories: 10 # branches per input seed
      trajectory_execution: batched # or sequential (reference implementation)
      iterations: 10 # exactly one of iterations / walk_time is required
      # walk_time: 0.1
      include_walk_points: true # include decoded SORBES points in the output, not only mutations
  ```

  Global deduplication, selection, oracle evaluation and Bayesian
  optimisation remain ordinary steps placed after `local_enumeration`, not
  inside it. See
  [Technical Architecture](technical-architecture.md#local-enumeration) for
  execution semantics.

Every list entry must contain exactly one operation key; a nested `steps` or
`iterations` indented at the wrong level produces "must contain exactly one
operation key" (see [Troubleshooting](#troubleshooting)).

## Available Components

Registry names and parameters below are public API — changing one requires
updating the responsible strategy's `parameter_contract` and this table (see
[Developer Guide](developer-guide.md#documentation)).

### Autoencoders

- `hydramp` — peptide autoencoder with strict or approximate decoder
  Jacobians; model variants selected through `autoencoder.model` (currently
  `article_25`).

### Walkers

- `sorbes` — one SORBES step: latent geometry, direction sampling, an
  adaptive position update and a boundary check, in that fixed order. Accepts
  `geometry`, `directions`, `scaling`, `position_update` and `boundary` as
  `{method, parameters}` sub-strategies. The built-in methods are
  `kappa_stable`, `active_inactive`, `stable_dimension`, `main`, and, for
  `position_update`, also `article` and `without_acceleration`. A legacy flat
  form (`horizontal_threshold`, `time_step`, `max_horizontal_update_norm`,
  `vertical_movement`) is converted to the corresponding nested defaults.

### Mutation Generators

- `mutang` — reads SORBES singular values and left vectors, selects
  residue substitutions from significant tangent directions and materialises
  their Cartesian product per parent. Accepts `maximum_candidates`,
  `strategies.geometry.method` (`shared` or `kappa_stable`), `max_len`,
  `direction_significance_threshold`, `min_number_of_directions`,
  `token_threshold`, `alphabet`. For HydrAMP, `max_len` must match the model
  sequence length (25); it is not a candidate limit.

MUTANG requires the `walker.singular_values`/`walker.left_vectors` fields
produced by SORBES, so a `mutation_generator` step must follow a `walker`
step (or run inside `local_enumeration`, which wires this automatically).

### Filters

Filters split into two extension styles
(`optimization/components/filters/{direct,ranked}/`):

- **Direct filters** are self-contained batch transformations:

  | method | parameters | description |
  |---|---|---|
  | `deduplicate` | `key`: `sequence` \| `sequence_and_latent` | keeps the first candidate per identity key |
  | `candidate_subset` | `count`, `mode`: `random`\|`highest`\|`lowest`, `score_field` (required unless `random`) | caps batch size by random or score-ranked subset |
  | `levenshtein` | `reference` or `reference_field` (exactly one), `maximum_distance` | keeps sequences within an edit-distance radius of a reference |
  | `sequence_length` | `minimum`, `maximum` | keeps sequences within an inclusive length interval |
  | `robot` | `objective`, `batch_size`, `maximize`, `diversity_threshold`, `acquisition_batch_size`, `standardize`, `device` | Bayesian ROBOT selection: GP + LogEI over molecular fingerprints, with a diversity threshold |
  | `trust_region` | `objective`, `geometry`: `sequence`\|`latent`, `initial_radius` | retains candidates inside the current per-objective trust-region radius |
  | `trust_region_update` | `objective`, `initial_radius`, `minimum_radius`, `maximum_radius`, `expand_factor`, `shrink_factor`, `success_tolerance`, `failure_tolerance`, `maximize` | expands or shrinks the trust-region radius from oracle scores already on the batch |
  | `random_walker` / `random_mutang` | `selection_fraction`, `temperature`, `maximum_positions`, `residues_per_position`, `maximum_candidates`, `alphabet` | random-baseline ablation controls (walker-position vs. MUTANG-pool randomisation) |

- **Ranked filters** compose one `ScoreFunction` with one `SelectionRule`
  (`filters/ranked/base.py`). Use `ranked` directly for a custom
  combination, or one of the pre-built method names that already pair a
  specific score with a specific selection rule:

  | method | scoring | selection | parameters |
  |---|---|---|---|
  | `ranked` | any of `tandem`, `lams`, `move`, `decoder_likelihood`, `esm` | any of `threshold`, `nucleus`, `top_k` | `scoring: {method, parameters}`, `selection: {method, parameters}` |
  | `lpbebo` | decoder log-probability | nucleus top-p | `top_p`, `temperature`, `maximum_candidates`, `alphabet` |
  | `lams` | worst compatible mutation-pair cosine similarity | threshold | `similarity_threshold`, `horizontal_threshold`, `maximum_candidates`, `alphabet` |
  | `tandem` | projected pairwise mutation-direction similarity | nucleus top-p | `top_p`, `temperature`, `horizontal_threshold`, `maximum_candidates`, `alphabet` |
  | `move` | negative predicted net latent displacement | nucleus top-p | `top_p`, `temperature`, `maximum_candidates`, `alphabet` |
  | `esm_plausibility` | ESM2 pseudo-log-likelihood | threshold XOR top-k | `threshold`, `top_k`, `model_name`, `device` |

  Mutation-choice filters (`lpbebo`, `lams`, `tandem`, `move`,
  `random_walker`, `random_mutang`) consume MUTANG's mutation-choice fields,
  so they must follow a `mutation_generator` step.

### Oracles

Built-in methods are `apex`, `apex_original` (preserved pre-refactor APEX for
comparison), `battleamp`, `eipred`, `hydrophobicity`, `mbc_attention` and
`toxipep`. Every oracle is a `BlackBoxOracle` adapter and accepts optional
POLI controls (`batch_size`, `parallelize`, `num_workers`,
`evaluation_budget`, `force_isolation`) plus `evaluation_batch_size`. APEX
accepts `model` as a named weight variant and defaults to `default` when it is
omitted. BattleAMP and MBC-Attention require TensorFlow in the runtime
environment. Model-specific parameters are validated before model
construction; a lazily-imported oracle module does not prevent library import
when its dependency is missing.

## Execution Backends

See [Commands](#run) for `local`/`subprocess`/`slurm` semantics and flags.

## Complete Examples

- `assets/experiments/configs/composable_example.yaml`: fully-commented
  reference covering every configuration section and step type — start here.
- `assets/experiments/configs/reference_lebo.yaml`: replicated SORBES–MUTANG local
  enumeration followed by ROBOT, APEX and trust-region updates.

## Troubleshooting

`must contain exactly one operation key` means one item under `steps`
contains multiple sibling keys, usually because `iterations` or `steps` is
indented at the wrong level.

`Grid path does not exist` means the dotted grid path does not identify a
leaf already present in the base `pipeline` mapping.

`MUTANG requires walker.singular_values and walker.left_vectors` means a
`mutation_generator` step executed without an earlier `walker` step in the
same data flow (or outside `local_enumeration`, which wires this
automatically).

A completed result with zero candidates and `null` objective fields is valid
when filters remove the entire pool before oracle evaluation.

`dry-run` reports `output=UNKNOWN` or `peak=UNKNOWN` for a node whose output
cardinality cannot be bounded statically (for example, a filter whose
pass-through rate is data-dependent) — this is a reported limit of the
estimator, not a configuration error.
