# Run data contract

A run owns `results.sqlite` and immutable `settings.json`. Memory storage uses the
same SQLite schema and exports it through `RunResult.save()`. One database contains
one run. Settings describe the constructor graph, initial inputs, seed, retention,
model artifacts and code hashes. SQLite stores execution facts and retained data.
There are no compatibility readers or migrations for previous declaration formats.

## Evaluated results and cache

`evaluations` is the sole persisted source of actual scoring records. No storage
option discards an evaluation. `peptides` interns retained sequence strings, and
`objectives` owns direction and configured identity. `evaluation_records` joins
these tables for analysis; `evaluation_archive` selects the latest actual score
for each objective/sequence pair without deleting earlier records.

Built-in oracle components score each distinct sequence once per configured
objective per run. Duplicate rows reuse that scalar and preserve batch order and
score dtype. Cache entries are fresh for each run. Conflicting configurations for
the same objective output field raise before scoring. Separate objective names
have separate caches. Native predictor microbatching does not change cache keys.
The soft oracle budget counts newly scored unique peptides, not cache hits or
predictor calls. A completed batch is retained even when it crosses the budget.

The engine returns its graph output directly. Finalization does not construct a
second evaluated archive in Python. The scalar working observations used by
optimization and cache are run state; SQL is the persistent result source.

## Identity, order and lineage

- `node_id`: stable placement identity in the declaration and executable graph.
- `execution_id`: one invocation of that node, with an optional parent invocation.
- `iteration_id`: one logical iteration, with an optional parent iteration.
- `candidate_id`: one generated candidate identity, preserved by selection.
- `peptide_id`: one retained sequence; different candidates can share it.
- `batch_id`: one observed immutable ordered batch state.

`batch_members(batch_id, position)` defines order, including duplicate candidate
occurrences. `save_lineage=True` is the default. Unscored ancestors can have a
`candidate_instances` row whose `peptide_id` is NULL: their identities and parent
relations survive without saving their strings or latent tensors.

A new candidate with an already evaluated sequence links to that existing
`peptide_id` when its cached score is attached. This records its occurrence and
provenance without invoking the predictor or adding an evaluation row.

`batch_lineage` stores half-open ranges. `pairwise` requires equal parent and child
lengths and maps corresponding positions. `broadcast` requires a singleton parent
range and maps it to every child position. Separate disjoint ranges use separate
records. Relations are emitted during transformations, before oracle selection.
Foreign keys are enabled, and all references belong to this run database.

## Retention

| Setting | Data |
| --- | --- |
| `reconstruction="none"` | Mandatory evaluations and requested structural lineage; no full reconstruction states |
| `reconstruction="boundaries"` | Additionally supported LocalEnumeration input, optimization/step state and RNG checkpoints |
| `reconstruction="trajectories"` | Additionally ordered step inputs/outputs, walker points and actual root graph output; includes boundaries |
| `component_data="summary"` | Compact component display metadata |
| `component_data="full"` | Additionally field payload references; peptide-string provenance columns are excluded |

`Snapshot(name)` adds a named reference to the same retained batch and requires
`boundaries` or `trajectories`. It cannot silently override `none`.
`get_final_batch()` requires `trajectories`; it follows the root execution's
`output_batch_id`, including filters after a walker. It never substitutes an
archive of scored peptides for the graph output.

`batch_payloads` binds field names to content-addressed `tensor_payloads`. Sequence
strings and member ordering are read from relational tables, not copied into each
ordinary batch payload. Identical field contents share their payload hash. Named
candidate sets reference `batch_id`; they do not carry another batch blob.
Checkpoints reference the retained input batch and store optimizer/RNG state.
They retain an evaluation-count boundary instead of copying scored results; replay
reconstructs working observations from that historical prefix of `evaluations`.

## Measurements

`metrics_level="none"` keeps identity, status and references but leaves per-step
and per-iteration measurements NULL. `basic` adds counts, wall duration and process
RSS. `basic_and_components` additionally records internal stages in
`execution_metrics`, aggregated by `(execution_id, stage_name)` across repeated
stage calls. These are aggregate measurements, never statistics rows per peptide.

Process RSS is sampled every 10 ms while scopes are active, with endpoints also
observed. `memory_peak_bytes` is the maximum observed RSS, not a guaranteed true
peak and not memory allocated by a particular component. Unsupported process-RSS
measurement produces NULL. The scope/method/interval are recorded in settings.
GPU allocated/reserved memory remains available through the separate stability
monitor; process RSS columns must not be interpreted as VRAM.

Nested iteration durations are measured as wall intervals. The sum of step times
includes only leaf invocations in that iteration and its descendants, so composite
and child durations are not added twice. Sampling does not consume computational
RNGs. `flush_every` commits each specified number of completed step/iteration
events and never samples evaluation records. Finalization commits the remainder.

## Reading and replay

`AnalysisReader` exposes lazy tables and bounded projected scans, all evaluations,
the unique evaluated archive, lineage, component artifacts and retained batches.
`ExperimentReader` discovers bundles and uses the same logical views.

Replay is explicit and supported at LocalEnumeration boundaries before
`prepare_call`. It checks package/model hashes, prepares the selected fragment,
restores recorded optimization/step state and Python/NumPy/Torch RNGs, executes the
fragment and verifies the ordered sequence count and digest. Caller RNG state is
restored. Reconstruction beyond the saved retention scope raises
`DataUnavailableError` rather than inventing missing data.
