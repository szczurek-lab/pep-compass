# Direct Python components

Import actual implementations from `pep_compass.steps.<category>` and shared
resources from `pep_compass.autoencoder`. Category `__init__.py` files explicitly
export their implementations under `strategies/`. Imports do not register objects
or load model weights. Ordinary preset functions live in `pep_compass.predefined`.

```python
from pep_compass import Pipeline
import pep_compass.steps as ps  # ps - PipelineSteps
import pep_compass.autoencoder as ae

pipeline = Pipeline(autoencoder=ae.HydrAMP(jacobian_mode="approx"))
node_id = pipeline.add_component(ps.filters.CandidateSubset(count=3))
pipeline.add_component(ps.oracles.Hydrophobicity())
pipeline.describe()
pipeline.describe_component(node_id=node_id)
pipeline.replace_component(node_id, ps.filters.CandidateSubset(count=2))
compiled = pipeline.build()
result = pipeline.run(["FLYKWWIRIGRLKL"], seed=7)
print(result.archive)
result.close()
```

`Pipeline` owns a `PipelineBuilder`; editing is implemented by that builder.
`PipelineSchema` stores immutable constructor snapshots, child ownership and
stable placement IDs. `build()` validates input-field dependencies, initializes
shared resources, creates fresh components and returns `ExecutablePipeline`.
Editing a builder does not change an already returned executable graph.
`compiled.run()` uses the same Runner and saver as `pipeline.run()`.

`Pipeline([...])`, incremental `add_component(...)`, and a standalone
`PipelineBuilder` are equivalent construction styles. `builder.loop(...)` is a
context manager for collecting a body; successful scope exit preserves child IDs.
A failed scope commits nothing. Required roles such as a local walker can be
replaced but cannot be removed. Insertions target ordered sequences.

`describe(display="display")` automatically renders formatted tables;
`display="pd"` returns a pandas DataFrame, and `display="raw"` returns records.
`describe_component(Class)` shows constructor defaults; passing a configured
instance also shows resolved internal stages. `technical=True` exposes field and
resource contracts. Constructors are the source of defaults; no parallel config
classes maintain a second set.

`show()` returns an inline SVG. Mermaid is textual diagram source requiring a
Mermaid renderer; SVG is the ready picture. `to_schema()` returns the immutable
schema, while `json`, `text`, `mermaid` and `svg` select portable representations.
Declared and compiled graphs share placement IDs. Candidate lineage is a separate
run-time structure, not a graph position.

Runner defaults are grouped under `execution`, `storage` and `results`. The new
notebook documents every setting and default. Per-run overrides do not modify
wrapper defaults. Oracle limits apply before a new evaluation batch; an entire
completed batch remains in the archive even if it crosses the threshold.

The run bundle contains `settings.json` and `results.sqlite`. Every actual
peptide/objective evaluation is retained in `evaluations`, independently of later
selection. The archive is a SQL view over those records. A run-local cache avoids
scoring the same sequence again for an identical configured objective, including
duplicates within a batch. Only newly scored unique sequences consume the oracle
budget. Different configurations cannot silently share the same objective field.

| Option | Default | Meaning |
| --- | --- | --- |
| `metrics_level` | `basic` | `none`: invocation identities only; `basic`: step/iteration timings, counts and sampled process RSS; `basic_and_components`: additionally aggregated internal stages |
| `save_lineage` | `True` | Ordered batch members and compressed pairwise/broadcast parent relations; unscored ancestor strings are unnecessary |
| `component_data` | `none` | `summary`: display metadata; `full`: metadata and shared output-field artifacts, excluding peptide-string provenance columns |
| `reconstruction` | `none` | `boundaries`: input/state/RNG replay checkpoints; `trajectories`: also ordered step outputs, walker points and the actual graph output |
| `flush_every` | `1` | Commit after this many completed step/iteration events; never sample or discard evaluation rows |

`Snapshot("label")` names a batch reference and requires `boundaries` or
`trajectories`. `RunResult.save(path)` exports retained data without recomputing.
Memory and persistent storage share the SQL schema. `AnalysisReader` opens one
bundle; `ExperimentReader` discovers collections and exposes projected scans.
Use `reader.get_evaluations()` for all actual scoring records, `get_archive()` for
unique evaluated sequence/objective pairs, and `get_lineage()` for structural
relations. The `evaluation_records` and `evaluation_archive` views expose sequence
strings by joining the sole `peptides` table. Missing reconstruction data raises
`DataUnavailableError`; it is never replaced with an unrelated evaluated batch.

See [the run data contract](../run-data-contract.md) for keys, retention and replay.

Replay is explicit: `reader.reconstruct_local_enumeration(checkpoint_id)` builds
only that fragment, restores saved state and RNGs, and verifies ordered output
against the recorded digest. It requires the recorded package source and model
identity. Locality analyses additionally require trajectories and a retained
candidate batch. Old CSV/PT readers and declaration formats are removed.

The full runnable sequence is in [wrapper_tutorial.ipynb](../tutorial/wrapper_tutorial.ipynb).
