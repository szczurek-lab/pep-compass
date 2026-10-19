# Direct component architecture

```mermaid
flowchart LR
    Python[Python component constructors] --> Wrapper[core.wrapper.Pipeline]
    YAML[Hydra YAML _target_] --> Python
    Wrapper --> Builder[core.builder.PipelineBuilder]
    Builder --> Schema[data_schemas.PipelineSchema]
    Schema --> Validation[builder.validation]
    Validation --> Compiled[optimization.engine.ExecutablePipeline]
    Compiled --> Runner[optimization.engine.Runner]
    Runner --> Steps[steps: common Step lifecycle]
    Steps --> Saver[optimization.saving.Saver]
    Saver --> SQL[SQLite logical schema]
    SQL --> Reader[AnalysisReader / ExperimentReader]
```

`Pipeline` uses composition. Its builder owns all graph edits and scope commits;
the wrapper delegates and manages execution defaults, display and the last result.
`PipelineSchema` owns immutable `NodeSpec` snapshots. A placement records the
actual import target, portable constructor values, named child roles and one
stable ID. Reusing a component at two positions creates two IDs. Replacing a
placement preserves its ID; insertion and removal do not renumber other IDs.

Every executable node implements `Step`. Flow, loop, branch and local enumeration
nodes declare their owned children through `graph_children()`. Ordinary components
have none. Internal SORBES/MUTANG/scoring strategies are `Configurable` objects,
not independently scheduled graph nodes. Their algorithm-specific interfaces
control geometry, batch dimensions and numerical operations.

`build()` validates IDs, construction and field flow before opening model weights.
It resolves one shared resource declaration, creates fresh component instances,
binds placement IDs and initializes resources. Previously built graphs remain
independent of later edits. Constructor snapshots and ownership are distinct from
mutable execution state. The saved schema is sufficient to reconstruct the same
constructor tree without algorithm-name dispatch.

`Step.contract()` declares required/optional/produced fields, invalidated prefixes
and resources. Missing required data fails before build. Composite validation
follows sequential flow, intersects guaranteed branch fields and validates private
walker/mutation/filter roles inside local enumeration. Model data are acquired by
actual implementations; they do not import the builder or maintain catalogs.

The common lifecycle is `precompute(context)` once per run, then
`prepare_call(batch, context)` and `_execute(batch, context)` per invocation.
`Step.__call__` owns scopes, accounting, timings and handled failure observations.
Components cannot override it. Context copies share one run's optimization state;
scoped RNG streams and identities remain explicit. Parallel branches execute in
stable order when sharing mutable services; independent jobs use subprocesses.

Runner owns fresh state, seeded RNG boundaries, saver and finalization. The
engine returns its graph output directly. Mandatory evaluations are persisted in
one table and read through SQL views; finalization does not build another archive.
Retention does not alter computation. Replay restores a retained operation input,
state, step state and all supported RNGs, then verifies ordered output hashes.

The orchestrator composes Hydra groups, freezes model-free plans and dispatches
independent wrapper invocations through a direct backend `_target_`. Server-specific
Slurm templates are loaded and frozen during planning.
