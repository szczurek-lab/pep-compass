# Architecture decisions

1. Actual Python objects and Hydra `_target_` select implementations. Mutable
   registries, managers and registration side effects are removed.
2. Category namespaces export actual implementations under `strategies/`.
   Package initializers contain public exports or package documentation only.
3. `Pipeline` composes `PipelineBuilder`. Editing is implemented once by the
   builder; `PipelineSchema` stores independent immutable declarations.
4. Graph placements and candidate lineage have different identities. Scope
   commits preserve child placement IDs; replacement preserves the target ID.
5. Compilation constructs new components and binds shared resources after
   validation. Configuration and execution state have distinct lifetimes.
6. Constructor defaults are authoritative. Redundant forwarding configuration
   dataclasses and duplicate preset registries are removed. Presets are functions.
7. `Step` owns lifecycle/accounting. Internal algorithm strategies use explicit
   interfaces and manage numerical batches internally. Family marker classes
   with no behavior are removed.
8. Retained abstract strategy classes define actual signatures (`score`, `select`,
   `generate`, geometry/update calls). `ExecutionBackend` defines dispatch and
   resource freezing. These contracts constrain implementations; they are not
   empty category labels. Exception subclasses express distinct failure cases.
9. Model file requirements belong to model implementations. Shared artifact
   validation/hashing is implemented once. Named weight variants describe data;
   they do not dispatch algorithm factories.
10. Ranked convenience policies are composition functions returning `Ranked`,
    avoiding empty subclasses that only choose scoring/selection defaults.
11. Memory and SQLite share a logical result schema. Evaluated archives and the
    outgoing batch remain separate. Replay retains state/RNG boundaries and
    checks ordered output hashes. Unsupported payload types fail explicitly.
12. No old import aliases, parsers, result-layout adapters or migration layer are
    maintained on this branch. The previous notebook is preserved unchanged;
    the supported notebook is `tutorial/wrapper_new.ipynb`.
