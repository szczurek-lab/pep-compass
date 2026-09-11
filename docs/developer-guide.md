# Developer Guide

This guide defines how to add and maintain PepCompass components. Runtime
relationships are documented in
[Technical Architecture](technical-architecture.md); public configuration is
documented in [User Guide](user-guide.md); package-split rationale is
documented in [Architecture Decisions](architecture-decisions.md).

## Table of Contents

- [Developer Guide](#developer-guide)
  - [Table of Contents](#table-of-contents)
  - [Library Structure](#library-structure)
    - [Package Map](#package-map)
    - [Module Responsibilities](#module-responsibilities)
    - [Naming Conventions](#naming-conventions)
    - [Public and Internal APIs](#public-and-internal-apis)
  - [Adding Components](#adding-components)
    - [Implementation Procedure](#implementation-procedure)
    - [Component Placement](#component-placement)
    - [Common Component Contract](#common-component-contract)
    - [Adding an Autoencoder](#adding-an-autoencoder)
    - [Adding a Walker](#adding-a-walker)
    - [Adding a Mutation Generator](#adding-a-mutation-generator)
    - [Adding a Filter](#adding-a-filter)
    - [Adding an Oracle](#adding-an-oracle)
    - [Adding a Merge Policy](#adding-a-merge-policy)
    - [Adding a Workflow](#adding-a-workflow)
    - [Strategy Registration](#strategy-registration)
    - [Parameter Validation](#parameter-validation)
  - [Mandatory Requirements](#mandatory-requirements)
    - [Type and Batch Safety](#type-and-batch-safety)
    - [Empty-Batch Handling](#empty-batch-handling)
    - [Deterministic Randomness](#deterministic-randomness)
    - [Tracking Compatibility](#tracking-compatibility)
    - [Logging](#logging)
    - [Docstrings](#docstrings)
    - [Documentation](#documentation)
    - [Unit Tests](#unit-tests)
    - [Contract Tests](#contract-tests)
    - [Registry Tests](#registry-tests)
    - [Validation Configurations](#validation-configurations)
  - [Extension Checklist (for LLMs)](#extension-checklist-for-llms)

## Library Structure

### Package Map

```text
src/pep_compass/
  core/
    specification.py     PipelineSpecification, ComponentSpecification, ...
    builder.py            PipelineBuilder: specification -> PepCompassPipeline
    validation.py         structural + registered-component validation
    estimation.py         static candidate/memory estimation (dry-run)
  autoencoder/
    base.py               Autoencoder contract
    registry.py            named model/checkpoint descriptors
    factory.py              method + model + parameters -> Autoencoder
    geometry.py            decoder-Jacobian-derived operations
    strategies/hydramp/
  data/
    optimization.py        CandidateBatch, batch fields (runtime data model)
    result_schema.py       versioned on-disk result format
    dataset.py              logical, lazy view of loaded results
  optimization/
    engine/
      execution/           Step, OptimizationContext, OptimizationState, OptimizationResult
      operations/           Flow, Loop, Parallel (+ merge), LocalEnumeration
    components/
      walkers/
        base.py manager.py strategies/
      mutation_generators/
        base.py manager.py strategies/
      filters/
        base.py manager.py registry.py
        direct/{constraints,controls,optimization,structural}/
        ranked/{base.py,scoring/{latent_geometry,model_scores,helpers},selection}/
      oracles/
        base.py manager.py strategies/
      helpers/
    stability_estimation/  estimation.py monitoring.py
    tracking/               ExecutionScope, StepTracker
    pipeline.py             PepCompassPipeline (public executable object)
  runtime/
    cli.py                  pep-compass entry point
    configuration/          loading.py validation.py pipeline.py schema.py
    planning/               input.py variants.py plan.py validation.py
    workflows/               base.py (RuntimeWorkflow protocol) composable.py pogs.py
    backends/                subprocess.py slurm.py
    output/                  writer.py tracking.py (CSVStepTracker)
    device.py test_run.py runner.py
  analysis/
    reader/                 ExperimentReader, ExperimentSelection, RunReplay, MetricsStore
    analysis_types/locality/
    resampling/ visualization/
  registry/
    core.py                 Registry: names, factories, services and parameter contracts
    components.py           component-family catalog used by PipelineBuilder
    bootstrap.py            one-time built-in registration loading
  utils/
    strategy_factory.py     parameter-contract compatibility helpers
```

Configurations belong under `assets/experiments/configs/`, with
validation-scale examples under `assets/experiments/configs/validation/`.
Persisted run output goes to `experiments/results/`. Executable repository
scripts (model-weight download helpers) belong under `assets/scripts/`.
Tests mirror this package layout under `tests/`.

### Module Responsibilities

`base.py` defines a family contract (an ABC every strategy in that family
subclasses). `manager.py` stores named factories (`{Family}Manager`, one per
walkers/mutation_generators/filters/oracles). `strategies/` (or, for filters,
`registry.py` plus `direct/`/`ranked/`) contains implementations and their
registration decorators. Do not add a new top-level package for one strategy
when it belongs to an existing family.

`core` wires components but does not implement their algorithms.
`optimization` defines generic execution/data contracts *and* the science —
it must remain runnable from plain Python without `core`, YAML, or the CLI.
`runtime` orchestrates configuration, planning, backends and persistence but
does not select scientific strategies implicitly.

### Naming Conventions

Use singular class names and snake-case public registry names. A strategy
name must identify behaviour, not the experiment that introduced it.

Use the established suffixes when applicable:

- `Walker` for latent-position transitions;
- `Generator` for one-to-many candidate generation;
- `Filter`/`Selector`/`Constraint`/`ScoreFunction`/`SelectionRule` for
  admissibility, ranking and policy transformations (see
  [Adding a Filter](#adding-a-filter) for which one applies);
- `Oracle`/`BlackBox` for objective evaluation;
- `Manager` for a strategy registry.

### Public and Internal APIs

Registry names and YAML parameters are public API. `CandidateBatch`, batch
field classes, `Step`, `OptimizationContext`/`OptimizationState`, and manager
`build`/`register`/`methods`/`validate` methods are developer contracts.
Strategy helper functions are internal unless exported explicitly.

Changing a public method name or parameter requires updating the responsible
`parameter_contract`/factory signature and the [User Guide](user-guide.md).

## Adding Components

### Implementation Procedure

1. Select the existing component family responsible for the behaviour.
2. Read its `base.py`, `manager.py` and one current strategy.
3. Implement the smallest class satisfying the family contract.
4. Add a factory and public registry name.
5. Declare accepted and required parameters.
6. Add contract and failure tests.
7. Add a validation configuration that executes the real runner (`pep-compass
   dry-run`, then `test-run`).
8. Document public parameters in the [User Guide](user-guide.md).

Do not modify `core.builder.PipelineBuilder._build_step`/`_build_component`
when adding another implementation of an existing family. Core changes are
required only for a new *operation type* (a new sibling to `walker`,
`mutation_generator`, `filter`, `oracle`, `loop`, `parallel`,
`local_enumeration`), not a new strategy.

### Component Placement

Place a small implementation directly below its family's `strategies/`
package (or, for filters, below `direct/<category>/` or
`ranked/scoring/<category>/`). Create a subpackage when the implementation
contains multiple cohesive modules, model files or adapters. Reuse an
existing semantic subcategory instead of inventing a new one per strategy.

Scientific model code retained for comparison stays inside the owning
strategy package (see `oracles/strategies/apex_original/` next to
`apex/`) rather than a separate backup tree.

### Common Component Contract

A pipeline component accepts one `CandidateBatch` and returns one
`CandidateBatch`. It may change row count, sequences, latent origins or
fields only through batch operations that preserve column alignment
(`select`, `repeat_from_parents`, `with_field`, `concatenate` — see
[Technical Architecture](technical-architecture.md#candidate-data-model)).

Algorithm data needed by later steps belongs in `CandidateBatch.fields`.
Execution diagnostics belong in tracking. Cross-iteration observations and
control state belong in `OptimizationState`.

### Adding an Autoencoder

Implement the `Autoencoder` contract (`autoencoder/base.py`) in
`autoencoder/strategies/<method>/`, providing encoding, decoding, decoder
Jacobian and field-derivative operations used by walkers and geometry-based
filters. Register the implementation in `autoencoder/registry.py`, and add
one or more **named model variants** (checkpoint descriptors) rather than
hard-coding a single checkpoint — `method` selects the implementation,
`model` selects the variant (see
[Architecture Decisions](architecture-decisions.md#autoencoder-vs-models)).
`AutoencoderFactory` constructs the instance; `PipelineBuilder` receives it
and makes it available as the `autoencoder` service to component factories
whose registrations declare that service.

### Adding a Walker

Subclass `Walker` (`optimization/components/walkers/base.py`) and implement
`_execute`. A walker normally replaces latent origins and decoded sequences
while preserving compatible incoming fields.

Fields required by MUTANG use the established names:

```text
walker.singular_values
walker.left_vectors
walker.adjusted_time_step
walker.tangent_space
```

If a new walker cannot produce these values, document that it is not
compatible with the current MUTANG strategy rather than creating placeholder
fields.

### Adding a Mutation Generator

Subclass `MutationGenerator`
(`optimization/components/mutation_generators/base.py`). For one-to-many
generation:

1. compute generated sequences per parent;
2. build one parent-index list;
3. call `batch.repeat_from_parents(parent_indices)`;
4. replace sequences with `with_sequences`;
5. attach reusable generation metadata as batch fields;
6. call `context.state.record_generated_candidates`.

Do not re-encode generated sequences when their required origin is the
concrete parent trajectory position.

### Adding a Filter

`optimization/components/filters/` has two extension styles — pick the one
matching what the transformation actually does:

- **Direct** (`filters/direct/{constraints,controls,optimization,structural}/`):
  subclass `Filter` (or `Selector` when the component depends on
  optimisation policy, observation history or resource targets —
  register it through `FilterManager` regardless, since `filter` is the one
  public step operation for both). Calculate any reusable score before
  selecting rows, attach it as a candidate-aligned field, and use
  `select(indices)` to return accepted candidates. State-dependent selectors
  read `context.state`; they must not encode persistent state into tracking
  fields.
- **Ranked** (`filters/ranked/{scoring,selection}/`): implement a
  `ScoreFunction` (under `scoring/`, grouped by what it scores — e.g.
  `latent_geometry/` for tangent-space-derived scores, `model_scores/` for
  model-derived scores) and/or a `SelectionRule` (under `selection/` —
  `threshold`, `nucleus`, `top_k`). `RankedFilter` composes any scorer with
  any selection rule; register a fixed combination under its own method name
  (as `lpbebo`/`lams`/`tandem`/`move` do) when that pairing is a named,
  reusable strategy, or let callers compose one directly through the generic
  `ranked` method.

A filter may return an empty batch. It must not raise only because no
candidate passed its rule.

### Adding an Oracle

Subclass `Oracle`, or adapt an existing POLI black box through
`BlackBoxOracle` (`optimization/components/oracles/strategies/black_box.py`
— see `_black_box_oracle` in `oracles/strategies/__init__.py` for the
established adapter pattern). An oracle must:

- attach a one-dimensional `oracle.<name>.score` tensor;
- attach objective name and direction fields;
- record observations through `OptimizationState`;
- respect the remaining oracle-call budget;
- return a correctly shaped empty score field without calling its model for
  an empty batch.

Import model-specific dependencies lazily inside the registered factory (see
`_black_box_oracle`'s `import_module`) so an unused oracle's dependency does
not prevent library import. `OracleManager` has no default injected services;
an oracle requiring one must declare it explicitly in its registration and
test the boundary.

### Adding a Merge Policy

Implement `BatchMerger.__call__(batches) -> CandidateBatch` in
`optimization/engine/operations/parallel/merge.py` and wire it into
`build_merger`. Define stable ordering, missing-field behaviour, required
score fields, output size and random-state usage before implementation.
`interleave`, `select_best` and `weighted_sample` are declared
(`MergeMethod`) but unimplemented — see
[Architecture Decisions](architecture-decisions.md#known-issues) before
picking one to implement.

A merge policy must return one batch. Deduplication remains a separate
filter unless deduplication is explicitly part of the merge policy contract.

### Adding a Workflow

A `RuntimeWorkflow` (`runtime/workflows/base.py`, a `Protocol`) builds one
`PepCompassPipeline` from a raw pipeline configuration mapping:
`build_pipeline(pipeline_configuration, *, tracker, stability_monitor) ->
PepCompassPipeline`. `ComposableWorkflow`
(`runtime/workflows/composable.py`) is the implemented reference: it builds
one autoencoder, wraps `core.builder.PipelineBuilder`, and parses+builds a
fresh pipeline per plan entry. `PogsWorkflow`
(`runtime/workflows/pogs.py`) is the open example — currently a stub raising
`NotImplementedError`; see [User Guide](user-guide.md#pogs) and
[Architecture Decisions](architecture-decisions.md#pogs) before implementing
it.

### Strategy Registration

Managers expose `register(name)`, `build(method, *, services=None,
**parameters)`, `methods()` and `validate(method, parameters)`
(`optimization/components/*/manager.py`) as facades over the shared
`Registry`. Built-in strategy packages are loaded once by
`registry.load_builtin_registrations()`, which imports their module-level
`@Manager.register(...)` decorators. Registration names must be unique
inside their family.

Walkers, mutation generators and filters route construction through
`Registry.build`. Their manager facades declare `autoencoder` as an
available composition-root service; the registry injects it only when the
factory signature declares that parameter, and rejects any YAML attempt to
override it:

```python
@WalkerManager.register("new_walker")
def build_new_walker(autoencoder, **parameters):
    return NewWalker(NewWalkerImplementation(autoencoder, **parameters))
```

Oracles do not receive a service by default and typically pin an explicit
parameter contract, since `BlackBoxOracle` factories translate parameters
rather than exposing them 1:1:

```python
@OracleManager.register("new_oracle")
@parameter_contract(accepted=_COMMON | {"model_specific_param"})
def build_new_oracle(**parameters):
    return _black_box_oracle("module.path", "NewBlackBox", "new_oracle", parameters)
```

### Parameter Validation

`Registry.validate` infers accepted and required parameters from the
registered factory signature by default. Use explicit `accepted`/`required`
sets through `@parameter_contract` when a factory translates parameters or
uses `*args`/`**kwargs` (as every oracle factory does). Service names declared
in a registry entry are excluded from user-parameter checks and must never be
accepted from YAML. Do not accept arbitrary unknown parameters only to ignore
them.

Every configuration grid is validated after value substitution — a method
grid therefore requires its parameter mapping to be accepted by every selected
method. Family validators additionally validate the nested declarations used
by `sorbes`, `mutang` and `ranked`.

## Mandatory Requirements

### Type and Batch Safety

- Return `CandidateBatch` from every `_execute` method.
- Keep the batch dimension first in tensor fields.
- Use `select` for row filtering.
- Use `repeat_from_parents` for one-to-many expansion.
- Preserve device compatibility for concatenated tensors.
- Use `OptionalField` only when a field is genuinely absent from a branch.

### Empty-Batch Handling

Test empty input when the component can follow a filter or selector.
Components must either return an aligned empty batch or document and
validate a required non-empty precondition. Oracle adapters must not call
third-party models with empty input.

### Deterministic Randomness

Use `context.rng` for NumPy sampling. Do not construct an unseeded generator
in the strategy. Parallel branch identity and independent seeding come from
`context.enter_branch(name, index)` (see
[Technical Architecture](technical-architecture.md#optimisation-context-and-state)),
not from the strategy itself.

Torch sampling currently uses the run-level Torch seed (set once in
`PepCompassPipeline.run`). A component requiring strict branch-local Torch
randomness must accept or derive an explicit `torch.Generator`, or use
`torch.random.fork_rng` the way `LocalEnumeration`'s batched trajectory
execution does, rather than relying on thread scheduling.

### Tracking Compatibility

Do not call tracker lifecycle methods from a strategy. Inherit `Step` and
let `__call__` record execution automatically. Add reusable algorithm values
to batch fields only when later computation actually consumes them.

### Logging

Nontrivial computation and orchestration modules initialise `logger =
get_custom_logger(__name__)`. Use lazy interpolation. Log stage-level
progress at `info`, dimensions and parameters at `debug`, recoverable
anomalies at `warning`, and contextualised handled failures at `error` with
exception information.

### Docstrings

Public Python classes and methods use Sphinx-compatible reStructuredText
docstrings. Document parameters, return values, raised exceptions,
assumptions, side effects and input constraints.

### Documentation

Update the document responsible for the change:

- public YAML method or parameter: [User Guide](user-guide.md);
- runtime relationship or data flow: [Technical Architecture](technical-architecture.md);
- extension or maintenance procedure: this guide;
- reading persisted results: [Analysis Guide](analysis-guide.md);
- why something is structured a certain way: [Architecture Decisions](architecture-decisions.md).

Do not duplicate the same explanation across documents.

### Unit Tests

Test method-specific computation with small deterministic tensors or mock
sequences. A bug fix includes a regression test reproducing the original
failure.

### Contract Tests

Every pipeline component test verifies:

- returned type;
- sequence and latent-origin alignment;
- field lengths;
- parent-origin propagation for expansion;
- expected candidate-count behaviour;
- empty-batch behaviour where applicable.

### Registry Tests

Manager contract tests iterate over every registered strategy in a family
and verify that registered implementations inherit the required base type.
Extend these tests when introducing another manager or operation family.

### Validation Configurations

Add or extend a small file in `assets/experiments/configs/validation/`. Each
file should isolate one component family and use a grid for comparable
methods or parameters. Use the mock sequence CSV under
`assets/peptides_data/`. Validate before a full run:

```bash
uv run --extra <cpu-or-cuda-extra> pep-compass dry-run \
  assets/experiments/configs/validation/<configuration>.yaml

uv run --extra <cpu-or-cuda-extra> pep-compass test-run \
  assets/experiments/configs/validation/<configuration>.yaml
```

Then execute the smallest real variant needed to exercise the component with
`pep-compass run`.

## Extension Checklist (for LLMs)

- [ ] The implementation belongs to an existing component family, or a new
      family is justified in [Architecture Decisions](architecture-decisions.md).
- [ ] Public code, identifiers, comments, logs and configuration are English.
- [ ] The class satisfies its base contract.
- [ ] Candidate columns remain aligned after every transformation.
- [ ] Empty batches are handled or rejected by an explicit precondition.
- [ ] Random operations use controlled random state.
- [ ] The strategy has a unique registry name.
- [ ] Public parameters have an explicit contract.
- [ ] Unit and regression tests cover computation.
- [ ] Contract tests cover batch behaviour.
- [ ] Registry tests recognise the strategy.
- [ ] A validation configuration exercises `dry-run` and `test-run`.
- [ ] The responsible documentation file is updated.
- [ ] Focused tests, full applicable tests and configuration dry-runs pass.
