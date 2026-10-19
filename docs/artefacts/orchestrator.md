# Hydra orchestration with direct constructors

Hydra composes groups and parameter overrides; each `_target_` selects an
importable class or ordinary preset function. The same Python constructors are
used interactively and in planned jobs. No registration or ConfigStore setup is
required. The implementation is checked against Hydra 1.3's documented
[object instantiation](https://hydra.cc/docs/1.3/advanced/instantiate_objects/overview/)
and [override grammar](https://hydra.cc/docs/1.3/advanced/override_grammar/basic/).

```bash
pep-compass plan pipeline=simple pipeline.mutation_candidates=8
pep-compass run pipeline=simple ++runner.execution.seed=7
pep-compass run --multirun pipeline=simple ++runner.execution.seed=7,8
pep-compass plan pipeline=lebo pipeline.iterations=2 pipeline.trajectories=1
pep-compass run backend=subprocess backend.max_workers=2
pep-compass run backend=slurm backend.template_path=/path/to/site-template.sh
```

Built-in pipeline YAML groups are short calls to `pep_compass.predefined.simple`
and `pep_compass.predefined.lebo`. A custom group may define components directly:

```yaml
_target_: pep_compass.Pipeline
autoencoder:
  _target_: pep_compass.autoencoder.HydrAMP
  jacobian_mode: approx
steps:
  - _target_: pep_compass.steps.filters.CandidateSubset
    count: 3
  - _target_: pep_compass.steps.oracles.Hydrophobicity
```

Use `--config-path` and `--config-name` for application-owned YAML groups. Ordinary
Hydra defaults, interpolation and group replacement apply. `+key=value` adds a
previously absent option; `++key=value` adds or overrides it.

The plan freezes graph IDs, resolved inputs, runner settings, package source
identity and independent result paths before loading model weights. A worker
restores the schema into the public `Pipeline` wrapper and executes one job.
Atomic locks prevent concurrent ownership of one result directory. Completed
results can be skipped only after validating settings, sources and model hashes.

Execution backends are ordinary `ExecutionBackend` implementations with a `run`
method. Select a custom importable backend with `_target_`; no manager changes
are needed. Built-ins are local sequential execution, bounded subprocesses and
Slurm array script generation. Subprocess/Slurm resources must be serializable
constructor declarations. Explicit application modules may be passed with
`--plugin module` to include their source files in plan fingerprints; this imports
code without registering factories.

A server-owned Slurm template contains `${array_max}` and `${command}`. Optional
`${working_directory}`, `${plan_path}` and `${directives}` are supported. Load the
template with `template_path` so it is frozen during planning. Shell placeholders
inside that frozen template are not interpreted again by OmegaConf. Script
submission remains an explicit cluster operation.

Every persistent run contains `settings.json` plus `results.sqlite`; the
orchestrator adds frozen plans and compact job statuses outside those bundles.
