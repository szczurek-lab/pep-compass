# User guide

Start with [the direct API tutorial](artefacts/wrapper_new.md) and run
[wrapper_tutorial.ipynb](tutorial/wrapper_tutorial.ipynb) for the full sequence.

```python
from pep_compass.predefined import simple, lebo

pipeline = simple(seed=7, mutation_candidates=4, oracle_calls=3)
pipeline.describe()
pipeline.describe_component(node_id=pipeline.to_schema().steps[-1].node_id)
pipeline.show()
compiled = pipeline.build()
result = pipeline.run(["FLYKWWIRIGRLKL"])
print(result.reader.get_archive())
result.save("results/example")
result.close()
```

Presets are ordinary functions returning editable wrappers. They initialize no
weights. Instantiate components directly to customize the computation, then use
`add_component`, `insert_component`, `replace_component` or `remove_component`.
Use stable node IDs for edits. A returned `ExecutablePipeline` is an independent
snapshot and supports `run`, `set_runner_settings` and `show`.

Use [Hydra orchestration](artefacts/orchestrator.md) for YAML composition, sweeps,
independent workers and Slurm templates. YAML selects the same constructors via
`_target_`. No registries, decorators or old declaration grammar are required.
