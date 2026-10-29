# Experiments

Analyses and reports, organized by project part.

## Catalog convention

```text
experiments/<project_part>/<problem>/<fragment_of_problem>/<analysis>.ipynb
```

Each level narrows the scope: project part, problem within it, fragment of the problem, single analysis.
Notebooks within a fragment are prefixed `0_NN_` in reading order.

Example: `pcpsackm/datasets_analysis/half_time/0_01_acqusition_analysis.ipynb` is the first analysis
of the half-life fragment of the dataset-analysis problem in the project part `pcpsackm`.

## Project parts

| Directory | Scope |
|---|---|
| [pcpsackm](pcpsackm/README.md) | Protease-conditioned peptide stability and cleavage kinetics modeling |
| `depracated` | Superseded analyses, kept for reference |
