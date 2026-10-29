# Repository script shortcuts

Executable scripts belong to the library that owns their implementation.
This directory contains relative directory symlinks, grouped by operation and
library. Edit the canonical library script; the shortcut immediately sees the
same file.

| Shortcut | Canonical location |
|---|---|
| `downloads/proteolysis_data_manager` | `assets/tools/proteolysis_data_manager/scripts/downloads` |
| `integration/proteolysis_data_manager` | `assets/tools/proteolysis_data_manager/scripts/integration` |
| `generation/blosum_matrix` | `assets/tools/blosum_matrix/scripts/generation` |
| `downloads/pep_compass` | `scripts/downloads` at the root pep_compass library |

Downloads acquire raw files or model artifacts. Normalization belongs to
integration; matrix construction belongs to generation.

Examples from the repository root:

```bash
.venv/bin/python assets/scripts/downloads/proteolysis_data_manager/half_time_data/download_peplife2.py --help
.venv/bin/python assets/scripts/downloads/proteolysis_data_manager/download_all.py --help
.venv/bin/python assets/scripts/integration/proteolysis_data_manager/half_time_data/normalize_datasets.py --help
.venv/bin/python assets/scripts/generation/blosum_matrix/export_blosum62.py --help
bash assets/scripts/downloads/pep_compass/apex/download_apex_models_default.sh
```

The preceding proteolysis script trees and legacy-only tests are preserved locally
under `tmp/archive_old_analysis/`, with their original relative paths. See that
directory's `README.md` before reviewing or restoring an archived producer.
