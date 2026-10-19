# Root pep_compass library scripts

The repository root is the package root for `pep_compass`. Its canonical artifact
downloaders live in `downloads/<artifact_type>/`:

- `apex/`: default and full APEX model weights.
- `hydramp/`: named HydrAMP model weights.
- `merops/`: prepared MEROPS matrices from an explicitly supplied repository.
- `data/`: the shared repository data archive.

`assets/scripts/downloads/pep_compass` is a relative directory symlink to this
`downloads` directory. Existing destinations and command options are preserved.
