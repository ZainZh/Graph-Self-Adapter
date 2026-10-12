# Twin task-package exporter

`twin_export_packages.py` validates a root-frozen
`graph-self-adapter.twin-final-selection.v1` document and exports the selected
task packages with their matching frozen source snapshots. Different tasks may
select different R0–R3 candidates and snapshots.

The selection contains unique task rows with `task`, `candidate_id`,
`packages`, `source_snapshot`, `graph_sha256`, `model_sha256`,
`reference_sha256`, and `candidate_freeze_sha256`. `packages` may identify a
packages root, one `twin_<task>` directory, or its `solution` directory.
`source_snapshot` may be a path string or `{path, sha256}` where the optional
hash binds `SOURCE_FREEZE.json`. Relative input paths resolve against the input
selection file.

Validate without writing output:

```bash
python tools/task_graph/twin_export_packages.py export \
  --selection /path/to/FINAL_SELECTION.json \
  --output task_packages/twin_rsi_v01 \
  --dry-run
```

Remove `--dry-run` to create a new export directory. Existing destinations are
never replaced. The export excludes training datasets, simulator assets,
demonstrations, and interpreters. Each package retains `EXTERNAL_BINDINGS.json`.
Its generated `run.sh` uses the hash-bound copy of this launcher inside the
export, verifies every exported hash, and invokes the task's selected
`source_snapshot/tools/task_graph/twin_rsi.py`. A launch dry run prints the
fully resolved command without importing MuJoCo or starting an episode.

```bash
bash task_packages/twin_rsi_v01/run.sh \
  --task collect_apple --condition-set development --trial 0 \
  --output /tmp/twin-collect-apple-check --dry-run
```

Use a new output directory for every execution. Test conditions remain subject
to the frozen selection and frozen runner's test authorization checks. Once a
held-out condition is opened, later reuse is development/regression evidence,
not a fresh independent test.
