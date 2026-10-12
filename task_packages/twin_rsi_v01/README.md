# Twin RSI selected task packages

This directory contains the root-frozen per-task selection. Each task runs the
source snapshot selected with that candidate; later live repository code is not
used for the episode.

Check a command without starting MuJoCo:

```bash
bash task_packages/twin_rsi_v01/run.sh --task collect_apple --condition-set development --trial 0 --output /tmp/twin-check --dry-run
```

Run into a new output directory by removing `--dry-run`. Set `TWIN_PYTHON` to
the Python environment containing MuJoCo, NumPy and SciPy when needed. Never
reuse an output directory. Independent test conditions require the frozen final
selection. Once a held-out condition has been opened, any later reuse belongs
to development/regression evidence and must not be reported as fresh test data.

Standard MuJoCo scenes, meshes, demonstrations, and Python environments remain
external. Their paths and hashes are retained in each `EXTERNAL_BINDINGS.json`.
