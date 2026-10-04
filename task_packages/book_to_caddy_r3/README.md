# Book-to-caddy P70 R3 task package

This LIGHT package keeps the frozen object-centric task graph, task controller,
five task-trained weight files, task references, provenance, and the annotated
engineering replay. Standard LIBERO/robosuite assets, the RLinf initial-state
resources, shared DINO/LiLa dependencies, and the Python/GPU runtime remain
installed dependencies.

From the repository root, verify the package without starting a simulator:

```bash
bash task_packages/book_to_caddy_r3/run.sh check
```

Machine-specific paths may be supplied in the ignored
`local/DEPENDENCIES.local.json` file or with `--dependencies PATH`. Runtime
`prove` and `tools` modules always load from this repository. The default
one-shot demo binding resolves to `solution/refs/demo/book_demo_0.hdf5`.

Review path rebasing without executing a rollout:

```bash
bash task_packages/book_to_caddy_r3/run.sh run --dry-run --cell 7 --output /tmp/book_r3_dry_run
```

The standalone release has passed Root's migration review. Explicitly replay
one opened historical condition with:

```bash
bash task_packages/book_to_caddy_r3/run.sh run --cell 7
```

Execution validates the reviewed launcher, task manifest and original runtime
source hashes. It preserves P70 R3 provenance and does not create new independent
test evidence. Each execution writes a new output directory.
