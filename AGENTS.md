# Graph Self Adapter repository guidance

This repository is the canonical workspace for Task Graph and stage-local Fast Adapter development, extracted from Interaction-RSI at the user's request. Do not implement new Graph/Adapter changes in the old repository. Preserve the `prove` and `tools` namespaces for checkpoint compatibility.

Read by task:
- Research scope or any major change: [spec.md](spec.md).
- Current work and evidence: [docs/current-work.md](docs/current-work.md), [dashboard/STATUS.json](dashboard/STATUS.json).
- Architecture and learning boundaries: [docs/architecture.md](docs/architecture.md).
- Agent context and stage judging: use the phase-specific reading contract in [architecture](docs/architecture.md); read only current-node evidence and related interfaces, expand on uncertainty or structural failure.
- Runtime setup and checks: [docs/development.md](docs/development.md), [docs/dependencies.md](docs/dependencies.md).
- Reusing the current task: [task package](task_packages/book_to_caddy_r3/README.md).

Before every major change, reread spec.md. Once adopted, update the spec and Dashboard in the same work phase. Routine fixes need only relevant sections; no planning-with-files ritual or per-action journal is required. Keep each result's canonical evidence owner; do not rewrite frozen reports, scopes, seals or checkpoints.

The system uses an object-centered graph: every stage declares its actor/object/goal roles, entry, completion, holding invariants, budgets and failure. Low-frequency Agent revisions may change graph, Code, training data and bounded skills during development. Frozen evaluation cannot revise these. Only the contracted local latent z may update inside the designated skill stage, using past/current permitted feedback; reset/archive it at stage boundaries.

Evaluator-only continuous object truth cannot enter normal runtime policy or adaptation inputs. Declare simulator visible masks, calibration, proprioception, fixed training priors and Boolean supervisors explicitly. Offline labels and post-seal diagnostics are separate from candidate inputs. Seal predictions/results before evaluation. Opened conditions are regression, not fresh held-out. Local skill completion is not full task success.

Preserve unrelated worktree changes. Stage only task files. Do not introduce an external model API as a prerequisite. The root agent owns scientific planning and acceptance; implementation may be delegated as explicitly requested by the user. Do not launch simulator rollouts as part of docs, packaging or tests. New experiments need an explicit scope and result-free freeze, not repeated permission for routine authorized work.

Task-specific trained weights live in solution/ and use Git LFS. External LIBERO, DINO, pretrained LiLa, Python installations and machine paths do not belong in Git. `local/` is ignored. Avoid code symlinks or imports back to Interaction-RSI. Old experiment drivers may retain historical asset references, but the active task package must use this repository's execution code.

External pushes require authorization for destination and scope. Initial publication to ZainZh/Graph-Self-Adapter is explicitly authorized in the current session; do not infer perpetual authorization to publish unrelated work.
