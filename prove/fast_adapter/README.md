# Fast Adapter P0/P1 Prototype

This directory contains the P0/P1 Fast Adapter prototype used for the numerical 2D toy study. It is not a DINO/RGB/LIBERO/robot rollout. The visual encoder class is an interface placeholder; P1 uses state/target/previous-action toy observations only.

## Scope

- Task: 2D continuous control toy with hidden per-axis gain and bias.
- Runtime observation: state, target, proprio placeholder, previous action.
- Hidden domain parameters: gain and bias are used only by the toy transition and expert action labels, never by runtime inputs or the inner loss.
- Base policy: frozen scripted nominal policy, `a = 0.5 * (target - state)`, clipped to `[-1, 1]`.
- Fast Adapter: `u=256`, `z=32`, bounded residual `0.15` per action dimension.
- Inner update: z-only dynamics-prediction update; model parameters are frozen at eval; query uses fixed z/context.
- Outer training loss: privileged expert action imitation on differentiable clipped action; goal error is measured only.

## Environment

Use the existing conda environment; no dependency installation was required.

```bash
export PYTHONPATH=.
PY=/home/zheng/miniforge3/envs/geometry/bin/python
```

P0 regression:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q tests/test_fast_adapter.py
```

Last recorded result: `9 passed`.

## Training Commands

The initial default inner step (`--inner-lr 0.05`) was kept as a negative development result: it learned a useful fixed adapter but showed negligible extra z-adaptation benefit. The selected final configuration uses `--inner-lr 5.0`, chosen by the preplanned second and final development check. No third configuration was introduced.

Selected Fast Adapter training, one seed shown:

```bash
PYTHONPATH=. $PY tools/fast_adapter/train_toy2d.py   --updates 1500   --batch-size 64   --support-k 3   --query-horizon 10   --train-seed 11   --inner-lr 5.0   --output-dir evaluations/fast_adapter_v01/train_seed11_u1500_innerlr5   --log-every 500
```

Repeat with `--train-seed 22` and `--train-seed 33`, changing `--output-dir` accordingly.

History baseline training, one seed shown:

```bash
PYTHONPATH=. $PY tools/fast_adapter/train_history_toy2d.py   --updates 1500   --batch-size 64   --support-k 3   --query-horizon 10   --train-seed 11   --output-dir evaluations/fast_adapter_v01/history_train_seed11_u1500   --log-every 500
```

Repeat for seeds `22` and `33`. The history baseline uses the same observations and residual bound. It updates GRU context during support only and freezes context during query.

## Final Evaluation Commands

Final Fast Adapter eval uses the frozen selected checkpoints, final seeds, and all K values. It writes `eval_raw.jsonl` plus `eval_summary.json` with freeze audit hashes.

```bash
PYTHONPATH=. $PY tools/fast_adapter/evaluate_toy2d.py   --checkpoint evaluations/fast_adapter_v01/train_seed11_u1500_innerlr5/checkpoint.pt   --output-dir evaluations/fast_adapter_v01/final_fast_seed11   --train-seed 11   --split final   --domain-seed 91000   --query-seed 101000   --shuffle-seed 51000   --num-domains 256   --query-repeats 8   --k-values 0 1 2 3 5   --query-horizon 10
```

Repeat for seeds `22` and `33`. The support seed is fixed inside the evaluator as `domain_seed + 1000`, which is `92000` for final. Shuffle-z uses the final roll-1 mapping.

Final History eval is run per K:

```bash
PYTHONPATH=. $PY tools/fast_adapter/evaluate_history_toy2d.py   --checkpoint evaluations/fast_adapter_v01/history_train_seed11_u1500/checkpoint.pt   --output-dir evaluations/fast_adapter_v01/final_history_seed11_k3   --train-seed 11   --split final   --domain-seed 91000   --support-seed 92000   --query-seed 101000   --num-domains 256   --query-repeats 8   --support-k 3   --query-horizon 10
```

Repeat for seeds `11,22,33` and K `0,1,2,3,5`.

## Artifacts

- Freeze manifest: `evaluations/fast_adapter_v01/FREEZE_MANIFEST_SELECTED_INNERLR5_R3_FINAL.json`
- Final receipt: `evaluations/fast_adapter_v01/FINAL_EVAL_RECEIPT.json`
- Parameter/time stats: `evaluations/fast_adapter_v01/PARAM_AND_TIME_STATS.json`
- Numerical scope note: `evaluations/fast_adapter_v01/NUMERICAL_SCOPE_NOTE.json`

The final receipt lists every raw JSONL path and hash. Records/reporting should consume the raw files rather than rerunning evaluation.
