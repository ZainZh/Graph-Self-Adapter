# Xtrainer migration v01

This additive runtime executes the object-centered frozen A2 graph through the existing Xtrainer service. It loads the actual frozen 29-input NumPy skill with fixed z, uses controller FK and a six-dimensional finite-difference DLS retarget, and keeps graph progress, actual joint arrival, grip, release and semantic goals separate. It does not retrain or modify historical graphs, models, sources or evaluations.

Use the existing environment `/home/zheng/miniforge3/envs/geo_new/bin/python`. Root verified imports of torch 2.7.0 CUDA, pyzed.sl and zmq without connecting devices. The older default and MuJoCo environments have different dependency gaps; no installation is required. SAM source and weights are external hash-bound dependencies. New modules remain lazy: importing, constructing and checking them makes no device connections.

## Completed offline validation

`evaluations/xtrainer_v01/implementation/` owns check, synthetic full-graph mock and recorded-image SAM evidence. Mock uses the real graph, weights, controller construction, safety checks and fake I/O, with an explicitly synthetic judge; it is not Root inference or physical task performance. Actual recorded SAM images come only from the source-bound double-cup episode `episode_20261004142813_579607`, not a latest-file search. Root supplied boxes from those actual RGB frames. The initial float32 kernel failure is retained; the bfloat16 attempt supplies masks and metric role geometry for six recorded frames. These timings are recorded SAM call durations, not live latency or segmentation accuracy.

```bash
/home/zheng/miniforge3/envs/geo_new/bin/python tools/task_graph/xtrainer_rsi.py --check
/home/zheng/miniforge3/envs/geo_new/bin/python tools/task_graph/xtrainer_rsi.py --mock --output /tmp/xtrainer-new-mock
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/zheng/github/Mujoco-RSI/.venv/bin/python -m pytest -q tests/test_xtrainer.py tests/test_project_dashboard.py
```

Default operation is file-only check. `--mock` completes the cup's full graph with fake I/O. Mock retains proposed real joint step/velocity limits; its explicitly fake FK cell uses separately labeled workspace/joint ranges. It cannot qualify the real workspace.

## Hardware time mapping and unresolved qualification

The original reference trajectory exceeds the proposed 0.04 rad per command / 0.5 rad/s limits. Check reports original and hardware-scaled envelopes for every stage. The migration explicitly sets `hardware_reference_speed_scale=0.1`, synchronizing nominal joints, gripper, reference progress and matched skill reference time. Hardware command budgets are bounded by `ceil(frozen_budget/scale)+2`; this differs from frozen simulation timing and leaves the desired endpoint unchanged. Both arms retain the original synchronous nominal trajectory, with residual correction limited to the actor. Safety rejects an unsafe residual; it never clips an issued target and calls that the desired goal.

`hardware_ready` stays false: source compatibility and mock integration do not verify actual clearance, firmware, workspace, tool setup, live feedback, grip or task performance. The proposed joint/workspace values require an actual cell review. There is no full-arm collision planner. No hardware connection or motion has been performed in this work.

## Real observation and stage judgments

Actual q, qdot and native TCP come from independent TCP30004 packets, with degree/mm conversion and calibrated User0/Tool1 checks. The existing service's gripper slot and velocity fields are rejected as measurements. Actual aperture remains `None`; commanded native gripper values (0 closed, 1 open) are recorded separately. Current Root multi-frame RGB can judge holding/release under this declared visual contract; it is not a measured aperture or contact certificate. Static spatial overlap cannot sustain grip evidence: a Root judgment has a bounded lease, while continued observation requires nonzero visible object/TCP co-motion. Occlusion, stale data and false or unknown invariants stop execution.

Real ZED capture uses metric depth and SDK intrinsics at the actual live 640×360 resolution. SAM initialization retains the exact first frame, exports it, and waits for a separate prompt file, e.g. `/tmp/xtrainer_actual_first_frame_boxes.json`:

```json
{"frame_sha256":"the exported PNG SHA256","role_boxes":{"white_mug":[157,245,233,333],"red_mug":[260,256,322,347],"box":[314,139,422,237]}}
```

Those example boxes belong to the recorded smoke frame; the operator must label the current frame anew. Initialization consumes that retained image, then grabs and tracks a new frame. Slow initialization output is discarded; live freshness still applies. The shared controller world is the left base; right-base feedback is transformed using `T_left_from_camera @ inverse(T_right_from_camera)`. Metric depth uses the frozen model's visible surface median pooling convention.

The real Root protocol is independent of the simulated paused-clock protocol. `REAL_AGENT_HANDOFFS/STATUS.json` points to a request containing three causal RGB frames and past actual robot feedback. Root requests are capped at 128 per task; task and stage wall budgets are 3600 and 600 seconds. Loop and feedback heartbeat enforce these bounds and stop on exhaustion. No new motion is sent during Root review. A read-only heartbeat checks current feedback, mode, freshness and pose drift while waiting; after reply the runtime acquires new RGB-D and robot feedback and checks age, pose and visible-role drift. The real world keeps moving.

Root writes `ROOT_REPLY.json` with exactly `schema="graph-self-adapter.real-stage-reply.v1"`, request_id/request_sha256/task/stage_id/graph_sha256 from the request, decision, judged_facts, rationale and evidence_frame_ids. Grip/release judgments require multiple causal evidence frames. The independent `ROOT_APPROVAL.json` contains schema `graph-self-adapter.real-root-approval.v1`, request_id, request_sha256, reply_sha256 and actual_root_identity matching the frozen configuration. Decisions cannot contain actions. An `unknown` decision makes every requested fact unknown, even if the reply contains a true fact; the complete original reply is retained. Missing, wrong, stale or drifting evidence remains unknown or stops; mock replies are never a real fallback.

## Future explicit live modes — not run in this scope

```bash
/home/zheng/miniforge3/envs/geo_new/bin/python tools/task_graph/xtrainer_rsi.py --shadow --config REVIEWED_CONFIG.json --scope REVIEWED_SCOPE.json --execution-freeze REVIEWED_FREEZE.json --output NEW_SHADOW_OUTPUT
/home/zheng/miniforge3/envs/geo_new/bin/python tools/task_graph/xtrainer_rsi.py --execute --config REVIEWED_CONFIG.json --scope REVIEWED_SCOPE.json --execution-freeze REVIEWED_FREEZE.json --output NEW_RUN_OUTPUT
```

Both connection modes require an exact scope/freeze before any socket or camera opens. Shadow only reads feedback, real RGB-D and controller FK, reporting independent actual-TCP/FK discrepancies; it sends no robot commands. Execute requires explicit hardware motion authorization, a positive episode budget, reviewed hardware safety and exact task/output bindings, including graph/model/reference/SDK/calibration/SAM/new-source hashes. The current scope permits zero hardware commands, so execute is rejected.

Use only an already-running local ZMQ service (`127.0.0.1:6001`). Native command layout is Lq6 radians, Lg, Rq6 radians, Rg with explicit arm flags. The migration never constructs `DobotRobot`, starts a service, enables, homes or opens serial gripper ownership. Before commands, independent feedback must match the native initial pose and confirmed initial gripper state; there is no automatic homing. Actual TCP is compared against native FK to check units/tool calibration. FK delays trigger a new feedback/freshness check before send. Failures attempt Dashboard `ResetRobot()` for both arms, with full reply validation, and retain failed stop acknowledgments.
