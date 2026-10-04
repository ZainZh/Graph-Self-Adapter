#!/usr/bin/env python3
"""Independent local payload audit of P58's 32 matched V2 hammer cells."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.fast_adapter import p58_hammer_runner_v02 as runner


def inspect_cell(index, directory, frozen, freeze_sha, review_sha):
    directory=Path(directory)
    seal=runner.base.read(directory/"SEAL.json")
    if (seal.get("schema")!="p58.hammer-development-cell-seal.v2"
            or seal.get("freeze_sha256")!=freeze_sha
            or seal.get("root_review_sha256")!=review_sha
            or seal.get("index")!=index):
        raise ValueError(f"Cell {index}: wrong seal identity")
    actual={path.name:runner.base.sha(path) for path in directory.iterdir()
            if path.is_file() and path.name!="SEAL.json"}
    if seal.get("files")!=actual:
        raise ValueError(f"Cell {index}: file SHA set mismatch")
    before=runner.base.read(directory/"PRE_RUN.json")
    result=runner.base.read(directory/"RESULT.json")
    if (before.get("schema")!="p58.hammer-prerun.v2"
            or before.get("freeze_sha256")!=freeze_sha
            or before.get("index")!=index
            or before.get("cell")!=frozen["cells"][index]
            or before.get("condition")!=frozen["conditions"][frozen["cells"][index]["condition_index"]]
            or before.get("root_review",{}).get("sha256")!=review_sha
            or result.get("schema")!="p58.hammer-development-cell.v2"
            or result.get("index")!=index or result.get("cell")!=frozen["cells"][index]):
        raise ValueError(f"Cell {index}: identity mismatch")
    actions=np.load(directory/"ACTIONS.npz",allow_pickle=False)["executed_actions"]
    metrics=runner.base.read(directory/"METRICS.json")
    telemetry=runner.base.read(directory/"TELEMETRY.json")
    T=len(actions)
    if (actions.shape!=(T,4) or not np.isfinite(actions).all()
            or len(metrics)!=T or len(telemetry)!=T
            or result.get("executed_task_steps")!=T):
        raise ValueError(f"Cell {index}: action/metric/telemetry length mismatch")
    if T and not np.allclose(actions[:,3],.9,rtol=0,atol=1e-7):
        raise ValueError(f"Cell {index}: grip changed")
    if (result.get("status")=="completed" and
            (result.get("frozen_parameters_unchanged") is not True
             or result.get("frozen_fingerprints_before")!=result.get("frozen_fingerprints_after"))):
        raise ValueError(f"Cell {index}: frozen network/readout changed")
    if T:
        with np.load(directory/"PUBLIC.npz",allow_pickle=False) as public:
            if (public["masks"].shape!=(T,2,2,448,448)
                    or public["mask_tokens"].shape!=(T,2,2,55)
                    or public["mask_valid"].shape!=(T,2,2)
                    or not np.array_equal(public["mask_valid"],public["masks"].any(axis=(-2,-1)))):
                raise ValueError(f"Cell {index}: public mask/token contract changed")
        with np.load(directory/"INTERACTION_STATE.npz",allow_pickle=False) as state:
            if any(state[key].shape[0]!=T+1 for key in ("qpos","qvel","mocap_pos","mocap_quat")):
                raise ValueError(f"Cell {index}: simulator-state seal length mismatch")
    original_success=any(bool(m["success"]) for m in metrics)
    if result.get("validated_success") is not original_success:
        raise ValueError(f"Cell {index}: native success differs")
    if original_success:
        first=next(i+1 for i,m in enumerate(metrics) if m["success"])
        m=metrics[first-1]
        if (first!=result.get("first_success_step") or not m["head_contact_seen"]
                or m["target_displacement"]<.045 or m["target_direct_contact"]
                or m["hold_streak"]<5):
            raise ValueError(f"Cell {index}: native contact/slide/hold criterion differs")
    arm=frozen["cells"][index]["arm"]
    if arm!="p30v2_visual_bc":
        for t,(a,row) in enumerate(zip(actions,telemetry)):
            if row.get("t")!=t or not np.allclose(a,row["action"],rtol=0,atol=1e-6):
                raise ValueError(f"Cell {index}: public controller action trace differs")
    pushing=[t for t,row in enumerate(telemetry) if row.get("pushing") is True]
    last=metrics[-1] if metrics else {}
    if original_success:native_reason="native_success"
    elif not last.get("bilateral_handle",0):native_reason="lost_bilateral_grip"
    elif last.get("target_direct_contact",0):native_reason="direct_target_contact_contamination"
    elif not last.get("head_contact_seen",0):native_reason="no_held_head_contact"
    elif last.get("target_displacement",0)<.045:native_reason="insufficient_target_slide"
    elif last.get("hold_streak",0)<5:native_reason="native_hold_not_completed"
    else:native_reason="other_native_failure"
    controller_phase=("historical_bc_no_geometry_phase" if arm=="p30v2_visual_bc"
                      else "public_near_contact_gate_reached" if pushing
                      else "public_near_contact_gate_not_reached")
    grip_loss=next((t+1 for t,m in enumerate(metrics) if not m["bilateral_handle"]),None)
    return dict(index=index,condition_index=frozen["cells"][index]["condition_index"],
        arm=arm,directory=str(directory),seal_sha256=runner.base.sha(directory/"SEAL.json"),
        status=result["status"],success=original_success,actions=T,
        preparation_passed=result.get("preparation_passed"),
        frozen_parameters_unchanged=result.get("frozen_parameters_unchanged"),
        first_push_step=pushing[0] if pushing else None,
        first_bilateral_grip_loss_step=grip_loss,
        max_target_displacement=max((m["target_displacement"] for m in metrics),default=0.),
        last_metric=last,controller_phase=controller_phase,native_failure_reason=native_reason)


def audit(freeze,review,original,nonbc,bc,visual_receipt,output):
    frozen=runner.verify(freeze)
    runner.verify_review(freeze,review)
    original=Path(original);nonbc=Path(nonbc);bc=Path(bc)
    source_receipts=(original/"QUEUE_RESULT.json",nonbc/"RESUME_RESULT.json",bc/"BC_REPAIR_RESULT.json")
    first,second,third=[runner.base.read(p) for p in source_receipts]
    if (first.get("launched")!=8 or second.get("launched")!=21 or third.get("launched")!=4
            or second.get("stopped_on_engineering_error") is not False
            or third.get("stopped_on_engineering_error") is not False):
        raise ValueError("Expected 8+21+4 sealed attempts")
    error=first["results"][6]
    if (error["index"]!=6 or error["result"]["status"]!="engineering_error"
            or error["result"]["executed_task_steps"]!=0
            or third["rows"][0]["index"]!=6 or third["rows"][0]["attempt"]!=2):
        raise ValueError("Original zero-action BC error or one repair changed")
    directories={}
    for row in first["results"]:
        if row["index"]!=6:directories[row["index"]]=original/f"cell_{row['index']:02d}"
    for row in second["rows"]:directories[row["index"]]=nonbc/f"cell_{row['index']:02d}"
    for row in third["rows"]:directories[row["index"]]=bc/f"cell_{row['index']:02d}"
    if sorted(directories)!=list(range(32)):
        raise ValueError("Matched 32-cell grid incomplete or duplicated")
    cells=[inspect_cell(i,directories[i],frozen,runner.base.sha(freeze),runner.base.sha(review))
           for i in range(32)]
    if any(cell["status"]!="completed" for cell in cells):
        raise ValueError("A matched cell has engineering error")
    per_arm={arm:dict(success=sum(c["success"] for c in cells if c["arm"]==arm),
        denominator=4,action_counts=[c["actions"] for c in cells if c["arm"]==arm])
        for arm in runner.base.ARMS}
    paired={}
    for representation in ("absolute_positions","role_relative"):
        groups={mode:{c["condition_index"]:c for c in cells if c["arm"]==f"{representation}_{mode}"}
                for mode in ("fixed","online","direct")}
        for compare in ("online","direct"):
            paired[f"{representation}_{compare}_vs_fixed"]=dict(
                rescue=[i for i in range(4) if not groups["fixed"][i]["success"] and groups[compare][i]["success"]],
                harm=[i for i in range(4) if groups["fixed"][i]["success"] and not groups[compare][i]["success"]])
    value=dict(schema="p58.hammer-development-local-audit.v2",status="complete_local_payload_verification",
        evidence_scope="sealed local payloads and independent metric recomputation; no new physics or GT runtime action",
        freeze=dict(path=str(freeze),sha256=runner.base.sha(freeze)),
        root_review=dict(path=str(review),sha256=runner.base.sha(review)),
        visual_dependency=dict(path=str(visual_receipt),sha256=runner.base.sha(visual_receipt)),
        source_receipts=[dict(path=str(p),sha256=runner.base.sha(p)) for p in source_receipts],
        original_index6_engineering_error=dict(actions=0,retained=True,
            seal_sha256=runner.base.sha(original/"cell_06/SEAL.json")),
        physical_attempts=33,matched_cells=32,automatic_retries=0,
        explicit_zero_action_engineering_retry=1,
        success=sum(c["success"] for c in cells),per_arm=per_arm,paired=paired,cells=cells)
    runner.base.write(output,value)
    return value


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("freeze","review","original","nonbc","bc","visual-receipt","output"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    result=audit(a.freeze,a.review,a.original,a.nonbc,a.bc,a.visual_receipt,a.output)
    print(json.dumps(dict(status=result["status"],matched_cells=result["matched_cells"],
        attempts=result["physical_attempts"],success=result["success"],per_arm=result["per_arm"])),flush=True)
