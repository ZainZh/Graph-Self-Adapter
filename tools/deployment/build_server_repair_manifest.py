#!/usr/bin/env python3
"""Record the additive evaluator/SAM compatibility repair for sunzheng."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    repo = Path(__file__).resolve().parents[2]
    origin_twin = Path("/home/zheng/github/Mujoco-RSI/twin")
    remote_twin = Path("/home/ubuntu/github/Mujoco-RSI-v22/twin")
    records = []
    for source in sorted(origin_twin.glob("*.py")):
        sha = digest(source)
        records.append({"kind": "exact_python_closure_copy", "source": str(source),
            "source_sha256": sha, "target": str(remote_twin / source.name), "target_sha256": sha})
    shim = repo / "tools/deployment/server_sam2_connected_components.py"
    records.append({"kind": "server_abi_compatible_connected_components_backend",
        "source": str(shim), "source_sha256": digest(shim),
        "target": "/home/ubuntu/github/SAM2_streaming/sam2/_C.py", "target_sha256": digest(shim),
        "semantics": "OpenCV connectivity=8; foreground component labels and per-pixel areas; int32 returned on input tensor device"})
    validation = repo / "evaluations/multitask_parallel_v22/dual_tool/server_shared/SAM_FORWARD_VALIDATION.json"
    records.append({"kind": "actual_initialize_and_track_evidence", "source": str(validation),
        "source_sha256": digest(validation),
        "target": "/home/ubuntu/github/Graph-Self-Adapter-v22/evaluations/multitask_parallel_v22/dual_tool/server_shared/SAM_FORWARD_VALIDATION.json",
        "target_sha256": digest(validation)})
    manifest = {
        "schema": "graph-self-adapter.remote-shared-repair.v1", "server": "sunzheng",
        "base_shared_manifest_sha256": "09d8173a3db19a62d28949f60ad8ceae11eab24153094a5ea2765f698071ea75",
        "reason": "complete evaluator relative-import closure and replace an ABI-incompatible copied SAM extension without disabling postprocessing",
        "server_has_nvcc": False, "server_glibc": "2.31", "torch": "2.4.1+cu121",
        "incompatible_binary_preserved": {"source": "/home/zheng/github/SAM2_streaming/sam2/_C.so",
            "source_sha256": "1212449f9650b257a81f17a36283827a8243f0a741494cf92f9f99a7ac93b761",
            "target": "/home/ubuntu/github/SAM2_streaming/sam2/_C.incompatible_glibc232_torch27.so",
            "target_sha256": "1212449f9650b257a81f17a36283827a8243f0a741494cf92f9f99a7ac93b761"},
        "records": records, "historical_sources_modified": False,
        "old_ledgers_or_scopes_modified": False, "task_episode_started_for_validation": False,
    }
    target = repo / "evaluations/multitask_parallel_v22/dual_tool/server_shared/REMOTE_REPAIR_MANIFEST.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(target), "sha256": digest(target), "records": len(records)}))


if __name__ == "__main__":
    main()
