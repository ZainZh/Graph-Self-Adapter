import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools/task_graph/twin_export_packages.py"
SPEC = importlib.util.spec_from_file_location("_twin_export_packages_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> Path:
    snapshot = tmp_path / "source"
    external = tmp_path / "external/scene.xml"
    external.parent.mkdir(parents=True); external.write_text("<mujoco/>\n")
    source_hashes = {}
    for relative in EXPORT.REQUIRED_SNAPSHOT_CODE:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen {relative}\n")
        source_hashes[relative] = _sha(path)
    _json(snapshot / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json", {"tasks": []})
    _json(snapshot / "evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json", {
        "task_assets": {"collect_apple": {"external_asset_hashes": {}}}})
    _json(snapshot / "evaluations/twin_rsi_v01/SCOPE.json", {"frozen": True})
    _json(snapshot / "SOURCE_FREEZE.json", source_hashes)

    solution = tmp_path / "packages/twin_collect_apple/solution"
    graph = solution / "graph/task_graph.json"
    model = solution / "models/local_skill.npz"
    refs = solution / "refs/visible_centroids.json"
    graph.parent.mkdir(parents=True); graph.write_text("{}\n")
    model.parent.mkdir(parents=True); model.write_bytes(b"offline-model")
    refs.parent.mkdir(parents=True); refs.write_text("{}\n")
    (solution / "training").mkdir(parents=True)
    (solution / "training/local_skill_dataset.npz").write_bytes(b"not-exported")
    freeze = {
        "schema": "graph-self-adapter.twin-candidate-freeze.v1",
        "task": "collect_apple", "candidate_id": "r2_final",
        "graph_sha256": _sha(graph), "model_sha256": _sha(model),
        "references_sha256": _sha(refs),
        "code": {key: source_hashes[key] for key in source_hashes if key != "tools/task_graph/twin_train_skills.py"},
    }
    _json(solution / "CANDIDATE_FREEZE.json", freeze)
    manifest = {
        "schema": "graph-self-adapter.twin-solution.v1",
        "task": "collect_apple", "candidate_id": "r2_final",
        "graph_sha256": _sha(graph), "references_sha256": _sha(refs),
        "model": {"path": "models/local_skill.npz", "sha256": _sha(model)},
        "candidate_freeze": {"path": "CANDIDATE_FREEZE.json",
                             "sha256": _sha(solution / "CANDIDATE_FREEZE.json")},
    }
    _json(solution / "MANIFEST.json", manifest)
    _json(solution / "EXTERNAL_BINDINGS.json", {"sources": {
        "scene.xml": {"path": str(external), "sha256": _sha(external)}}})
    evidence = tmp_path / "development.json"; evidence.write_text("{}\n")
    selection = {
        "schema": EXPORT.SCHEMA, "status": "frozen",
        "candidate_id": "per_task_selected_R0_to_R3",
        "tasks": [{
            "task": "collect_apple", "candidate_id": "r2_final",
            "packages": "packages", "source_snapshot": "source",
            "graph_sha256": _sha(graph), "model_sha256": _sha(model),
            "reference_sha256": _sha(refs),
            "candidate_freeze_sha256": _sha(solution / "CANDIDATE_FREEZE.json"),
        }],
        "development_evidence": [{"path": "development.json", "sha256": _sha(evidence)}],
        "conditions": {"test_seed": 71005, "trials_per_task_per_arm": 10},
        "arms": ["fixed_z", "online_z"],
    }
    path = tmp_path / "FINAL_SELECTION_INPUT.json"
    _json(path, selection)
    return path


def test_export_is_portable_hash_bound_and_launch_dry_run_only(tmp_path):
    selection = _fixture(tmp_path)
    output = tmp_path / "exported"
    plan = EXPORT.export(selection, output, dry_run=True)
    assert plan["dry_run"] and not output.exists()

    report = EXPORT.export(selection, output, dry_run=False)
    assert report["source_snapshots"] == 1
    normalized = json.loads((output / "FINAL_SELECTION.json").read_text())
    row = normalized["tasks"][0]
    assert row["packages"] == "tasks/twin_collect_apple/solution"
    assert row["source_snapshot"].startswith("source_snapshots/")
    assert not (output / row["packages"] / "training").exists()
    assert normalized["development_evidence"][0]["path"].startswith(
        "provenance/development_evidence/")
    assert (output / row["source_snapshot"] / "evaluations/twin_rsi_v01/SCOPE.json").is_file()
    EXPORT.export_manifest(output)

    args = EXPORT.parser().parse_args([
        "launch", "--export-root", str(output), "--task", "collect_apple",
        "--condition-set", "test", "--trial", "3", "--online-z",
        "--output", str(tmp_path / "new-results"), "--dry-run",
    ])
    launch = EXPORT.launch(args)
    assert launch["dry_run"] and not (tmp_path / "new-results").exists()
    assert launch["external_runtime_files_verified"] == 1
    assert str(output / row["source_snapshot"] / "tools/task_graph/twin_rsi.py") in launch["command"]
    assert "--selection-freeze" in launch["command"]


def test_export_rejects_hash_change_and_unknown_symbolic_package(tmp_path):
    selection = _fixture(tmp_path)
    data = json.loads(selection.read_text())
    data["tasks"][0]["graph_sha256"] = "0" * 64
    _json(selection, data)
    with pytest.raises(ValueError, match="selected graph hash mismatch"):
        EXPORT.export(selection, tmp_path / "bad", dry_run=True)
