#!/usr/bin/env python3
"""Export and launch hash-bound, mixed-version Twin RSI task packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


SCHEMA = "graph-self-adapter.twin-final-selection.v1"
EXPORT_SCHEMA = "graph-self-adapter.twin-package-export.v1"
REQUIRED_SNAPSHOT_CODE = (
    "prove/task_graph/twin_graph.py",
    "prove/task_graph/twin_physics.py",
    "prove/fast_adapter/twin_local_skill.py",
    "tools/task_graph/twin_rsi.py",
    "tools/task_graph/twin_train_skills.py",
)
HEX = frozenset("0123456789abcdef")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require_hash(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in HEX for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def resolve_input(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def source_snapshot_path(value: Any, base: Path) -> tuple[Path, str | None]:
    if isinstance(value, str):
        return resolve_input(value, base), None
    if isinstance(value, Mapping) and isinstance(value.get("path"), str):
        expected = value.get("sha256")
        return resolve_input(value["path"], base), (
            None if expected is None else require_hash(expected, "source_snapshot.sha256"))
    raise ValueError("source_snapshot must be a path string or {path, sha256?}")


def solution_path(value: Any, task: str, base: Path) -> Path:
    raw = value.get("path") if isinstance(value, Mapping) else value
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{task}: packages must be a nonempty path or path binding")
    path = resolve_input(raw, base)
    candidates = (path, path / "solution", path / f"twin_{task}" / "solution")
    matches = [candidate for candidate in candidates
               if (candidate / "MANIFEST.json").is_file()]
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise ValueError(f"{task}: packages path does not identify one solution: {path}")
    if isinstance(value, Mapping) and value.get("sha256") is not None:
        expected = require_hash(value["sha256"], f"{task}.packages.sha256")
        if sha256(unique[0] / "MANIFEST.json") != expected:
            raise ValueError(f"{task}: package MANIFEST hash mismatch")
    return unique[0]


def verify_snapshot(snapshot: Path, expected_freeze: str | None = None) -> tuple[dict[str, str], str]:
    freeze_path = snapshot / "SOURCE_FREEZE.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"source snapshot freeze missing: {freeze_path}")
    freeze_sha = sha256(freeze_path)
    if expected_freeze is not None and freeze_sha != expected_freeze:
        raise ValueError(f"source snapshot freeze hash mismatch: {snapshot}")
    freeze = load_json(freeze_path)
    if not isinstance(freeze, Mapping):
        raise ValueError(f"SOURCE_FREEZE must be an object: {freeze_path}")
    for relative in REQUIRED_SNAPSHOT_CODE:
        expected = require_hash(freeze.get(relative), f"SOURCE_FREEZE[{relative}]")
        path = snapshot / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source snapshot code mismatch: {path}")
    return {str(k): str(v) for k, v in freeze.items()}, freeze_sha


def verify_solution(row: Mapping[str, Any], solution: Path,
                    snapshot_freeze: Mapping[str, str]) -> dict[str, Any]:
    task, candidate = str(row["task"]), str(row["candidate_id"])
    files = {
        "graph": solution / "graph/task_graph.json",
        "model": solution / "models/local_skill.npz",
        "reference": solution / "refs/visible_centroids.json",
        "candidate_freeze": solution / "CANDIDATE_FREEZE.json",
        "manifest": solution / "MANIFEST.json",
        "external_bindings": solution / "EXTERNAL_BINDINGS.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{task}: package files missing: {missing}")
    expected = {
        "graph": require_hash(row.get("graph_sha256"), f"{task}.graph_sha256"),
        "model": require_hash(row.get("model_sha256"), f"{task}.model_sha256"),
        "reference": require_hash(row.get("reference_sha256"), f"{task}.reference_sha256"),
        "candidate_freeze": require_hash(
            row.get("candidate_freeze_sha256"), f"{task}.candidate_freeze_sha256"),
    }
    actual = {name: sha256(path) for name, path in files.items()}
    for name, digest in expected.items():
        if actual[name] != digest:
            raise ValueError(f"{task}: selected {name} hash mismatch")
    manifest, freeze = load_json(files["manifest"]), load_json(files["candidate_freeze"])
    for document, label in ((manifest, "MANIFEST"), (freeze, "CANDIDATE_FREEZE")):
        if document.get("task") != task or document.get("candidate_id") != candidate:
            raise ValueError(f"{task}: {label} task/candidate mismatch")
    cross = {
        "graph_sha256": expected["graph"],
        "references_sha256": expected["reference"],
    }
    for key, digest in cross.items():
        if manifest.get(key) != digest or freeze.get(key) != digest:
            raise ValueError(f"{task}: {key} disagrees across selection/package freezes")
    if freeze.get("model_sha256") != expected["model"]:
        raise ValueError(f"{task}: model hash disagrees with CANDIDATE_FREEZE")
    if manifest.get("model", {}).get("sha256") != expected["model"]:
        raise ValueError(f"{task}: model hash disagrees with MANIFEST")
    if manifest.get("candidate_freeze", {}).get("sha256") != expected["candidate_freeze"]:
        raise ValueError(f"{task}: candidate-freeze hash disagrees with MANIFEST")
    code = freeze.get("code", {})
    for relative in REQUIRED_SNAPSHOT_CODE[:-1]:
        if code.get(relative) != snapshot_freeze[relative]:
            raise ValueError(
                f"{task}: candidate code disagrees with selected source snapshot: {relative}")
    for relative, digest in code.items():
        if relative in snapshot_freeze and snapshot_freeze[relative] != digest:
            raise ValueError(f"{task}: candidate code disagrees with selected source snapshot: {relative}")
    return {**actual, "task": task, "candidate_id": candidate}


def verify_hash_refs(value: Any, base: Path, label: str) -> int:
    if value is None:
        return 0
    rows = value if isinstance(value, list) else [value]
    count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError(f"{label}[{index}] must be a path+sha256 binding")
        path = resolve_input(row["path"], base)
        expected = require_hash(row.get("sha256"), f"{label}[{index}].sha256")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{label}[{index}] hash mismatch: {path}")
        count += 1
    return count


def validate_selection(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection, base = load_json(path), path.resolve().parent
    if selection.get("schema") != SCHEMA or selection.get("status") != "frozen":
        raise ValueError("selection must have final-selection.v1 schema and frozen status")
    if selection.get("candidate_id") != "per_task_selected_R0_to_R3":
        raise ValueError("selection candidate_id must identify per-task R0-to-R3 selection")
    rows = selection.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection tasks must be a nonempty list")
    tasks = [str(row.get("task")) for row in rows if isinstance(row, Mapping)]
    if len(tasks) != len(rows) or len(set(tasks)) != len(tasks):
        raise ValueError("selection tasks must contain unique task names")
    arms = selection.get("arms")
    if arms != ["fixed_z", "online_z"] and set(arms or ()) != {"fixed_z", "online_z"}:
        raise ValueError("selection arms must contain fixed_z and online_z")
    conditions = selection.get("conditions", {})
    if (conditions.get("test_seed") != 71005 or
            not isinstance(conditions.get("trials_per_task_per_arm"), int) or
            conditions["trials_per_task_per_arm"] <= 0):
        raise ValueError("selection conditions require test_seed=71005 and positive trial count")
    verify_hash_refs(selection.get("development_evidence", []), base, "development_evidence")
    verified: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row.get("candidate_id"), str) or not row["candidate_id"]:
            raise ValueError(f"{row.get('task')}: candidate_id must be nonempty")
        snapshot, expected_snapshot = source_snapshot_path(row.get("source_snapshot"), base)
        snapshot_freeze, freeze_sha = verify_snapshot(snapshot, expected_snapshot)
        solution = solution_path(row.get("packages"), str(row["task"]), base)
        hashes = verify_solution(row, solution, snapshot_freeze)
        verified.append({"row": dict(row), "solution": solution, "snapshot": snapshot,
                         "source_freeze_sha256": freeze_sha, "hashes": hashes})
    return selection, verified


def selected_solution_files(solution: Path) -> list[Path]:
    files = [solution / name for name in (
        "MANIFEST.json", "CANDIDATE_FREEZE.json", "EXTERNAL_BINDINGS.json")]
    for directory in ("graph", "refs", "models", "contracts", "adapter"):
        root = solution / directory
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    files.extend(path for path in solution.glob("*.json") if path.is_file())
    return sorted(set(files))


def snapshot_files(snapshot: Path) -> list[Path]:
    return sorted(path for path in snapshot.rglob("*") if path.is_file() and
                  "__pycache__" not in path.parts and path.suffix != ".pyc")


def relative_binding(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}


def export(selection_path: Path, output: Path, *, dry_run: bool) -> dict[str, Any]:
    selection, rows = validate_selection(selection_path.resolve())
    output = output.resolve()
    snapshot_names: dict[str, str] = {}
    for item in rows:
        snapshot_names.setdefault(item["source_freeze_sha256"],
                                  item["source_freeze_sha256"][:16])
    plan = {
        "schema": EXPORT_SCHEMA,
        "dry_run": dry_run,
        "output": str(output),
        "tasks": [{"task": item["row"]["task"],
                   "candidate_id": item["row"]["candidate_id"],
                   "snapshot": snapshot_names[item["source_freeze_sha256"]]}
                  for item in rows],
        "source_snapshots": len(snapshot_names),
    }
    if dry_run:
        return plan
    if output.exists():
        raise FileExistsError(f"export output already exists: {output}")
    output.mkdir(parents=True)
    provenance = output / "provenance/INPUT_SELECTION.json"
    provenance.parent.mkdir(parents=True)
    shutil.copy2(selection_path, provenance)
    normalized = json.loads(json.dumps(selection))
    for index, binding in enumerate(normalized.get("development_evidence", [])):
        source = resolve_input(binding["path"], selection_path.resolve().parent)
        target = output / "provenance/development_evidence" / f"{index:03d}_{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        binding["path"] = target.relative_to(output).as_posix()
    normalized_rows = {row["task"]: row for row in normalized["tasks"]}
    copied_snapshots: set[str] = set()
    for item in rows:
        task = item["row"]["task"]
        snapshot_name = snapshot_names[item["source_freeze_sha256"]]
        snapshot_target = output / "source_snapshots" / snapshot_name
        if snapshot_name not in copied_snapshots:
            for source in snapshot_files(item["snapshot"]):
                target = snapshot_target / source.relative_to(item["snapshot"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            copied_snapshots.add(snapshot_name)
        solution_target = output / "tasks" / f"twin_{task}" / "solution"
        for source in selected_solution_files(item["solution"]):
            target = solution_target / source.relative_to(item["solution"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        normalized_rows[task]["packages"] = f"tasks/twin_{task}/solution"
        normalized_rows[task]["source_snapshot"] = f"source_snapshots/{snapshot_name}"
    normalized_path = output / "FINAL_SELECTION.json"
    dump_json(normalized_path, normalized)
    launcher_tool = output / "tools/task_graph/twin_export_packages.py"
    launcher_tool.parent.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), launcher_tool)
    run_sh = output / "run.sh"
    run_sh.write_text("""#!/usr/bin/env bash
set -euo pipefail
export_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "${PYTHON:-python3}" "$export_root/tools/task_graph/twin_export_packages.py" launch --export-root "$export_root" "$@"
""")
    run_sh.chmod(0o755)
    readme = output / "README.md"
    readme.write_text("""# Twin RSI selected task packages

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
""")
    bound = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and
                       p.name != "EXPORT_MANIFEST.json"):
        bound.append(relative_binding(path, output))
    manifest = {**plan, "dry_run": False,
                "input_selection": relative_binding(provenance, output),
                "normalized_selection": relative_binding(normalized_path, output),
                "files": bound,
                "omitted": ["training datasets", "MuJoCo scenes and meshes",
                            "demonstrations", "Python interpreters and environments"]}
    dump_json(output / "EXPORT_MANIFEST.json", manifest)
    return manifest


def export_manifest(export_root: Path) -> dict[str, Any]:
    manifest_path = export_root / "EXPORT_MANIFEST.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError("wrong export manifest schema")
    for row in manifest.get("files", []):
        path = export_root / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"exported file changed or missing: {path}")
    return manifest


def verify_external_runtime(snapshot: Path, solution: Path, task: str) -> int:
    """Hash-check external runtime inputs without importing or stepping MuJoCo."""
    bindings = load_json(solution / "EXTERNAL_BINDINGS.json")
    sources = bindings.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError(f"{task}: EXTERNAL_BINDINGS sources must be nonempty")
    checked = 0
    for label, binding in sources.items():
        if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
            raise ValueError(f"{task}: invalid external source binding {label}")
        path = Path(binding["path"]).expanduser()
        expected = require_hash(binding.get("sha256"), f"{task}.sources.{label}.sha256")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{task}: external source changed or missing: {path}")
        checked += 1
    catalog = load_json(snapshot / "evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json")
    task_assets = catalog.get("task_assets", {}).get(task)
    if not isinstance(task_assets, Mapping):
        raise ValueError(f"{task}: external asset catalog entry missing")
    for raw_path, digest in task_assets.get("external_asset_hashes", {}).items():
        path = Path(raw_path).expanduser()
        expected = require_hash(digest, f"{task}.external_asset_hashes[{raw_path}]")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{task}: external scene asset changed or missing: {path}")
        checked += 1
    return checked


def launch(args: argparse.Namespace) -> dict[str, Any]:
    root = args.export_root.resolve()
    manifest = export_manifest(root)
    selection = load_json(root / "FINAL_SELECTION.json")
    matches = [row for row in selection["tasks"] if row["task"] == args.task]
    if len(matches) != 1:
        raise ValueError(f"task is not uniquely selected: {args.task}")
    row = matches[0]
    snapshot = root / row["source_snapshot"]
    solution = root / row["packages"]
    freeze, _ = verify_snapshot(snapshot)
    verify_solution(row, solution, freeze)
    conditions = selection["conditions"]
    trial_count = int(conditions["trials_per_task_per_arm"])
    if not 0 <= args.trial < trial_count:
        raise ValueError(f"trial must be in [0, {trial_count})")
    external_files = verify_external_runtime(snapshot, solution, args.task)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    python = args.python or os.environ.get("TWIN_PYTHON") or "/home/zheng/github/Mujoco-RSI/.venv/bin/python"
    command = [python, str(snapshot / "tools/task_graph/twin_rsi.py"),
               "--plan", str(snapshot / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json"),
               "--assets", str(snapshot / "evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json"),
               "--packages", str(root / "tasks"), "run", "--task", args.task,
               "--condition-set", args.condition_set, "--trial", str(args.trial),
               "--candidate-id", row["candidate_id"], "--output", str(output)]
    if args.online_z:
        command.append("--online-z")
    if args.condition_set == "test":
        command.extend(("--selection-freeze", str(root / "FINAL_SELECTION.json")))
    report = {"schema": "graph-self-adapter.twin-export-launch.v1", "dry_run": args.dry_run,
              "task": args.task, "candidate_id": row["candidate_id"],
              "source_snapshot": row["source_snapshot"], "output": str(output),
              "command": command, "export_files_verified": len(manifest["files"]),
              "external_runtime_files_verified": external_files}
    if args.dry_run:
        return report
    completed = subprocess.run(command, check=False)
    report["returncode"] = completed.returncode
    if completed.returncode:
        raise RuntimeError(f"frozen task run failed with status {completed.returncode}")
    return report


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("export")
    q.add_argument("--selection", type=Path, required=True)
    q.add_argument("--output", type=Path, default=root / "task_packages/twin_rsi_v01")
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("verify")
    q.add_argument("--export-root", type=Path, required=True)
    q = sub.add_parser("launch")
    q.add_argument("--export-root", type=Path, required=True)
    q.add_argument("--task", required=True)
    q.add_argument("--condition-set", choices=("development", "test"), default="development")
    q.add_argument("--trial", type=int, default=0)
    q.add_argument("--online-z", action="store_true")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--python")
    q.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "export":
        report = export(args.selection, args.output, dry_run=args.dry_run)
    elif args.command == "verify":
        report = export_manifest(args.export_root.resolve())
    else:
        report = launch(args)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
