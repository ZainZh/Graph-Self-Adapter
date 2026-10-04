#!/usr/bin/env python3
"""Build and verify the standalone Graph Self Adapter source inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = REPO_ROOT / "migration" / "SOURCE_INVENTORY.json"
CODE_ROOTS = (
    "prove/task_graph",
    "prove/fast_adapter",
    "tools/task_graph",
    "tools/fast_adapter",
)
SUPPORT_FILES = (
    "prove/__init__.py",
    "prove/observation/__init__.py",
    "prove/observation/agent_semantic.py",
    "prove/observation/catalog.py",
    "prove/observation/contracts.py",
    "prove/observation/event_trigger.py",
    "prove/observation/observer.py",
    "prove/observation/phase_state.py",
    "prove/observation/primitive.py",
    "prove/observation/primitives/__init__.py",
    "prove/observation/primitives/belief.py",
    "prove/observation/primitives/container_geometry.py",
    "prove/observation/primitives/features.py",
    "prove/observation/primitives/geometry.py",
    "prove/observation/primitives/relation.py",
    "prove/observation/primitives/robot_state.py",
    "prove/observation/primitives/temporal.py",
    "prove/observation/retained_object_observer.py",
    "prove/observation/tool_index.py",
    "tools/__init__.py",
)
MODULES = (
    "prove",
    "prove.fast_adapter.p68_book_runtime",
    "prove.task_graph.p70_object_controller",
    "tools.task_graph.p70_stage_runner",
)
TEST_FIXTURES = (
    "evaluations/fast_adapter_p68_cross_task_reference_v01/INSERT_SCOPE.json",
    "evaluations/outer_rsi_p69_book_v01/r2_book_silhouette_v01/BOOK_REFERENCE.json",
    "evaluations/outer_rsi_p69_book_v01/r2_grounding_reference_v01/GROUNDING_REFERENCE_R2_FINAL.json",
    "evaluations/outer_rsi_p69_book_v01/r2_grounding_reference_v01/demo_initial/private/geometry_0000.json",
    "evaluations/outer_rsi_p69_book_v01/r2_grounding_reference_v01/demo_initial/public/frame_0000.npz",
    "evaluations/outer_rsi_p70_object_graph_v01/P70_CONTROLLER_CONFIG_DRAFT.json",
    "evaluations/outer_rsi_p70_object_graph_v01/P70_OBJECT_RELATION_TARGETS_v02.json",
    "evaluations/outer_rsi_p70_object_graph_v01/training/grasp_seed70010_v02/DEMO_RELATIONS.json",
)
MODIFIED_WRAPPER_FILES = (
    "task_packages/book_to_caddy_r3/task_bundle.py",
    "task_packages/book_to_caddy_r3/solution/DEPENDENCIES.json",
    "task_packages/book_to_caddy_r3/solution/MANIFEST.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_paths(root: Path) -> list[Path]:
    paths = {p.relative_to(root) for name in CODE_ROOTS for p in (root / name).rglob("*.py")}
    paths.update(Path(name) for name in SUPPORT_FILES)
    return sorted(paths)


def frozen_sources(root: Path) -> list[dict[str, object]]:
    freeze_path = root / "task_packages/book_to_caddy_r3/solution/provenance/PHYSICAL_FREEZE_R3.json"
    freeze = json.loads(freeze_path.read_text())
    rows = []
    for original, expected in sorted(freeze["sources"].items()):
        rel = original.split("/versions/R3/", 1)[1]
        current = root / rel
        rows.append({
            "original_path": original,
            "new_path": rel,
            "sha256": expected,
            "new_sha256": sha256(current),
            "bytes": current.stat().st_size,
        })
    return rows


def build_inventory(original_root: Path, new_root: Path) -> dict[str, object]:
    files = []
    for rel in canonical_paths(new_root):
        original = original_root / rel
        current = new_root / rel
        files.append({
            "original_path": str(original),
            "new_path": rel.as_posix(),
            "sha256": sha256(original),
            "new_sha256": sha256(current),
            "bytes": current.stat().st_size,
            "identical": sha256(original) == sha256(current),
        })
    fixtures = []
    for name in TEST_FIXTURES:
        original = original_root / name
        current = new_root / name
        fixtures.append({
            "original_path": str(original), "new_path": name,
            "sha256": sha256(original), "new_sha256": sha256(current),
            "bytes": current.stat().st_size,
        })
    modified = []
    for name in MODIFIED_WRAPPER_FILES:
        original = original_root / name
        current = new_root / name
        modified.append({
            "original_path": str(original), "new_path": name,
            "original_sha256": sha256(original), "new_sha256": sha256(current),
            "bytes": current.stat().st_size,
        })
    return {
        "schema": "graph-self-adapter.source-inventory.v1",
        "source_repository": str(original_root.resolve()),
        "destination_repository": str(new_root.resolve()),
        "canonical_python_file_count": len(files),
        "canonical_python_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
        "frozen_runtime_sources": frozen_sources(new_root),
        "test_fixtures": fixtures,
        "intentional_modified_task_wrapper": modified,
        "root_review_mutable_files": [
            {
                "original_path": str(original_root / "task_packages/book_to_caddy_r3/solution/RELEASE.json"),
                "new_path": "task_packages/book_to_caddy_r3/solution/RELEASE.json",
                "purpose": "Root records the final Graph-Self-Adapter migration review after implementation freezes"
            }
        ],
    }


def verify(root: Path, inventory_path: Path) -> list[str]:
    errors: list[str] = []
    inventory = json.loads(inventory_path.read_text())
    rows = inventory["files"]
    if len(rows) != 748:
        errors.append(f"inventory Python count is {len(rows)}, expected 748")
    if sum(int(row["bytes"]) for row in rows) != 6196519:
        errors.append("inventory Python byte total differs from 6,196,519")
    for row in rows:
        path = root / row["new_path"]
        if not path.is_file():
            errors.append(f"missing source: {row['new_path']}")
        elif sha256(path) != row["sha256"]:
            errors.append(f"source hash mismatch: {row['new_path']}")
    for row in inventory["frozen_runtime_sources"]:
        path = root / row["new_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            errors.append(f"frozen runtime mismatch: {row['new_path']}")
    for row in inventory.get("test_fixtures", []):
        path = root / row["new_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            errors.append(f"test fixture mismatch: {row['new_path']}")
    for row in inventory.get("intentional_modified_task_wrapper", []):
        path = root / row["new_path"]
        if not path.is_file() or sha256(path) != row["new_sha256"]:
            errors.append(f"modified wrapper inventory stale: {row['new_path']}")
    root_resolved = root.resolve()
    dependency_file = root / "task_packages/book_to_caddy_r3/solution/DEPENDENCIES.json"
    dependencies = json.loads(dependency_file.read_text())
    local_file = root / "local/DEPENDENCIES.local.json"
    if local_file.is_file():
        dependencies.update(json.loads(local_file.read_text()))
    python = dependencies.get("python") or dependencies.get("python_default") or sys.executable
    overlay = dependencies.get("runtime_overlay") or dependencies.get("runtime_overlay_default")
    script = (
        "import importlib,json,sys; "
        "print(json.dumps({n:importlib.import_module(n).__file__ for n in sys.argv[1:]}))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (overlay, str(root_resolved))))
    completed = subprocess.run(
        [python, "-c", script, *MODULES], env=env, cwd=root_resolved,
        text=True, capture_output=True,
    )
    if completed.returncode:
        errors.append(f"module-origin subprocess failed: {completed.stderr.strip()}")
    else:
        for name, value in json.loads(completed.stdout).items():
            origin = Path(value).resolve()
            if not origin.is_relative_to(root_resolved):
                errors.append(f"module {name} resolved outside repository: {origin}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--write-inventory", action="store_true")
    parser.add_argument("--original-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = args.inventory.resolve()
    if args.write_inventory:
        if args.original_root is None:
            parser.error("--write-inventory requires --original-root")
        data = build_inventory(args.original_root.resolve(), root)
        inventory.parent.mkdir(parents=True, exist_ok=True)
        inventory.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    errors = verify(root, inventory)
    print(json.dumps({"ok": not errors, "errors": errors, "inventory": str(inventory)}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
