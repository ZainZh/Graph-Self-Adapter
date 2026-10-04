#!/usr/bin/env python3
"""Lightweight P70 R3 task bundle.

This successor package stores only task-specific graph/config/model artifacts.
Standard LIBERO assets, DINO/LiLa pretrained assets, Python envs and vendor
packages are external installed dependencies. Runtime Python code always comes
from this repository; machine-specific assets can be bound with --dependencies.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SOLUTION = PACKAGE_DIR / "solution"
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_DEPENDENCIES = REPO_ROOT / "local" / "DEPENDENCIES.local.json"
SOURCE_FREEZE = Path("evaluations/outer_rsi_p70_object_graph_v01/final_v01/R3/PHYSICAL_FREEZE.json")
SOURCE_REVIEW = Path("evaluations/outer_rsi_p70_object_graph_v01/final_v01/R3/ROOT_PHYSICAL_REVIEW.json")
SOURCE_FREEZE_SHA = "6fc7e6fa79b1847213b2ba1eac8d9f7f940a7df02d58be89102164e2ff99f446"
SOURCE_REVIEW_SHA = "e996e9971d5f6038cda1c13a93a8b35f0aaf7db7884e2288562ce585eb4058b7"
DEFAULT_CELL = {"index": 7, "init_id": 7, "execution_seed": 70037, "state_sha256": "2f359eaf0499d91a9e34f60059daf9c65b871bdc754a2860f66f07dfd1597fa8"}

REMOTE_FALLBACKS = {
    "evaluations/fast_adapter_p55_graph_approach_v01/BINDING_v01.json": "evaluations/fast_adapter_p55_graph_approach_v01/remote_results_v01/BINDING_v01.json",
    "evaluations/fast_adapter_p54_book_scale_v01/BINDING_v01.json": "evaluations/fast_adapter_p54_book_scale_v01/remote_results_v01/BINDING_v01.json",
    "evaluations/fast_adapter_p53_micro_insert_v01/full_successor_v01/FULL_TASK_FREEZE.json": "evaluations/fast_adapter_p53_micro_insert_v01/remote_results_v01/full_successor_v01/FULL_TASK_FREEZE.json",
    "evaluations/fast_adapter_p53_micro_insert_v01/CODE_MICRO_FREEZE_v03.json": "task_packages/book_to_caddy_r3/staging/P53_CODE_MICRO_FREEZE_v03.json",
    "evaluations/fast_adapter_p68_cross_task_reference_v01/book/evaluation_v01/DINO_BINDING.json": "task_packages/book_to_caddy_r3/staging/SERVER_DINO_BINDING.json",
}

TASK_FILE_KEYS = {
    "a2_head", "action_checkpoint", "encoder_checkpoint", "precision_head", "priors"
}

EXPECTED_SOLVER_ROUTING = {
    "approach_book": "Code above_current_book -> frozen grasp MLP pregrasp",
    "short_grasp": "frozen grasp MLP direct_close",
    "lift_clear": "Code: P70ObjectController",
    "orient_clear": "Code: P70ObjectController",
    "transport_above_caddy": "Code: P70ObjectController",
    "near_entry": "Code: P70ObjectController",
    "p68_insert": "Fast Adaptive Skill: p69_full_rsi_runner.run_s9_book -> p68_book_runtime.ReferenceBookRuntime (continuous z, H24)",
    "verify_placement": "Code: p70_stage_runner.run_placement_verification (stationary closed hold, max3)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def rel_from_any(path_text: str) -> str | None:
    p = str(path_text).replace("\\", "/")
    root = str(REPO_ROOT).replace("\\", "/")
    if p.startswith(root + "/"):
        return p[len(root) + 1:]
    if "/versions/R3/" in p:
        return p.split("/versions/R3/", 1)[1]
    for anchor in ("evaluations/", "prove/", "tools/", "data/", "task_packages/"):
        if anchor in p:
            return p[p.index(anchor):]
    return None


def audit_assets() -> list[dict[str, Any]]:
    p = PACKAGE_DIR / "ASSET_AUDIT.json"
    return load_json(p).get("assets", []) if p.exists() else []


def resolve_path(repo_root: Path, path_text: str, expected_sha: str | None = None) -> Path:
    candidates = []
    raw = Path(path_text)
    if raw.exists():
        candidates.append(raw)
    rel = rel_from_any(path_text)
    if rel:
        candidates.append(repo_root / rel)
        if rel in REMOTE_FALLBACKS:
            candidates.append(repo_root / REMOTE_FALLBACKS[rel])
    for a in audit_assets():
        if expected_sha and a.get("sha256") == expected_sha:
            candidates += [Path(x) for x in a.get("local_matches", []) + a.get("local_candidates", [])]
    for c in candidates:
        if c.exists() and (expected_sha is None or sha256(c) == expected_sha):
            return c.resolve()
    raise FileNotFoundError(f"cannot resolve {path_text!r} sha={expected_sha}")


def copy_bound(src: Path, dst: Path, expected_sha: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha256(dst) != expected_sha:
        raise ValueError(f"hash mismatch after copy: {dst}")


def solution_manifest(solution: Path) -> dict[str, Any]:
    return load_json(solution / "MANIFEST.json")


def write_solution(repo_root: Path, solution: Path, force: bool = False) -> dict[str, Any]:
    if solution.exists():
        if not force:
            raise FileExistsError(f"solution exists: {solution}")
        shutil.rmtree(solution)
    solution.mkdir(parents=True)
    freeze_path = repo_root / SOURCE_FREEZE
    review_path = repo_root / SOURCE_REVIEW
    if sha256(freeze_path) != SOURCE_FREEZE_SHA or sha256(review_path) != SOURCE_REVIEW_SHA:
        raise ValueError("source freeze/review hash mismatch")
    freeze = load_json(freeze_path)
    controller_config_src = resolve_path(repo_root, freeze["controller_config"]["path"], freeze["controller_config"]["sha256"])
    controller_config = load_json(controller_config_src)

    files: dict[str, dict[str, Any]] = {}
    def add(label: str, src_path: str, digest: str, dst_rel: str) -> None:
        src = resolve_path(repo_root, src_path, digest)
        copy_bound(src, solution / dst_rel, digest)
        files[label] = {"path": dst_rel, "sha256": digest, "bytes": (solution / dst_rel).stat().st_size}

    # Task-specific model/config/reference artifacts.
    for key in TASK_FILE_KEYS:
        item = freeze["files"][key]
        suffix = Path(rel_from_any(item["path"]) or Path(item["path"]).name).name
        add(f"files.{key}", item["path"], item["sha256"], f"models/{key}/{suffix}")
    add("files.dino_binding", freeze["files"]["dino_binding"]["path"], freeze["files"]["dino_binding"]["sha256"], "configs/SERVER_DINO_BINDING.json")
    add("controller_config", freeze["controller_config"]["path"], freeze["controller_config"]["sha256"], "configs/P70_CONTROLLER_CONFIG_R3.json")
    for label, item in freeze["controller_file_bindings"].items():
        add(f"controller_file_bindings.{label}", item["path"], item["sha256"], "refs/" + label.replace(".", "/") + "/" + Path(item["path"]).name)

    # Canonical provenance inputs.  Kept as original bytes.
    add("source_freeze", str(SOURCE_FREEZE), SOURCE_FREEZE_SHA, "provenance/PHYSICAL_FREEZE_R3.json")
    add("source_review", str(SOURCE_REVIEW), SOURCE_REVIEW_SHA, "provenance/ROOT_PHYSICAL_REVIEW_R3.json")
    add("p69_freeze", freeze["p69_freeze"]["path"], freeze["p69_freeze"]["sha256"], "provenance/P69_PHYSICAL_FREEZE_R2.json")
    add("init_scope", freeze["init_scope"]["path"], freeze["init_scope"]["sha256"], "provenance/INIT_SCOPE.json")
    add("p55_binding", freeze["p55_binding"]["path"], freeze["p55_binding"]["sha256"], "provenance/parent_chain/P55_BINDING.json")
    for i, item in enumerate(freeze["parent_binding_chain"]):
        add(f"parent_binding_chain.{i}", item["path"], item["sha256"], f"provenance/parent_chain/{i:02d}_{Path(item['path']).name}")

    graph = {
        "schema": "prove.book-to-caddy-r3.task-graph.v1",
        "source": "P70_CONTROLLER_CONFIG_R3.stage_ids plus frozen P70 runner routing",
        "active_controller_module": "prove.task_graph.p70_object_controller.P70ObjectController",
        "runner": "tools/task_graph/p70_stage_runner.py",
        "budgets": freeze["budgets"],
        "stage_ids": controller_config.get("stage_ids", []),
        "solver_routing": EXPECTED_SOLVER_ROUTING,
        "controller_config_binding": files["controller_config"],
        "claims": controller_config.get("claims", []),
    }
    dump_json(solution / "graph/task_graph.json", graph)
    files["graph.task_graph"] = {"path": "graph/task_graph.json", "sha256": sha256(solution / "graph/task_graph.json"), "bytes": (solution / "graph/task_graph.json").stat().st_size}

    deps = {
        "schema": "prove.book-to-caddy-r3.dependencies.v1",
        "external_requirements": ["installed Interaction-RSI repository", "matching Python/LIBERO runtime", "DINOv3 model directory", "LiLa source and VISUAL_WEIGHTS", "standard LIBERO/robosuite assets"],
        "repo_root_default": str(repo_root.resolve()),
        "python_default": load_json(PACKAGE_DIR / "ASSET_AUDIT.json").get("local_environment", {}).get("python") if (PACKAGE_DIR / "ASSET_AUDIT.json").exists() else sys.executable,
        "dino_model_dir_default": "/home/zheng/github/LiLa-WAM/dinov3_pretrain/dinov3-vitl16-pretrain-lvd1689m",
        "lila_source_default": str((repo_root / "data/sources/lila-wam-reference-20260923/models/vla_model_fm.py").resolve()),
        "lila_checkpoint_default": str((repo_root / "data/sources/lila-wam-reference-20260923/VISUAL_WEIGHTS.pt").resolve()),
        "libero_root_default": str((repo_root / "data/sources/LIBERO-PRO/libero").resolve()),
        "runtime_overlay_default": str((repo_root / "data/runtime/p58-vision-deps").resolve()),
    }
    # Key external DINO files remain outside solution; bind known hashes when ASSET_AUDIT is available.
    dino_hashes = {}
    for a in audit_assets():
        if a.get("category") == "dino_model":
            dino_hashes[Path(a["path"]).name] = a.get("sha256")
    deps["dino_expected_files"] = dino_hashes
    dump_json(solution / "DEPENDENCIES.json", deps)
    files["DEPENDENCIES"] = {"path": "DEPENDENCIES.json", "sha256": sha256(solution / "DEPENDENCIES.json"), "bytes": (solution / "DEPENDENCIES.json").stat().st_size}

    manifest = {
        "schema": "prove.book-to-caddy-r3.solution.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_freeze_sha256": SOURCE_FREEZE_SHA,
        "default_cell": DEFAULT_CELL,
        "files": files,
        "notes": "Task-specific solution bundle only; standard simulator assets/base models are external dependencies. RELEASE.json is reviewed separately and binds this manifest hash.",
    }
    dump_json(solution / "MANIFEST.json", manifest)
    release = {"schema": "prove.book-to-caddy-r3.release.v1", "root_reviewed": False, "physical_execution_authorized": False, "reviewer": "pending_root", "solution_manifest_sha256": sha256(solution / "MANIFEST.json")}
    dump_json(solution / "RELEASE.json", release)
    return {"schema": "prove.book-to-caddy-r3.build-solution.v1", "solution": str(solution.resolve()), "files": len(files), "bytes": sum(v.get("bytes", 0) for v in files.values())}


def verify_solution(solution: Path) -> list[dict[str, Any]]:
    errors = []
    m = solution_manifest(solution)
    for label, item in m["files"].items():
        p = solution / item["path"]
        if not p.exists():
            errors.append({"label": label, "path": item["path"], "error": "missing"})
        elif sha256(p) != item["sha256"]:
            errors.append({"label": label, "path": item["path"], "error": "sha256_mismatch", "actual": sha256(p), "expected": item["sha256"]})
    return errors


def release_authorized(release: dict[str, Any]) -> bool:
    """Require a review for this repository; never inherit the old release bit."""
    review = release.get("migration_review", {})
    return bool(
        release.get("root_reviewed")
        and release.get("physical_execution_authorized")
        and review.get("repository") == "Graph-Self-Adapter"
        and review.get("status") == "approved"
    )


def dependency_settings(solution: Path, dependencies_path: Path | None) -> dict[str, Any]:
    d = load_json(solution / "DEPENDENCIES.json")
    selected = dependencies_path
    if selected is None and DEFAULT_DEPENDENCIES.is_file():
        selected = DEFAULT_DEPENDENCIES
    if selected is not None:
        selected = selected.resolve()
        overrides = load_json(selected)
        allowed = {
            "python", "dino_model_dir", "lila_source", "lila_checkpoint",
            "libero_root", "runtime_overlay", "rlinf_libero_root",
            "libero_one_shot_root",
        }
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"unknown dependency override keys in {selected}: {unknown}")
        d.update(overrides)
    return d


def deps(
    solution: Path,
    repo_root: Path | None,
    python: str | None,
    dino_model_dir: Path | None,
    dependencies_path: Path | None = None,
) -> dict[str, Any]:
    d = dependency_settings(solution, dependencies_path)
    rr = (repo_root or REPO_ROOT).resolve()

    def external_path(key: str) -> Path:
        explicit = d.get(key)
        value = explicit or d.get(f"{key}_default")
        if not value:
            raise ValueError(f"missing external dependency binding: {key}")
        path = Path(value).expanduser()
        if explicit is None and not path.is_absolute():
            path = solution / path
        return path.resolve()

    return {
        "repo_root": rr,
        "python": python or d.get("python") or d.get("python_default") or sys.executable,
        "dino_model_dir": (dino_model_dir or external_path("dino_model_dir")).resolve(),
        "lila_source": external_path("lila_source"),
        "lila_checkpoint": external_path("lila_checkpoint"),
        "libero_root": external_path("libero_root"),
        "runtime_overlay": external_path("runtime_overlay"),
        "rlinf_libero_root": external_path("rlinf_libero_root"),
        "libero_one_shot_root": external_path("libero_one_shot_root"),
    }


def replace_paths(obj: Any, mapping: dict[str, str]) -> Any:
    if isinstance(obj, str):
        if not mapping:
            return obj
        # Replace every historical prefix against the original string in one
        # regex pass.  This handles XML containing many absolute asset paths,
        # while preventing a replacement value from being rewritten again.
        alternatives = "|".join(
            re.escape(old) for old in sorted(mapping, key=len, reverse=True)
        )
        return re.sub(alternatives, lambda match: mapping[match.group(0)], obj)
    if isinstance(obj, list): return [replace_paths(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {replace_paths(k, mapping) if isinstance(k, str) else k: replace_paths(v, mapping) for k, v in obj.items()}
    return obj


def derive_encoder(
    src: Path,
    dst: Path,
    mapping: dict[str, str],
    py: str,
    lila_source: Path,
    lila_checkpoint: Path,
) -> None:
    script = r'''
import json, re, sys, torch
src,dst,map_path,lila_source,lila_checkpoint=sys.argv[1:6]
m=json.load(open(map_path))
def rw(x):
    if isinstance(x,str):
        if not m: return x
        pattern='|'.join(re.escape(a) for a in sorted(m,key=len,reverse=True))
        return re.sub(pattern,lambda match:m[match.group(0)],x)
    if isinstance(x,dict): return {k:rw(v) for k,v in x.items()}
    if isinstance(x,list): return [rw(v) for v in x]
    if isinstance(x,tuple): return tuple(rw(v) for v in x)
    return x
ck=torch.load(src,map_location='cpu'); new=rw(ck)
new['constructor']['lila_source']=lila_source
new['constructor']['lila_checkpoint']=lila_checkpoint
new['lila_binding']['source']=lila_source
new['lila_binding']['checkpoint']=lila_checkpoint
def walk(a,b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a: walk(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): walk(x,y)
walk(ck.get('state_dict',ck), new.get('state_dict',new))
torch.save(new,dst)
'''
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(mapping, f); mp = f.name
    try:
        subprocess.run([
            py, "-c", script, str(src), str(dst), mp,
            str(lila_source), str(lila_checkpoint),
        ], check=True)
    finally:
        Path(mp).unlink(missing_ok=True)


def materialize(solution: Path, output: Path, repo_root: Path | None = None, python: str | None = None, dino_model_dir: Path | None = None, dependencies_path: Path | None = None, authorize: bool = False) -> dict[str, Any]:
    errs = verify_solution(solution)
    if errs: raise ValueError("solution verification failed: " + json.dumps(errs[:3]))
    output.mkdir(parents=True, exist_ok=False)
    m = solution_manifest(solution); dep = deps(solution, repo_root, python, dino_model_dir, dependencies_path)
    freeze = load_json(solution / m["files"]["source_freeze"]["path"])
    p69 = load_json(solution / m["files"]["p69_freeze"]["path"])
    # Mapping from historical absolute roots to installed dependencies.
    rr = dep["repo_root"]
    mapping = {
        "/home/ubuntu/github/prove-p52-stage-local/data/sources/LIBERO-PRO/libero": str(dep["libero_root"]),
        "/home/ubuntu/github/prove-p52-stage-local/data/sources/LIBERO-PRO": str(dep["libero_root"].parent),
        "/home/ubuntu/github/prove-p52-stage-local": str(rr),
        "/home/ubuntu/github/Interaction-RSI": str(rr),
        "/home/zheng/github/Interaction-RSI": str(rr),
        "/home/ubuntu/github/prove-p52-stage-local/data/sources/libero_one_shot_p48": str(dep["libero_one_shot_root"]),
        "/home/ubuntu/github/Interaction-RSI/data/sources/RLinf-LIBERO-PRO": str(dep["rlinf_libero_root"]),
        "/home/ubuntu/github/Interaction-RSI/data/runtime": str(dep["runtime_overlay"].parent),
        "/home/zheng/github/Interaction-RSI/data/runtime": str(dep["runtime_overlay"].parent),
        "/home/zheng/github/Interaction-RSI/data/sources/lila-wam-reference-20260923/models/vla_model_fm.py": str(dep["lila_source"]),
        "/home/zheng/github/Interaction-RSI/data/sources/lila-wam-reference-20260923/VISUAL_WEIGHTS.pt": str(dep["lila_checkpoint"]),
    }
    # Parent chain as materialized absolute JSON files in output workdir.
    parent_items = [m["files"][f"parent_binding_chain.{i}"] for i in range(4)]
    parent_paths = []
    for i, item in enumerate(parent_items):
        data = replace_paths(load_json(solution / item["path"]), mapping)
        dst = output / "materialized" / "parent_chain" / f"{i:02d}_{Path(item['path']).name}"
        dump_json(dst, data); parent_paths.append(dst.resolve())
    for i in range(len(parent_paths)-2, -1, -1):
        data = load_json(parent_paths[i])
        if isinstance(data.get("parent"), dict):
            data["parent"]["path"] = str(parent_paths[i+1])
            data["parent"]["sha256"] = sha256(parent_paths[i+1])
            dump_json(parent_paths[i], data)
    # Controller config and nested task reference closure.
    controller = replace_paths(load_json(solution / m["files"]["controller_config"]["path"]), {})
    def sol(label: str) -> Path: return (solution / m["files"][label]["path"]).resolve()
    mat_ref_dir = output / "materialized" / "refs"
    def mat_json(label: str, name: str, extra_map: dict[str, str] | None = None) -> Path:
        src = sol(label)
        dst = mat_ref_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if extra_map:
            data = replace_paths(load_json(src), extra_map)
            dump_json(dst, data)
        else:
            shutil.copy2(src, dst)
        return dst.resolve()
    book_ref = mat_json("controller_file_bindings.grounding_reference.book_reference", "BOOK_REFERENCE.json")
    private_geom = sol("controller_file_bindings.grounding_reference.source_demo.private_geometry")
    public_frame = sol("controller_file_bindings.grounding_reference.source_demo.public_frame")
    grounding_map = {
        "evaluations/outer_rsi_p69_book_v01/r2_book_silhouette_v01/BOOK_REFERENCE.json": str(book_ref),
        "evaluations/outer_rsi_p69_book_v01/r2_grounding_reference_v01/demo_initial/private/geometry_0000.json": str(private_geom),
        "evaluations/outer_rsi_p69_book_v01/r2_grounding_reference_v01/demo_initial/public/frame_0000.npz": str(public_frame),
    }
    grounding = mat_json("controller_file_bindings.grounding_reference", "GROUNDING_REFERENCE_R2_FINAL.json", grounding_map)
    targets = mat_json("controller_file_bindings.object_relation_targets", "P70_OBJECT_RELATION_TARGETS_v02.json")
    demo_rel = mat_json("controller_file_bindings.grasp_policy.demo_relations", "DEMO_RELATIONS.json")
    train_freeze = mat_json("controller_file_bindings.grasp_policy.training_freeze", "TRAINING_FREEZE.json")
    train_receipt = mat_json("controller_file_bindings.grasp_policy.training_receipt", "TRAINING_RECEIPT.json")
    controller["grasp_policy"]["checkpoint"]["path"] = str(sol("controller_file_bindings.grasp_policy.checkpoint"))
    controller["grasp_policy"]["demo_relations"]["path"] = str(demo_rel)
    controller["grasp_policy"]["training_freeze"]["path"] = str(train_freeze)
    controller["grasp_policy"]["training_receipt"]["path"] = str(train_receipt)
    controller["grounding_reference"]["path"] = str(grounding)
    controller["object_relation_targets"]["path"] = str(targets)
    controller_path = output / "materialized" / "P70_CONTROLLER_CONFIG_R3.json"; dump_json(controller_path, controller)
    # DINO binding external model dir.
    dino = load_json(solution / m["files"]["files.dino_binding"]["path"]); dino["model_dir"] = str(dep["dino_model_dir"])
    dino_path = output / "materialized" / "DINO_BINDING.json"; dump_json(dino_path, dino)
    # Encoder constructor path derivation.
    enc_src = solution / m["files"]["files.encoder_checkpoint"]["path"]
    enc_dst = output / "materialized" / "encoder_checkpoint.pt"
    derive_encoder(
        enc_src, enc_dst, mapping, dep["python"],
        dep["lila_source"], dep["lila_checkpoint"],
    )
    # P69 freeze.
    p69 = replace_paths(p69, mapping)
    p69["p55_binding"] = {"path": str(parent_paths[0]), "sha256": sha256(parent_paths[0])}
    p69["parent_binding_chain"] = [{"path": str(p), "sha256": sha256(p), "schema": load_json(p).get("schema")} for p in parent_paths[:4]]
    p69["init_scope"] = {"path": str((solution / m["files"]["init_scope"]["path"]).resolve()), "sha256": m["files"]["init_scope"]["sha256"]}
    p69["files"]["dino_binding"] = {"path": str(dino_path.resolve()), "sha256": sha256(dino_path)}
    for key in ("a2_head", "action_checkpoint", "precision_head", "priors"):
        p69["files"][key] = {"path": str((solution / m["files"][f"files.{key}"]["path"]).resolve()), "sha256": m["files"][f"files.{key}"]["sha256"]}
    p69["files"]["encoder_checkpoint"] = {"path": str(enc_dst.resolve()), "sha256": sha256(enc_dst)}
    p69_path = output / "materialized" / "P69_PHYSICAL_FREEZE.json"; dump_json(p69_path, p69)
    # P70 freeze.
    freeze = replace_paths(freeze, mapping)
    freeze["controller_config"] = {"path": str(controller_path.resolve()), "sha256": sha256(controller_path)}
    freeze["controller"] = controller
    freeze["p69_freeze"] = {"path": str(p69_path.resolve()), "sha256": sha256(p69_path)}
    freeze["init_scope"] = p69["init_scope"]
    freeze["p55_binding"] = p69["p55_binding"]
    freeze["parent_binding_chain"] = p69["parent_binding_chain"]
    freeze["files"] = copy.deepcopy(p69["files"])
    freeze["controller_file_bindings"] = {}
    materialized_binding_paths = {
        "grounding_reference": grounding,
        "grounding_reference.book_reference": book_ref,
        "grounding_reference.source_demo.private_geometry": private_geom,
        "grounding_reference.source_demo.public_frame": public_frame,
        "object_relation_targets": targets,
        "grasp_policy.demo_relations": demo_rel,
        "grasp_policy.training_freeze": train_freeze,
        "grasp_policy.training_receipt": train_receipt,
    }
    for label, item in m["files"].items():
        if label.startswith("controller_file_bindings."):
            short = label.removeprefix("controller_file_bindings.")
            pth = materialized_binding_paths.get(short, (solution / item["path"]).resolve())
            freeze["controller_file_bindings"][short] = {"path": str(pth), "sha256": sha256(Path(pth))}
    # Runtime sources are external installed repo files; preserve expected hashes.
    freeze["sources"] = {}
    source_freeze = load_json(solution / m["files"]["source_freeze"]["path"])
    for old, digest in source_freeze["sources"].items():
        rel = rel_from_any(old)
        if not rel: continue
        freeze["sources"][str((dep["repo_root"] / rel).resolve())] = digest
    freeze["task_bundle_materialization"] = {"schema": "prove.book-to-caddy-r3.materialized.v1", "solution": str(solution.resolve()), "code_repo_root": str(dep["repo_root"]), "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    freeze_path = output / "PHYSICAL_FREEZE_TASK_BUNDLE.json"; dump_json(freeze_path, freeze)
    review = {"schema": "prove.p70-root-review.v1", "freeze_sha256": sha256(freeze_path), "approved": bool(authorize), "root_reviewed": bool(authorize), "physical_execution_authorized": bool(authorize), "reviewer": "task_bundle_release" if authorize else "task_bundle_pending_root", "scope": "book_to_caddy_r3_task_bundle_replay"}
    review_path = output / "ROOT_PHYSICAL_REVIEW_TASK_BUNDLE.json"; dump_json(review_path, review)
    return {"freeze": str(freeze_path), "freeze_sha256": sha256(freeze_path), "review": str(review_path), "review_sha256": sha256(review_path), "python": dep["python"], "repo_root": str(dep["repo_root"])}


def check(solution: Path, repo_root: Path | None = None, python: str | None = None, dino_model_dir: Path | None = None, dependencies_path: Path | None = None) -> dict[str, Any]:
    errors = verify_solution(solution)
    release = load_json(solution / "RELEASE.json") if (solution / "RELEASE.json").exists() else {}
    manifest_sha = sha256(solution / "MANIFEST.json")
    if release.get("solution_manifest_sha256") != manifest_sha:
        errors.append({"label": "RELEASE", "error": "solution_manifest_sha256_mismatch", "expected": manifest_sha, "actual": release.get("solution_manifest_sha256")})
    review_authorized = release_authorized(release)
    dep = deps(solution, repo_root, python, dino_model_dir, dependencies_path)
    ext = {}
    for k in ("repo_root", "dino_model_dir", "lila_source", "lila_checkpoint", "libero_root", "runtime_overlay", "rlinf_libero_root", "libero_one_shot_root"):
        p = dep[k]
        ext[k] = {"path": str(p), "exists": p.exists()}
        if not p.exists(): errors.append({"label": k, "path": str(p), "error": "missing_dependency"})
    # Verify runtime source hashes in installed repo.
    source_freeze = load_json(solution / solution_manifest(solution)["files"]["source_freeze"]["path"])
    for old, digest in source_freeze["sources"].items():
        rel = rel_from_any(old)
        if not rel: continue
        p = dep["repo_root"] / rel
        if not p.exists() or sha256(p) != digest:
            errors.append({"label": "runtime_source", "path": str(p), "expected": digest, "actual": sha256(p) if p.exists() else None})
    graph = load_json(solution / "graph/task_graph.json")
    controller_cfg = load_json(solution / solution_manifest(solution)["files"]["controller_config"]["path"])
    if graph.get("active_controller_module") != "prove.task_graph.p70_object_controller.P70ObjectController":
        errors.append({"label": "graph", "error": "active_controller_mismatch"})
    if graph.get("stage_ids") != controller_cfg.get("stage_ids"):
        errors.append({"label": "graph", "error": "stage_ids_do_not_match_controller"})
    expected_stage_ids = {s.get("id") for s in controller_cfg.get("stage_ids", [])}
    if set(graph.get("solver_routing", {})) != expected_stage_ids:
        errors.append({"label": "graph", "error": "solver_routing_stage_set_mismatch"})
    if graph.get("solver_routing") != EXPECTED_SOLVER_ROUTING:
        errors.append({"label": "graph", "error": "solver_routing_value_mismatch", "expected": EXPECTED_SOLVER_ROUTING, "actual": graph.get("solver_routing")})
    for fname, digest in load_json(solution / "DEPENDENCIES.json").get("dino_expected_files", {}).items():
        p = dep["dino_model_dir"] / fname
        if not p.exists() or sha256(p) != digest:
            errors.append({"label": "dino_model", "path": str(p), "expected": digest, "actual": sha256(p) if p.exists() else None})
    return {"schema": "prove.book-to-caddy-r3.check.v1", "ok": not errors, "errors": errors, "external_dependencies": ext, "solution": str(solution.resolve()), "review_authorized": review_authorized, "release": {"root_reviewed": bool(release.get("root_reviewed")), "physical_execution_authorized": bool(release.get("physical_execution_authorized")), "reviewer": release.get("reviewer"), "solution_manifest_sha256": release.get("solution_manifest_sha256")}}


def run(solution: Path, output: Path, cell: int | None, all10: bool, repo_root: Path | None, python: str | None, dino_model_dir: Path | None, dependencies_path: Path | None, dry_run: bool) -> dict[str, Any]:
    if all10 and cell is not None:
        raise ValueError("use all10 or cell, not both")
    cells = list(range(10)) if all10 else [7 if cell is None else cell]
    bad_cells = [c for c in cells if c < 0 or c > 9]
    if bad_cells:
        raise ValueError(f"cell index out of range 0..9: {bad_cells}")
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    chk = check(solution, repo_root, python, dino_model_dir, dependencies_path)
    if not chk["ok"]: raise ValueError("check failed: " + json.dumps(chk["errors"][:3]))
    rel = load_json(solution / "RELEASE.json")
    authorized = release_authorized(rel)
    if not dry_run and not authorized:
        raise PermissionError("solution RELEASE.json is not Root-authorized for physical execution")
    mat = materialize(solution, output / "work", repo_root, python, dino_model_dir, dependencies_path, authorize=authorized)
    env = os.environ.copy(); dep = deps(solution, repo_root, python, dino_model_dir, dependencies_path)
    env["PYTHONPATH"] = os.pathsep.join([str(dep["runtime_overlay"]), str(dep["repo_root"])])
    env.update({"HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "MUJOCO_GL":"egl", "PYOPENGL_PLATFORM":"egl", "CUDA_VISIBLE_DEVICES":"0", "MUJOCO_EGL_DEVICE_ID":"0", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1", "PYTHONDONTWRITEBYTECODE":"1", "LIBERO_CONFIG_PATH": str((output / "libero_config_preimport").resolve())})
    commands = []
    for c in cells:
        cell_out = output / f"cell_{c:03d}"
        commands.append([mat["python"], str(dep["repo_root"] / "tools/task_graph/p70_stage_runner.py"), "run-cell", "--freeze", mat["freeze"], "--approval", mat["review"], "--index", str(c), "--output", str(cell_out)])
    if dry_run:
        return {"schema": "prove.book-to-caddy-r3.run-plan.v1", "dry_run": True, "commands": commands, "materialized": mat}
    output.mkdir(parents=True, exist_ok=True)
    results=[]
    for cmd in commands:
        rc = subprocess.run(cmd, cwd=str(dep["repo_root"]), env=env).returncode
        results.append({"command": cmd, "returncode": rc})
        if rc != 0: break
    return {"schema": "prove.book-to-caddy-r3.run-result.v1", "results": results, "materialized": mat}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-solution"); p.add_argument("--repo-root", type=Path, default=REPO_ROOT); p.add_argument("--solution", type=Path, default=DEFAULT_SOLUTION); p.add_argument("--force", action="store_true")
    p = sub.add_parser("check"); p.add_argument("--solution", type=Path, default=DEFAULT_SOLUTION); p.add_argument("--repo-root", type=Path); p.add_argument("--python"); p.add_argument("--dino-model-dir", type=Path); p.add_argument("--dependencies", type=Path)
    p = sub.add_parser("run"); p.add_argument("--solution", type=Path, default=DEFAULT_SOLUTION); p.add_argument("--output", type=Path); p.add_argument("--cell", type=int); p.add_argument("--all10", action="store_true"); p.add_argument("--repo-root", type=Path); p.add_argument("--python"); p.add_argument("--dino-model-dir", type=Path); p.add_argument("--dependencies", type=Path); p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build-solution": print(json.dumps(write_solution(args.repo_root, args.solution, args.force), indent=2, sort_keys=True)); return
    if args.cmd == "check":
        rep = check(args.solution, args.repo_root, args.python, args.dino_model_dir, args.dependencies); print(json.dumps(rep, indent=2, sort_keys=True)); raise SystemExit(0 if rep["ok"] else 1)
    if args.all10 and args.cell is not None:
        raise SystemExit("use --all10 or --cell, not both")
    if args.output is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "all10" if args.all10 else f"cell{7 if args.cell is None else args.cell}"
        args.output = PACKAGE_DIR / "runs" / f"{stamp}_{suffix}"
    try:
        rep = run(args.solution, args.output, args.cell, args.all10, args.repo_root, args.python, args.dino_model_dir, args.dependencies, args.dry_run)
    except (ValueError, FileExistsError, PermissionError) as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(rep, indent=2, sort_keys=True))
    if not args.dry_run and any(r.get("returncode",0)!=0 for r in rep.get("results", [])): raise SystemExit(1)


if __name__ == "__main__":
    main()
