import copy
import sys

import torch

from task_packages.book_to_caddy_r3 import task_bundle
from tools.migration.verify_extraction import DEFAULT_INVENTORY, REPO_ROOT, verify


def test_canonical_source_inventory_and_module_origins():
    assert verify(REPO_ROOT, DEFAULT_INVENTORY) == []


def test_package_code_root_is_the_standalone_repository():
    solution = REPO_ROOT / "task_packages/book_to_caddy_r3/solution"
    dependencies = task_bundle.deps(solution, None, None, None)
    assert dependencies["repo_root"] == REPO_ROOT
    assert not (dependencies["repo_root"] / "data").exists()


def test_release_binds_current_launcher_manifest_and_migration_review():
    solution = REPO_ROOT / "task_packages/book_to_caddy_r3/solution"
    release = task_bundle.load_json(solution / "RELEASE.json")
    package = REPO_ROOT / "task_packages/book_to_caddy_r3"
    assert release["solution_manifest_sha256"] == task_bundle.sha256(solution / "MANIFEST.json")
    assert release["launcher_sha256"]["run.sh"] == task_bundle.sha256(package / "run.sh")
    assert release["launcher_sha256"]["task_bundle.py"] == task_bundle.sha256(package / "task_bundle.py")
    review = release["migration_review"]
    assert review["repository"] == "Graph-Self-Adapter"
    if release["root_reviewed"] or release["physical_execution_authorized"]:
        assert review["status"] == "approved"
        assert task_bundle.release_authorized(release)
    else:
        assert review["status"] == "pending_root_review"
        assert not task_bundle.release_authorized(release)


def test_legacy_or_unreviewed_release_cannot_inherit_authorization():
    legacy = {"root_reviewed": True, "physical_execution_authorized": True}
    assert not task_bundle.release_authorized(legacy)
    pending = copy.deepcopy(legacy)
    pending["migration_review"] = {
        "repository": "Graph-Self-Adapter", "status": "pending_root_review"
    }
    assert not task_bundle.release_authorized(pending)


def test_encoder_derivation_rebases_lila_without_changing_tensors(tmp_path):
    original_root = "/old/Interaction-RSI"
    source = tmp_path / "source.pt"
    derived = tmp_path / "derived.pt"
    lila_source = tmp_path / "external/lila.py"
    lila_checkpoint = tmp_path / "external/VISUAL_WEIGHTS.pt"
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save({
        "constructor": {
            "lila_source": f"{original_root}/data/sources/lila.py",
            "lila_checkpoint": f"{original_root}/data/sources/VISUAL_WEIGHTS.pt",
        },
        "lila_binding": {
            "source": f"{original_root}/data/sources/lila.py",
            "checkpoint": f"{original_root}/data/sources/VISUAL_WEIGHTS.pt",
        },
        "state_dict": {"weight": tensor},
    }, source)
    task_bundle.derive_encoder(
        source, derived, {original_root: str(REPO_ROOT)}, sys.executable,
        lila_source, lila_checkpoint,
    )
    package = torch.load(derived, map_location="cpu", weights_only=True)
    assert package["constructor"]["lila_source"] == str(lila_source)
    assert package["constructor"]["lila_checkpoint"] == str(lila_checkpoint)
    assert package["lila_binding"]["source"] == str(lila_source)
    assert package["lila_binding"]["checkpoint"] == str(lila_checkpoint)
    assert torch.equal(package["state_dict"]["weight"], tensor)


def test_path_rebase_rewrites_multiple_roots_without_cascading():
    old_root = "/old/Interaction-RSI"
    old_runtime = f"{old_root}/data/runtime"
    external_runtime = "/installed/Interaction-RSI/data/runtime"
    xml = (
        f'<mesh file="{old_runtime}/robosuite/link0.stl"/> '
        f'<include file="{old_root}/prove/task_graph/model.xml"/>'
    )
    rewritten = task_bundle.replace_paths(
        xml, {old_runtime: external_runtime, old_root: str(REPO_ROOT)}
    )
    assert f'{external_runtime}/robosuite/link0.stl' in rewritten
    assert f'{REPO_ROOT}/prove/task_graph/model.xml' in rewritten
    assert f'{REPO_ROOT}/data/runtime' not in rewritten
