#!/home/clover/miniconda3/envs/geometry/bin/python
"""Verify the V23 package, then preload the existing isolated SAM2 backend."""
import hashlib
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "source/tools/deployment/all_code_stage_operator_entry.py"
COMPAT = Path("/home/clover/github/Graph-Self-Adapter/deployments/cup_gemini_v19_r1_sam_compat")
CHECKPOINT = Path("/home/clover/DataModel/sam2.1/sam2.1_hiera_small.pt")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


receipt = json.loads((ROOT / "OPERATOR_TRIAL_RECEIPT.json").read_text())
if receipt.get("schema") != "graph-self-adapter.all-code-stage-operator-package.v1":
    raise RuntimeError("unexpected V23 operator receipt")
for row in receipt["files"]:
    path = ROOT / row["relative_path"]
    if not path.is_file() or sha256(path) != row["sha256"]:
        raise RuntimeError("V23 package binding changed: " + row["relative_path"])
compat = json.loads((COMPAT / "SOURCE_HASH_MANIFEST.json").read_text())
if compat.get("schema") != "graph-self-adapter.isolated-sam2-compat.v1":
    raise RuntimeError("unexpected isolated SAM compatibility manifest")
for logical, expected in compat["bindings"].items():
    if logical.startswith("sam2/") or logical.startswith("configs/"):
        path = COMPAT / logical
    elif logical == "checkpoint/sam2.1_hiera_small.pt":
        path = CHECKPOINT
    else:
        continue
    if sha256(path) != expected:
        raise RuntimeError("SAM compatibility binding changed: " + logical)
if len(sys.argv) < 2 or Path(sys.argv[1]).resolve() != ENTRY.resolve():
    raise SystemExit("V23 bootstrap only accepts the package operator entry")
sys.path.insert(0, str(COMPAT))
import sam2
from sam2 import _C
if Path(sam2.__file__).resolve().parent != COMPAT / "sam2" or Path(_C.__file__).resolve() != COMPAT / "sam2/_C.py":
    raise RuntimeError("isolated SAM2 compatibility backend was not selected")
sys.argv = [str(ENTRY), *sys.argv[2:]]
runpy.run_path(str(ENTRY), run_name="__main__")

