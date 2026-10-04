#!/usr/bin/env python3
"""Validate dashboard bindings and render a small static project page."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS = REPO_ROOT / "dashboard" / "STATUS.json"
DEFAULT_OUTPUT = REPO_ROOT / "dashboard" / "index.html"
REQUIRED_BINDINGS = (
    "task_graph", "readme", "engineering_acceptance", "science_report", "video"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_status(status_path: Path, root: Path) -> dict:
    status = json.loads(status_path.read_text())
    bindings = status.get("bindings", {})
    missing = [key for key in REQUIRED_BINDINGS if key not in bindings]
    if missing:
        raise ValueError(f"missing dashboard bindings: {missing}")
    for key, binding in bindings.items():
        path = (root / binding["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"binding escapes repository: {key}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != binding["sha256"]:
            raise ValueError(f"binding hash mismatch for {key}: {actual}")
    return status


def render(status: dict) -> str:
    bindings = status["bindings"]
    graph = json.loads((REPO_ROOT / bindings["task_graph"]["path"]).read_text())
    routing = graph["solver_routing"]
    rows = []
    for stage in graph["stage_ids"]:
        sid = stage["id"]
        rows.append(
            "<tr>"
            f"<td><code>{escape(sid)}</code></td>"
            f"<td>{escape(stage.get('entry', ''))}</td>"
            f"<td>{escape(stage.get('exit', ''))}</td>"
            f"<td>{escape(routing[sid])}</td>"
            "</tr>"
        )
    def link(key: str, label: str) -> str:
        path = bindings[key]["path"]
        return f'<a href="../{escape(path, quote=True)}">{escape(label)}</a>'
    video = "../" + bindings["video"]["path"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(status['headline'])}</title>
<style>body{{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui;color:#17202a}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd1d1;padding:.55rem;text-align:left;vertical-align:top}}th{{background:#eef2f3}}video{{max-width:100%}}code{{white-space:nowrap}}</style></head>
<body><h1>{escape(status['headline'])}</h1><p>{escape(status['summary'])}</p>
<p>{link('readme','Repository guide')} · {link('engineering_acceptance','Engineering acceptance')} · {link('science_report','Scientific report')} · {link('task_graph','Frozen task graph')}</p>
<h2>Frozen P70 R3 stage routing</h2><table><thead><tr><th>Stage</th><th>Entry</th><th>Exit</th><th>Solver</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Annotated engineering replay</h2><video controls preload="metadata" src="{escape(video, quote=True)}"></video>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    status = load_status(args.status.resolve(), REPO_ROOT)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(status))
    print(output)


if __name__ == "__main__":
    main()
