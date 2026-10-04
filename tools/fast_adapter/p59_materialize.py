#!/usr/bin/env python3
"""Materialize the sealed P59 tensor dataset; zero physics, offline labels explicit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from prove.fast_adapter.p59_data import load_training_dataset


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--p50", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("materialized dataset already exists; no overwrite")
    value = load_training_dataset(args.dataset_root, args.manifest, args.pca, args.p50,
                                  allow_private=True, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(value, stream)
    print(json.dumps({"schema": value["schema"], "path": str(args.output),
                      "sha256": sha(args.output),
                      "train_episodes": len(value["splits"]["train"]),
                      "developer_episodes": len(value["splits"]["developer"]),
                      "physics_rollouts": 0}, indent=2))


if __name__ == "__main__":
    main()
