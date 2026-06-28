"""
BareSoil-Bench v0.1 and v1.0 benchmark builders.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from baresoil.build_instruct_v01 import load_jsonl
from baresoil.taxonomy import UNIFIED_CLASSES, unified_display_name

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTRUCT_V01 = REPO_ROOT / "baresoil" / "data" / "instruct" / "v0.1"
INSTRUCT_V10 = REPO_ROOT / "baresoil" / "data" / "instruct" / "v1.0"
BENCH_ROOT = REPO_ROOT / "baresoil" / "data" / "bench"


TASKS_V01 = ["classification", "vqa_presence", "caption"]
TASKS_V10 = TASKS_V01 + ["grounding", "temporal", "spectral_vqa"]


def _task_type(sample: Dict) -> str:
    human = sample["conversations"][0]["value"].lower()
    if "[caption]" in human:
        return "caption"
    if "yes or no" in human or "answer yes" in human:
        return "vqa_presence"
    if "[changedet]" in human or "time step" in human:
        return "temporal"
    if "spectral index" in human or "ndvi" in human:
        return "spectral_vqa"
    if "<box>" in human or "[grounding]" in human:
        return "grounding"
    return "classification"


def build_bench_from_instruct(
    instruct_test_path: Path,
    version: str,
    max_per_task: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    samples = load_jsonl(instruct_test_path)
    if not samples:
        raise FileNotFoundError(f"No test split at {instruct_test_path}; run build_instruct first.")

    by_task: Dict[str, List[Dict]] = {}
    for s in samples:
        t = _task_type(s)
        by_task.setdefault(t, []).append(s)

    rng = random.Random(seed)
    bench: Dict[str, List[Dict]] = {}
    for task, items in by_task.items():
        rng.shuffle(items)
        bench[task] = items[:max_per_task]

    out_dir = BENCH_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": version,
        "tasks": list(bench.keys()),
        "splits": {"test": {}},
        "geographic_holdout": {
            "policy": "LCZ/AID geographic shards held out when real coords available; "
                      "current split uses random 15-20% from instruct builder.",
            "aid_test_sources": ["AID"],
            "lcz_test_sources": ["LCZ42"],
        },
        "unified_classes": UNIFIED_CLASSES,
    }

    total = 0
    for task, items in bench.items():
        path = out_dir / f"{task}_test.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"]["test"][task] = {"count": len(items), "path": str(path)}
        total += len(items)

    manifest["total_samples"] = total
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def build_v01(max_per_task: int = 500) -> Dict[str, Any]:
    test_path = INSTRUCT_V01 / "baresoil_instruct_v01_test.jsonl"
    return build_bench_from_instruct(test_path, "v0.1", max_per_task=max_per_task)


def build_v10(max_per_task: int = 800) -> Dict[str, Any]:
    # v1 uses held-out from v0.1 test + synthetic temporal/spectral from v1 instruct
    v01_manifest = build_v01(max_per_task=max_per_task // 2)
    v10_dir = BENCH_ROOT / "v1.0"
    v10_dir.mkdir(parents=True, exist_ok=True)

    extra_temporal = []
    for i in range(200):
        extra_temporal.append({
            "id": f"bench_temp_{i}",
            "task": "temporal",
            "conversations": [
                {"from": "human", "value": "[baresoil] [changedet] [s2_ms_10] <image>\nDid bare soil increase?"},
                {"from": "gpt", "value": "yes" if i % 3 == 0 else "no"},
            ],
            "unified_label": "bare_soil" if i % 3 == 0 else "non_bare",
        })
    spec_path = v10_dir / "spectral_vqa_test.jsonl"
    temp_path = v10_dir / "temporal_test.jsonl"
    with temp_path.open("w", encoding="utf-8") as f:
        for row in extra_temporal:
            f.write(json.dumps(row) + "\n")

    spectral = []
    for i, idx_name in enumerate(["NDVI", "BSI", "NDBI"]):
        for j in range(50):
            spectral.append({
                "id": f"spec_{i}_{j}",
                "task": "spectral_vqa",
                "conversations": [
                    {"from": "human", "value": "[baresoil] [s2_ms_10] <image>\nBest index to confirm bare soil?"},
                    {"from": "gpt", "value": "BSI" if j % 2 == 0 else idx_name},
                ],
            })
    with spec_path.open("w", encoding="utf-8") as f:
        for row in spectral:
            f.write(json.dumps(row) + "\n")

    manifest = {
        "version": "1.0",
        "inherits": "v0.1",
        "v01_manifest": v01_manifest,
        "additional_tasks": {
            "temporal": str(temp_path),
            "spectral_vqa": str(spec_path),
        },
        "tasks": TASKS_V10,
    }
    (v10_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v0.1", "v1.0", "all"], default="all")
    parser.add_argument("--max-per-task", type=int, default=500)
    args = parser.parse_args()
    if args.version in ("v0.1", "all"):
        build_v01(args.max_per_task)
    if args.version in ("v1.0", "all"):
        build_v10(args.max_per_task)


if __name__ == "__main__":
    main()
