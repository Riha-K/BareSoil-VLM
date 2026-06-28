"""
BareSoil-Instruct v1.0 — scales v0.1 with BigEarthNet S2 and Copernicus weak-label templates.
Requires BigEarthNet shards under baresoil/data/external/ or EarthDial trainset paths.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from baresoil.build_instruct_v01 import build_v01, load_jsonl, earthdial_conversation
from baresoil.taxonomy import CORINE19_TO_UNIFIED, unified_display_name

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "baresoil" / "data" / "instruct" / "v1.0"

# CORINE classes relevant to bare soil
BARE_CORINE_LABELS = list(CORINE19_TO_UNIFIED.keys())

TEMPORAL_TEMPLATES = [
    "[baresoil] [changedet] [hr_rgb_temp_0.5] <image>\nCompare the two time steps. Did bare soil area increase?",
    "[changedet] [s2_ms_10] <image>\n[baresoil] Describe bare land change between observations.",
]

COPERNICUS_WEAK_TEMPLATES = [
    "[baresoil] [s2_rgb_10] [classify] <image>\nPost-harvest bare soil detected (Copernicus weak label). Confirm: bare soil or vegetated?",
    "[s2_ms_10] [classify] <image>\n[baresoil] European cropland bare soil layer — classify surface.",
]


def bigearthnet_synthetic_entries(per_label: int = 2000, seed: int = 42) -> List[Dict]:
    """Template entries for BigEarthNet S2; replace with real shard indices when data mounted."""
    rng = random.Random(seed)
    rows = []
    for label in BARE_CORINE_LABELS:
        unified = CORINE19_TO_UNIFIED[label]
        answer = unified_display_name(unified)
        for i in range(per_label):
            human = (
                f"[baresoil] [s2_ms_10] [classify] <image>\n"
                f"Multi-label CORINE scene. Primary bare-related class? "
                f"Options: {', '.join(unified_display_name(u) for u in set(CORINE19_TO_UNIFIED.values()))}."
            )
            rows.append({
                "id": f"ben_s2_{label}_{i}",
                "source": "BigEarthNet_S2",
                "scheme": "corine19",
                "raw_label": label,
                "unified_label": unified,
                "split": "train" if rng.random() < 0.9 else "test",
                "weak_label": False,
                **earthdial_conversation(human, answer),
                "placeholder_image": True,
            })
    return rows


def copernicus_weak_entries(count: int = 5000, seed: int = 43) -> List[Dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        is_bare = rng.random() < 0.5
        template = rng.choice(COPERNICUS_WEAK_TEMPLATES)
        answer = "bare soil" if is_bare else "non bare"
        rows.append({
            "id": f"cop_hrl_{i}",
            "source": "Copernicus_BareSoil_HRL",
            "scheme": "copernicus",
            "raw_label": "bare_soil_after_harvest" if is_bare else "vegetated",
            "unified_label": "bare_soil" if is_bare else "non_bare",
            "split": "train" if rng.random() < 0.85 else "test",
            "weak_label": True,
            **earthdial_conversation(template, answer),
            "placeholder_image": True,
        })
    return rows


def temporal_entries(count: int = 2000, seed: int = 44) -> List[Dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        increased = rng.random() < 0.4
        template = rng.choice(TEMPORAL_TEMPLATES)
        answer = (
            "Yes, bare soil area increased with reduced vegetation cover."
            if increased
            else "No significant bare soil expansion detected."
        )
        rows.append({
            "id": f"temp_{i}",
            "source": "temporal_synthetic",
            "scheme": "temporal",
            "unified_label": "bare_soil" if increased else "non_bare",
            "split": "train" if rng.random() < 0.85 else "test",
            **earthdial_conversation(template, answer),
            "placeholder_image": True,
        })
    return rows


def merge_with_v01(target_size: int = 150000, seed: int = 42) -> Dict[str, Any]:
    v01_stats = build_v01(target_size=min(50000, target_size // 3), seed=seed)
    v01_train = load_jsonl(Path(v01_stats["train_path"]))

    extra = []
    extra.extend(bigearthnet_synthetic_entries(per_label=1500, seed=seed))
    extra.extend(copernicus_weak_entries(count=8000, seed=seed + 1))
    extra.extend(temporal_entries(count=3000, seed=seed + 2))

    combined = v01_train + extra
    rng = random.Random(seed)
    rng.shuffle(combined)
    if len(combined) > target_size:
        combined = combined[:target_size]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "baresoil_instruct_v10_train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "version": "1.0",
        "total": len(combined),
        "from_v01": len(v01_train),
        "bigearthnet_s2": sum(1 for x in extra if x["source"] == "BigEarthNet_S2"),
        "copernicus_weak": sum(1 for x in extra if x["source"] == "Copernicus_BareSoil_HRL"),
        "temporal": sum(1 for x in extra if x["source"] == "temporal_synthetic"),
        "train_path": str(train_path),
        "note": "Replace placeholder_image entries with real shards when BigEarthNet/Copernicus rasters mounted.",
    }
    (OUT_DIR / "baresoil_instruct_v10_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    merge_with_v01(target_size=args.target_size, seed=args.seed)


if __name__ == "__main__":
    main()
