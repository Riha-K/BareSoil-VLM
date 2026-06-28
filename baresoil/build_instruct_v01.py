"""
Build BareSoil-Instruct v0.1 from AID, LCZ eval metadata + RS caption subsets.
Exports EarthDial-compatible JSONL and HuggingFace dataset shards when images available.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from baresoil.instruct_templates import (
    BINARY_VQA_NO,
    BINARY_VQA_YES,
    expand_all_templates,
    format_options,
)
from baresoil.taxonomy import (
    AID_TO_UNIFIED,
    LCZ_TO_UNIFIED,
    is_bare_positive,
    map_label,
    unified_display_name,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CLS = REPO_ROOT / "src" / "earthdial" / "eval" / "rs_classification" / "results"
EVAL_CAP = REPO_ROOT / "src" / "earthdial" / "eval" / "rs_image_caption" / "results"
OUT_DIR = REPO_ROOT / "baresoil" / "data" / "instruct" / "v0.1"


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def earthdial_conversation(human: str, assistant: str) -> Dict:
    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": assistant},
        ]
    }


def aid_samples(seed: int) -> Iterator[Dict]:
    items = load_jsonl(EVAL_CLS / "AID.jsonl")
    bare_keys = set(AID_TO_UNIFIED.keys())
    items = [x for x in items if x.get("annotation") in bare_keys]
    templates = expand_all_templates()["classify_rgb"]
    rng = random.Random(seed)
    for idx, item in enumerate(items):
        unified = map_label(item["annotation"], "aid")
        answer = unified_display_name(unified or "non bare")
        for t_idx, template in enumerate(templates):
            yield {
                "id": f"aid_{idx}_{t_idx}",
                "source": "AID",
                "scheme": "aid",
                "raw_label": item["annotation"],
                "unified_label": unified,
                "split": "train" if rng.random() < 0.85 else "test",
                **earthdial_conversation(template, answer),
                "metadata": {"eval_question": item.get("question", "")},
            }
        # binary VQA
        human = "[baresoil] [hr_rgb_0.5] <image>\nIs bare or barren land the dominant cover? Answer yes or no."
        ans = BINARY_VQA_YES if is_bare_positive(unified) else BINARY_VQA_NO
        yield {
            "id": f"aid_{idx}_vqa",
            "source": "AID",
            "scheme": "aid",
            "raw_label": item["annotation"],
            "unified_label": unified,
            "split": "train" if rng.random() < 0.85 else "test",
            **earthdial_conversation(human, ans),
        }


def lcz_samples(seed: int) -> Iterator[Dict]:
    items = load_jsonl(EVAL_CLS / "rs_LCZ_test.jsonl")
    keys = {k.lower() for k in LCZ_TO_UNIFIED}
    items = [x for x in items if x.get("annotation", "").lower() in keys]
    templates_rgb = expand_all_templates()["classify_rgb"][:2]
    templates_ms = expand_all_templates()["classify_ms"]
    rng = random.Random(seed + 1)
    for idx, item in enumerate(items):
        raw = item["annotation"]
        unified = map_label(raw, "lcz")
        answer = unified_display_name(unified or "non bare")
        for t_idx, template in enumerate(templates_rgb + templates_ms):
            yield {
                "id": f"lcz_{idx}_{t_idx}",
                "source": "LCZ42",
                "scheme": "lcz",
                "raw_label": raw,
                "unified_label": unified,
                "split": "train" if rng.random() < 0.85 else "test",
                **earthdial_conversation(template, answer),
            }


def caption_samples(seed: int) -> Iterator[Dict]:
    keywords = ("bareland", "bare land", "bare soil", "barren", "wasteland", "desert", "bare ground", "khaki")
    rng = random.Random(seed + 2)
    templates = expand_all_templates()["caption"]
    cap_id = 0
    for fname in ["RSICD_Captions.jsonl", "RSITMD_Captions.jsonl", "UCM_captions.jsonl"]:
        for item in load_jsonl(EVAL_CAP / fname):
            text = item.get("answer", "").lower()
            if not any(k in text for k in keywords):
                continue
            for t_idx, template in enumerate(templates):
                yield {
                    "id": f"cap_{cap_id}_{t_idx}",
                    "source": fname.replace(".jsonl", ""),
                    "scheme": "caption",
                    "raw_label": item.get("answer", ""),
                    "unified_label": "bare_soil",
                    "split": "train" if rng.random() < 0.8 else "test",
                    **earthdial_conversation(template, item.get("answer", "")),
                }
            cap_id += 1


def augment_to_target(samples: List[Dict], target: int, seed: int) -> List[Dict]:
    """Paraphrase augmentation via template cycling to reach ~target size."""
    if len(samples) >= target:
        return samples[:target]
    rng = random.Random(seed)
    out = list(samples)
    paraphrases = [
        ("Classify: {opts}.", "Land cover class?"),
        ("Surface type from options: {opts}", "One label only."),
    ]
    opts = format_options()
    i = 0
    while len(out) < target:
        base = samples[i % len(samples)]
        p_h, p_a = paraphrases[i % len(paraphrases)]
        human = f"[baresoil] [hr_rgb_0.5] [classify] <image>\n{p_h.format(opts=opts)} {p_a}"
        assistant = base["conversations"][1]["value"]
        out.append({
            **{k: v for k, v in base.items() if k != "id"},
            "id": f"aug_{len(out)}",
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": assistant},
            ],
            "augmented": True,
        })
        i += 1
    return out


def build_v01(target_size: int = 50000, seed: int = 42) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples: List[Dict] = []
    samples.extend(list(aid_samples(seed)))
    samples.extend(list(lcz_samples(seed)))
    samples.extend(list(caption_samples(seed)))
    samples = augment_to_target(samples, target_size, seed)

    rng = random.Random(seed)
    rng.shuffle(samples)
    train = [s for s in samples if s.get("split") == "train"]
    test = [s for s in samples if s.get("split") == "test"]

    train_path = OUT_DIR / "baresoil_instruct_v01_train.jsonl"
    test_path = OUT_DIR / "baresoil_instruct_v01_test.jsonl"
    for path, data in [(train_path, train), (test_path, test)]:
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "version": "0.1",
        "target_size": target_size,
        "total": len(samples),
        "train": len(train),
        "test": len(test),
        "by_source": {},
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    for s in samples:
        stats["by_source"][s["source"]] = stats["by_source"].get(s["source"], 0) + 1

    stats_path = OUT_DIR / "baresoil_instruct_v01_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def export_hf_dataset(jsonl_path: Path, output_dir: Path) -> None:
    """Export JSONL to HuggingFace datasets on-disk format (text-only fallback)."""
    try:
        from datasets import Dataset
    except ImportError:
        print("datasets package required for HF export")
        return
    rows = load_jsonl(jsonl_path)
    ds = Dataset.from_list(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"Saved HF dataset to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-hf", action="store_true")
    args = parser.parse_args()
    stats = build_v01(target_size=args.target_size, seed=args.seed)
    if args.export_hf:
        export_hf_dataset(Path(stats["train_path"]), OUT_DIR / "hf_train")


if __name__ == "__main__":
    main()
