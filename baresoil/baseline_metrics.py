"""
Extract EarthDial baseline metrics on bare-soil-related eval subsets.
Works from existing rs_classification / rs_image_caption result JSONL files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from baresoil.taxonomy import (
    AID_TO_UNIFIED,
    BARE_POSITIVE_CLASSES,
    LCZ_TO_UNIFIED,
    is_bare_positive,
    map_label,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "src" / "earthdial" / "eval"
RESULTS_CLS = EVAL_ROOT / "rs_classification" / "results"
RESULTS_CAP = EVAL_ROOT / "rs_image_caption" / "results"
OUTPUT_DIR = REPO_ROOT / "baresoil" / "data" / "metrics"


def token_f1(reference: str, candidate: str) -> float:
    """EarthDial classification eval: token overlap recall."""
    ref = reference.strip().lower().replace(" ", "")
    cand = candidate.strip().lower().replace(" ", "")
    if not ref or not cand:
        return 0.0
    ref_tokens = set(re.findall(r"\w+", ref))
    cand_tokens = set(re.findall(r"\w+", cand))
    if not ref_tokens:
        return 0.0
    common = ref_tokens & cand_tokens
    if not common:
        return 0.0
    return len(common) / len(ref_tokens)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aid_bare_subset(items: List[Dict]) -> List[Dict]:
    keys = set(AID_TO_UNIFIED.keys())
    return [x for x in items if x.get("annotation", "") in keys]


def lcz_bare_subset(items: List[Dict]) -> List[Dict]:
    keys = {k.lower() for k in LCZ_TO_UNIFIED.keys()}
    return [x for x in items if x.get("annotation", "").lower() in keys]


def caption_bare_subset(items: List[Dict]) -> List[Dict]:
    keywords = (
        "bareland", "bare land", "bare soil", "barren", "wasteland",
        "desert", "bare ground", "khaki", "maroon soil", "uncultivated",
    )
    out = []
    for x in items:
        text = (x.get("answer", "") + " " + " ".join(
            x.get(f"caption{i}", "") for i in range(5)
        )).lower()
        if any(k in text for k in keywords):
            out.append(x)
    return out


def compute_subset_metrics(
    items: List[Dict],
    scheme: str,
    label_key: str = "annotation",
    pred_key: str = "answer",
) -> Dict[str, Any]:
    if not items:
        return {"count": 0, "accuracy": 0.0, "bare_f1": 0.0}

    per_class_scores: Dict[str, List[float]] = defaultdict(list)
    binary_tp = binary_fp = binary_fn = binary_tn = 0

    for item in items:
        ref = item.get(label_key, "")
        pred = item.get(pred_key, "")
        score = token_f1(ref, pred)
        per_class_scores[ref].append(score)

        unified = map_label(ref, scheme)
        if unified is None:
            continue
        ref_bin = is_bare_positive(unified)
        pred_unified = map_label(pred, scheme) or (
            "bare_soil" if any(w in pred.lower() for w in ("bare", "desert", "sand", "barren")) else "non_bare"
        )
        pred_bin = is_bare_positive(pred_unified)
        if ref_bin and pred_bin:
            binary_tp += 1
        elif ref_bin and not pred_bin:
            binary_fn += 1
        elif not ref_bin and pred_bin:
            binary_fp += 1
        else:
            binary_tn += 1

    accuracy = sum(token_f1(i[label_key], i[pred_key]) for i in items) / len(items)
    prec = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) else 0.0
    rec = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) else 0.0
    bare_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    per_class_acc = {k: sum(v) / len(v) for k, v in per_class_scores.items()}
    return {
        "count": len(items),
        "accuracy": round(accuracy, 4),
        "bare_binary_f1": round(bare_f1, 4),
        "bare_binary_precision": round(prec, 4),
        "bare_binary_recall": round(rec, 4),
        "per_class_accuracy": {k: round(v, 4) for k, v in sorted(per_class_acc.items())},
        "confusion_binary": {
            "tp": binary_tp, "fp": binary_fp, "fn": binary_fn, "tn": binary_tn,
        },
    }


def run_baseline_extraction() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "description": "EarthDial baseline on bare-soil-related eval subsets",
        "model": "EarthDial_4B (from existing eval JSONL)",
        "subsets": {},
    }

    aid_all = load_jsonl(RESULTS_CLS / "AID.jsonl")
    aid_bare = aid_bare_subset(aid_all)
    report["subsets"]["AID_bare_related"] = {
        **compute_subset_metrics(aid_bare, "aid"),
        "full_dataset_accuracy": compute_subset_metrics(aid_all, "aid")["accuracy"],
        "class_distribution": dict(Counter(x["annotation"] for x in aid_bare)),
    }

    lcz_all = load_jsonl(RESULTS_CLS / "rs_LCZ_test.jsonl")
    lcz_bare = lcz_bare_subset(lcz_all)
    report["subsets"]["LCZ_bare_related"] = {
        **compute_subset_metrics(lcz_bare, "lcz"),
        "full_dataset_accuracy": compute_subset_metrics(lcz_all, "lcz")["accuracy"],
        "class_distribution": dict(Counter(x["annotation"] for x in lcz_bare)),
    }

    caption_sets = {}
    for fname in ["RSICD_Captions.jsonl", "RSITMD_Captions.jsonl", "UCM_captions.jsonl"]:
        items = load_jsonl(RESULTS_CAP / fname)
        bare = caption_bare_subset(items)
        # caption task: check if model mentions bare-soil terms when GT does
        hits = 0
        for x in bare:
            ans = x.get("answer", "").lower()
            if any(k in ans for k in ("bare", "desert", "barren", "wasteland", "soil")):
                hits += 1
        caption_sets[fname.replace(".jsonl", "")] = {
            "bare_keyword_samples": len(bare),
            "bare_term_in_prediction": hits,
            "bare_term_rate": round(hits / len(bare), 4) if bare else 0.0,
        }
    report["subsets"]["caption_bare_keyword"] = caption_sets

  # aggregate summary
    report["summary"] = {
        "aid_bare_accuracy": report["subsets"]["AID_bare_related"]["accuracy"],
        "aid_bare_f1": report["subsets"]["AID_bare_related"]["bare_binary_f1"],
        "lcz_bare_accuracy": report["subsets"]["LCZ_bare_related"]["accuracy"],
        "lcz_bare_f1": report["subsets"]["LCZ_bare_related"]["bare_binary_f1"],
        "total_bare_eval_samples": (
            report["subsets"]["AID_bare_related"]["count"]
            + report["subsets"]["LCZ_bare_related"]["count"]
        ),
    }

    out_path = OUTPUT_DIR / "earthdial_baresoil_baseline.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Full report: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Extract bare-soil baseline metrics")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    report = run_baseline_extraction()
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
