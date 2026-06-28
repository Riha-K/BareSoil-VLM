"""
Download EarthDial checkpoints and datasets for BareSoil pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
DATA_DIR = REPO_ROOT / "baresoil" / "data" / "external"


def download_model(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(local_dir))
    print(f"Downloaded {repo_id} -> {local_dir}")


def download_dataset(pattern: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="akshaydudhane/EarthDial-Dataset",
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=str(local_dir),
    )
    print(f"Downloaded dataset pattern {pattern} -> {local_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download BareSoil pipeline assets")
    parser.add_argument("--rgb", action="store_true", help="EarthDial_4B_RGB")
    parser.add_argument("--ms", action="store_true", help="EarthDial_4B_MS")
    parser.add_argument("--eval-shards", action="store_true", help="Classification eval shards")
    parser.add_argument("--train-shards", action="store_true", help="Training shards subset")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.rgb:
        download_model("akshaydudhane/EarthDial_4B_RGB", CHECKPOINT_DIR / "EarthDial_4B_RGB")
    if args.all or args.ms:
        download_model("akshaydudhane/EarthDial_4B_MS", CHECKPOINT_DIR / "EarthDial_4B_MS")
    if args.all or args.eval_shards:
        download_dataset(
            "Eardial_downstream_task_datasets/Classification/**",
            REPO_ROOT / "src" / "earthdial" / "eval",
        )
    if args.all or args.train_shards:
        download_dataset("training_set/**", REPO_ROOT / "src" / "earthdial" / "trainset")


if __name__ == "__main__":
    main()
