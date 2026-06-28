#!/usr/bin/env python3
"""Run full BareSoil pipeline setup (Phases 0-3 data prep)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(REPO))


def main():
    py = sys.executable
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO)}

    steps = [
        [py, "-m", "baresoil.baseline_metrics"],
        [py, "-m", "baresoil.build_instruct_v01", "--target-size", "50000"],
        [py, "-m", "baresoil.build_bench", "--version", "v0.1"],
        [py, "-m", "baresoil.build_instruct_v1", "--target-size", "150000"],
        [py, "-m", "baresoil.build_bench", "--version", "v1.0"],
        [py, "-m", "baresoil.eval_bench", "--mock", "--bench-version", "v0.1"],
        [py, "-m", "baresoil.agent.baresoil_agent", "--no-vlm", "--aoi", "Setup_Demo",
         "--bbox", "73.7,18.4,74.0,18.7"],
    ]
    for cmd in steps:
        subprocess.check_call(cmd, cwd=str(REPO), env=env)
    print("\nBareSoil pipeline setup complete.")


if __name__ == "__main__":
    main()
