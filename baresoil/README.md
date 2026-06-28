# BareSoil VLM Extension for EarthDial

Domain-specialized bare/barren land monitoring built on [EarthDial](https://arxiv.org/abs/2412.15190) (CVPR 2025).

## Quick Start

```bash
# From repo root
conda activate earthdial
pip install -e .
set PYTHONPATH=%CD%   # Windows
export PYTHONPATH=$(pwd)  # Linux

# Phase 0: EarthDial bare-soil baseline from existing eval JSONL
python -m baresoil.baseline_metrics

# Phase 1: Build BareSoil-Instruct v0.1 (~50K QA) + benchmark
python -m baresoil.build_instruct_v01 --target-size 50000
python -m baresoil.build_bench --version v0.1

# Download models (requires HuggingFace access)
python -m baresoil.download_assets --rgb --eval-shards

# Fine-tune BareSoilDial (4+ GPUs)
bash baresoil/shell/finetune_baresoil_rgb.sh

# Evaluate benchmark
python -m baresoil.eval_bench --bench-version v0.1 --mock

# Phase 2: v1.0 instruct + MS training
python -m baresoil.build_instruct_v1 --target-size 150000
python -m baresoil.build_bench --version v1.0
bash baresoil/shell/finetune_baresoil_ms.sh

# Phase 3: Agent demo
python -m baresoil.agent.baresoil_agent --no-vlm --aoi Demo --bbox 73.7,18.4,74.0,18.7
```

## Directory Layout

```
baresoil/
├── taxonomy.py              # Unified LULC taxonomy mappings
├── baseline_metrics.py        # EarthDial bare-subset baseline
├── build_instruct_v01.py      # 50K instruction dataset
├── build_instruct_v1.py       # 150K + BigEarthNet/Copernicus
├── build_bench.py             # BareSoil-Bench v0.1 / v1.0
├── eval_bench.py              # Benchmark evaluation
├── download_assets.py         # HF model/dataset download
├── docs/TAXONOMY.md
├── data/
│   ├── metrics/               # Baseline & eval JSON reports
│   ├── instruct/v0.1|v1.0/    # Generated JSONL shards
│   └── bench/v0.1|v1.0/       # Test splits per task
├── shell/                     # Training launch scripts
├── agent/                     # BareSoil-Agent (STAC + VLM + report)
└── paper/                     # IEEE draft
```

## New EarthDial Token

`[baresoil]` added in `src/earthdial/train/constants.py` and registered in `finetune.py`.

## Training Configs

- RGB: `src/shell/data/Stage2_BareSoil.json`
- MS:  `src/shell/data/Stage3_BareSoil_MS.json`

## Baseline Summary (EarthDial on bare subsets)

| Subset | Samples | Accuracy | Bare Binary F1 |
|--------|---------|----------|----------------|
| AID bare-related | 489 | 0.96 | 0.997 |
| LCZ bare-related | 777 | 0.67 | 0.812 |

See `baresoil/data/metrics/earthdial_baresoil_baseline.json`.

## Citation

```bibtex
@article{baresoildial2026,
  title={BareSoilDial: A Multi-Spectral Vision-Language Model for Interactive Bare Land Monitoring},
  author={},
  journal={IEEE JSTARS},
  year={2026}
}
```

Based on EarthDial (Soni et al., CVPR 2025).
