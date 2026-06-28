#!/bin/bash
# Evaluate BareSoilDial on BareSoil-Bench (requires trained checkpoint)
set -e
CHECKPOINT=${CHECKPOINT:-./checkpoints/BareSoilDial_4B_RGB_v01}
BENCH=${BENCH:-v0.1}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -m baresoil.eval_bench --bench-version "${BENCH}" --mock
echo "For real eval, run classification_test.py on bare subsets then:"
echo "  python -m baresoil.eval_bench --bench-version ${BENCH} --predictions path/to/preds.jsonl"
