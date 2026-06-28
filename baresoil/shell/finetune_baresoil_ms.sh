#!/bin/bash
# BareSoilDial-MS fine-tune (Stage 3 style, multispectral)
set -x

GPUS=${GPUS:-4}
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export MASTER_PORT=34231

BASE_CKPT=${BASE_CKPT:-./checkpoints/BareSoilDial_4B_RGB_v01}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/BareSoilDial_4B_MS_v10}

mkdir -p "$OUTPUT_DIR"

torchrun \
  --nnodes=1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  src/earthdial/train/finetune.py \
  --model_name_or_path "${BASE_CKPT}" \
  --conv_style "phi3-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "shell/data/Stage3_BareSoil_MS.json" \
  --overwrite_output_dir True \
  --force_image_size 224 \
  --max_dynamic_patch 6 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --learning_rate 4e-5 \
  --max_seq_length 4096 \
  --dynamic_image_size True \
  --use_thumbnail True \
  --deepspeed "src/shell/zero_stage1_config.json" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
