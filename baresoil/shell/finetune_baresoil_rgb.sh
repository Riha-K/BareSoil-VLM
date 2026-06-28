#!/bin/bash
# BareSoilDial RGB fine-tune on BareSoil-Instruct v0.1 (Stage 2 style)
set -x

GPUS=${GPUS:-4}
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export MASTER_PORT=34230
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

BASE_CKPT=${BASE_CKPT:-./checkpoints/EarthDial_4B_RGB}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/BareSoilDial_4B_RGB_v01}

mkdir -p "$OUTPUT_DIR"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  src/earthdial/train/finetune.py \
  --model_name_or_path "${BASE_CKPT}" \
  --conv_style "phi3-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "shell/data/Stage2_BareSoil.json" \
  --overwrite_output_dir True \
  --force_image_size 224 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "src/shell/zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
