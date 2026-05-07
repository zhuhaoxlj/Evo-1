#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}/Evo_1"

CONDA_ENV="${CONDA_ENV:-Evo1}"
CONDA_SH="${CONDA_SH:-/home/bl/miniforge3/etc/profile.d/conda.sh}"
GPU_ID="${GPU_ID:-0}"
RUN_NAME="${RUN_NAME:-Evo1_metaworld_stage1_bs128_5000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CKPT_INTERVAL="${CKPT_INTERVAL:-2500}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DATASET_CONFIG_PATH="${DATASET_CONFIG_PATH:-dataset/config.yaml}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
BACKGROUND="${BACKGROUND:-1}"
ENABLE_PROXY="${ENABLE_PROXY:-0}"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

PROXY_PREFIX=""
if [[ "${ENABLE_PROXY}" != "1" ]]; then
  PROXY_PREFIX="unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; "
fi

TRAIN_CMD="${PROXY_PREFIX}cd \"${PROJECT_DIR}\" && CUDA_VISIBLE_DEVICES=\"${GPU_ID}\" HF_ENDPOINT=\"${HF_ENDPOINT}\" PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF}\" conda run -n \"${CONDA_ENV}\" python scripts/train.py --run_name \"${RUN_NAME}\" --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size \"${BATCH_SIZE}\" --image_size 448 --max_steps \"${MAX_STEPS}\" --log_interval \"${LOG_INTERVAL}\" --ckpt_interval \"${CKPT_INTERVAL}\" --warmup_steps \"${WARMUP_STEPS}\" --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --num_workers \"${NUM_WORKERS}\" --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path \"${DATASET_CONFIG_PATH}\" --per_action_dim 24 --state_dim 24 --save_dir \"${SAVE_DIR}\""

echo "[stage1] project: ${PROJECT_DIR}"
echo "[stage1] conda_env: ${CONDA_ENV}"
echo "[stage1] gpu: ${GPU_ID}"
echo "[stage1] run_name: ${RUN_NAME}"
echo "[stage1] batch_size: ${BATCH_SIZE}"
echo "[stage1] max_steps: ${MAX_STEPS}"
echo "[stage1] save_dir: ${SAVE_DIR}"
echo "[stage1] log_file: ${LOG_FILE}"
echo "[stage1] enable_proxy: ${ENABLE_PROXY}"

if [[ "${BACKGROUND}" == "1" ]]; then
  echo "[stage1] mode: background"
  zsh -lic "source \"${CONDA_SH}\" && nohup bash -lc '${TRAIN_CMD}' > '${LOG_FILE}' 2>&1 < /dev/null & echo \$!"
  echo "[stage1] tail: tail -f \"${LOG_FILE}\""
else
  echo "[stage1] mode: foreground"
  zsh -lic "source \"${CONDA_SH}\" && bash -lc '${TRAIN_CMD}'" 2>&1 | tee "${LOG_FILE}"
fi
