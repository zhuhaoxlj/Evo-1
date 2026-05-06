#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
STAGE1_SAVE_DIR="${STAGE1_SAVE_DIR:-${ROOT_DIR}/checkpoints/metaworld_stage1_bs32_20000_fix}"
STAGE1_FINAL_STEP="${STAGE1_FINAL_STEP:-20000}"
STAGE1_FINAL_DIR="${STAGE1_FINAL_DIR:-${STAGE1_SAVE_DIR}/step_${STAGE1_FINAL_STEP}}"
STAGE1_FINAL_MODEL="${STAGE1_FINAL_MODEL:-${STAGE1_FINAL_DIR}/pytorch_model.bin}"
STAGE2_SCRIPT="${STAGE2_SCRIPT:-${ROOT_DIR}/run_stage2_train.sh}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/watch_stage1_then_stage2.log}"
LOCK_FILE="${LOCK_FILE:-${ROOT_DIR}/.stage2_started}"
BACKGROUND="${BACKGROUND:-1}"

mkdir -p "${LOG_DIR}"

watch_loop() {
  echo "[watch] started at $(date '+%F %T')"
  echo "[watch] stage1_final_model: ${STAGE1_FINAL_MODEL}"
  echo "[watch] stage2_script: ${STAGE2_SCRIPT}"
  echo "[watch] check_interval: ${CHECK_INTERVAL}s"
  echo "[watch] lock_file: ${LOCK_FILE}"

  while true; do
    if [[ -f "${LOCK_FILE}" ]]; then
      echo "[watch] lock exists, stage2 was already started: ${LOCK_FILE}"
      exit 0
    fi

    if [[ -f "${STAGE1_FINAL_MODEL}" ]]; then
      echo "[watch] detected Stage 1 final checkpoint at $(date '+%F %T')"
      echo "[watch] launching Stage 2..."
      RESUME_PATH="${STAGE1_FINAL_DIR}" "${STAGE2_SCRIPT}"
      echo "started_at=$(date '+%F %T')" > "${LOCK_FILE}"
      echo "resume_path=${STAGE1_FINAL_DIR}" >> "${LOCK_FILE}"
      echo "[watch] Stage 2 launch command finished at $(date '+%F %T')"
      exit 0
    fi

    if pgrep -af 'Evo1_metaworld_stage1_bs32_20000_fix|scripts/train.py' >/dev/null 2>&1; then
      echo "[watch] $(date '+%F %T') Stage 1 still running; waiting for ${STAGE1_FINAL_MODEL}"
    else
      echo "[watch] $(date '+%F %T') Stage 1 process not found; still waiting for checkpoint"
    fi

    sleep "${CHECK_INTERVAL}"
  done
}

if [[ "${BACKGROUND}" == "1" ]]; then
  nohup bash -lc "$(declare -f watch_loop); ROOT_DIR='${ROOT_DIR}' LOG_DIR='${LOG_DIR}' STAGE1_SAVE_DIR='${STAGE1_SAVE_DIR}' STAGE1_FINAL_STEP='${STAGE1_FINAL_STEP}' STAGE1_FINAL_DIR='${STAGE1_FINAL_DIR}' STAGE1_FINAL_MODEL='${STAGE1_FINAL_MODEL}' STAGE2_SCRIPT='${STAGE2_SCRIPT}' CHECK_INTERVAL='${CHECK_INTERVAL}' WATCH_LOG='${WATCH_LOG}' LOCK_FILE='${LOCK_FILE}' watch_loop" > "${WATCH_LOG}" 2>&1 < /dev/null &
  echo "[watch] background pid: $!"
  echo "[watch] log: ${WATCH_LOG}"
  echo "[watch] tail: tail -f \"${WATCH_LOG}\""
else
  watch_loop 2>&1 | tee -a "${WATCH_LOG}"
fi
