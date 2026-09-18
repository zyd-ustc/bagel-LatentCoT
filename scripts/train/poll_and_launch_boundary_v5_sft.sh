#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/private/yida_workspace/bagel-LatentCoT
PYTHON_BIN=/private/software/conda/envs/lcot/bin/python
CONFIG_PATH=configs/training/loop_sft.yaml
DATA_PATH=/private/yida_workspace/data/CORT-V4/cort_sft_consis_v4/meta/training/cort_v4_train_unicot_fixed_v2.jsonl
MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT
PREFLIGHT_DIR=/private/yida_workspace/outputs/loop_sft_boundary_10_18_format_v6_preflight_20470
FORMAL_DIR=/private/yida_workspace/outputs/loop_sft_boundary_10_18_format_v6
MIN_FREE_MIB=${MIN_FREE_MIB:-55000}
POLL_SECONDS=${POLL_SECONDS:-60}
EXPECTED_GPUS=8

mkdir -p "${PREFLIGHT_DIR}" "${FORMAL_DIR}"
exec > >(tee -a "${FORMAL_DIR}/poller.log") 2>&1

cd "${PROJECT_DIR}"
echo "$(date -Is) poller started: min_free=${MIN_FREE_MIB}MiB gpus=${EXPECTED_GPUS}"

while true; do
    mapfile -t FREE_MIB < <(
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    )
    ready=1
    if [[ ${#FREE_MIB[@]} -lt ${EXPECTED_GPUS} ]]; then
        ready=0
    else
        for ((gpu = 0; gpu < EXPECTED_GPUS; gpu++)); do
            value=${FREE_MIB[$gpu]//[[:space:]]/}
            if [[ ! ${value} =~ ^[0-9]+$ ]] || (( value < MIN_FREE_MIB )); then
                ready=0
                break
            fi
        done
    fi
    echo "$(date -Is) free_mib=${FREE_MIB[*]:-unavailable} ready=${ready}"
    if (( ready == 1 )); then
        break
    fi
    sleep "${POLL_SECONDS}"
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${PREFLIGHT_DIR}/loop_adapter_step_0000001.safetensors" ]]; then
    echo "$(date -Is) starting one-GPU preflight"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" \
        scripts/train/bagel_loop_sft_train.py \
        --config "${CONFIG_PATH}" \
        --model-path "${MODEL_PATH}" \
        --data-path "${DATA_PATH}" \
        --output-dir "${PREFLIGHT_DIR}" \
        --sample-size 1 \
        --max-steps 1 \
        --fixed-timestep 0.80 \
        --gradient-accumulation-steps 1 \
        --num-workers 0 \
        2>&1 | tee "${PREFLIGHT_DIR}/launcher.log"
    echo "$(date -Is) preflight passed"
fi

echo "$(date -Is) starting formal 8-GPU SFT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${EXPECTED_GPUS}" \
    scripts/train/bagel_loop_sft_train.py \
    --config "${CONFIG_PATH}" \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${FORMAL_DIR}" \
    2>&1 | tee "${FORMAL_DIR}/launcher.log"
echo "$(date -Is) formal SFT completed"
