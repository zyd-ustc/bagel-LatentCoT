#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/private/yida_workspace/bagel-LatentCoT
LCOT_PYTHON=/private/software/conda/envs/lcot/bin/python
REWARD_PYTHON=/private/software/conda/envs/eval/bin/python
MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT
VLM_PATH=/private/yida_workspace/models/Qwen3-VL-8B-Instruct
FORMAT_ADAPTER=/private/yida_workspace/outputs/loop_sft_boundary_10_18_format_v6/loop_adapter_step_0000100.safetensors
HELD_OUT=${PROJECT_DIR}/experiments/data/geneval2_hard_heldout.jsonl
ZERO_SHOT_OUT=/private/yida_workspace/outputs/bagel_geneval2_hard_zero_shot_v1
PAIR_OUT=/private/yida_workspace/outputs/bagel_loop_format_v6_quality_probe
RM_OUT=/private/yida_workspace/outputs/bagel_loop_format_v6_quality_probe/dina_scores.json
RL_OUT=/private/yida_workspace/outputs/bagel_loop_grpo_geneval2_hard_v6_smoke
PIPELINE_OUT=/private/yida_workspace/outputs/bagel_loop_rl_v6_pipeline
POLL_SECONDS=${POLL_SECONDS:-30}

mkdir -p "${PIPELINE_OUT}" "${ZERO_SHOT_OUT}" "${PAIR_OUT}" "${RL_OUT}"
exec > >(tee -a "${PIPELINE_OUT}/launcher.log") 2>&1
cd "${PROJECT_DIR}"

echo "$(date -Is) waiting for format adapter and Qwen3-VL"
while [[ ! -f "${FORMAT_ADAPTER}" || ! -f "${VLM_PATH}/config.json" ]]; do
    echo "$(date -Is) format=$([[ -f ${FORMAT_ADAPTER} ]] && echo ready || echo wait) vlm=$([[ -f ${VLM_PATH}/config.json ]] && echo ready || echo wait)"
    sleep "${POLL_SECONDS}"
done
while pgrep -f "bagel_loop_sft_train.py.*loop_sft_boundary_10_18_format_v6" >/dev/null; do
    echo "$(date -Is) waiting for format workers to exit"
    sleep "${POLL_SECONDS}"
done

if ! curl -fsS http://127.0.0.1:18086 >/dev/null 2>&1; then
    echo "$(date -Is) starting GenEval2 Soft-TIFA on GPU 7"
    CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 "${REWARD_PYTHON}" \
        scripts/evaluate/serve_geneval2_soft_tifa.py \
        --model-path "${VLM_PATH}" --device cuda:0 --port 18086 \
        > "${PIPELINE_OUT}/geneval2_server.log" 2>&1 &
    echo $! > "${PIPELINE_OUT}/geneval2_server.pid"
fi
for _ in $(seq 1 60); do
    curl -fsS http://127.0.0.1:18086 >/dev/null 2>&1 && break
    sleep 5
done
curl -fsS http://127.0.0.1:18086 >/dev/null

if [[ ! -f "${ZERO_SHOT_OUT}/report.json" ]]; then
    echo "$(date -Is) starting frozen BAGEL zero-shot on 32 atomicity-10 prompts"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${LCOT_PYTHON}" \
        scripts/evaluate/bagel_geneval2_hard_zero_shot.py \
        --model-path "${MODEL_PATH}" \
        --benchmark-data "${HELD_OUT}" \
        --output-dir "${ZERO_SHOT_OUT}" \
        --device cuda:0 --min-atom-count 10 --max-prompts 20 \
        --num-steps 30 \
        2>&1 | tee "${ZERO_SHOT_OUT}/generate.log"
fi
if [[ ! -f "${ZERO_SHOT_OUT}/score_lists.json" ]]; then
    echo "$(date -Is) scoring zero-shot images"
    "${LCOT_PYTHON}" scripts/evaluate/score_geneval2_server.py \
        --benchmark-data "${ZERO_SHOT_OUT}/benchmark_hard.jsonl" \
        --image-paths "${ZERO_SHOT_OUT}/image_paths.json" \
        --output "${ZERO_SHOT_OUT}/score_lists.json" \
        --server-url http://127.0.0.1:18086 \
        2>&1 | tee "${ZERO_SHOT_OUT}/score.log"
    "${LCOT_PYTHON}" scripts/evaluate/geneval2_report.py \
        --benchmark-data "${ZERO_SHOT_OUT}/benchmark_hard.jsonl" \
        --run "bagel_depth1=${ZERO_SHOT_OUT}/score_lists.json" \
        --output-dir "${ZERO_SHOT_OUT}/report" \
        2>&1 | tee "${ZERO_SHOT_OUT}/report.log"
fi

if [[ ! -f "${PAIR_OUT}/report.json" ]]; then
    echo "$(date -Is) generating paired format-adapter candidates"
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "${LCOT_PYTHON}" \
        scripts/evaluate/bagel_loop_generate.py \
        --model-path "${MODEL_PATH}" --adapter "${FORMAT_ADAPTER}" \
        --output-dir "${PAIR_OUT}" \
        --prompt 'three elephants on top of seven yellow violins in front of five green bears' \
        --seeds 42 43 44 45 --device cuda:0 --num-steps 30 \
        2>&1 | tee "${PAIR_OUT}/generate.log"
fi
if [[ ! -f "${RM_OUT}" ]]; then
    echo "$(date -Is) auditing DiNa-LRM stability on paired BAGEL candidates"
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/private/yida_workspace/diffusion-rm \
        PYTHONUNBUFFERED=1 "${LCOT_PYTHON}" \
        /private/yida_workspace/diffusion-rm/scripts/score_bagel_candidates.py \
        --config /private/yida_workspace/diffusion-rm/config/flux/thurstone-19layer-hpdv3.yaml \
        --checkpoint /private/yida_workspace/outputs/dina_flux_rm/DiNa-FLUX-HPDv3-19layers_2026.09.12_08.55/checkpoints/epoch_001 \
        --reports "${PAIR_OUT}/report.json" --output "${RM_OUT}" \
        --device cuda:0 --noise-levels 0.05 0.10 --noise-seeds 17 42 123 \
        2>&1 | tee "${PAIR_OUT}/dina.log"
fi

echo "$(date -Is) starting one-step GenEval2 + DiNa-LRM RL smoke"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${LCOT_PYTHON}" \
    scripts/train/bagel_loop_grpo_train.py \
    --config configs/training/loop_grpo.yaml \
    --adapter-path "${FORMAT_ADAPTER}" --output-dir "${RL_OUT}" --max-steps 1 \
    2>&1 | tee "${RL_OUT}/launcher.log"
echo "$(date -Is) RL smoke completed"
