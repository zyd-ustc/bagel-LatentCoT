#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=${1:?usage: train_interaction_npu.sh sft|rl OUTPUT --parent|--checkpoint|--resume PATH --train PATH --val PATH [options]}
RUN=${2:?output required}; shift 2
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
ARGS=(-m qwen_latent_cot.interaction.cli "$STAGE" --model "$MODEL" --device npu:0
      --config "$ROOT/configs/interaction_$STAGE.json" --output "$RUN"
      --wandb-mode "${WANDB_MODE:-online}")
case "$STAGE" in
 sft) NPROC=${NPROC_PER_NODE:-16}
      export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15} ;;
 rl) NPROC=${NPROC_PER_NODE:-14}
     JUDGE_PYTHON=${JUDGE_PYTHON:-/root/venvs/qwen3vl-npu/bin/python}
     JUDGE=${JUDGE_PATH:-/data/zyd_workspace/models/Qwen3-VL-8B-Instruct}
     SOURCE=${GENEVAL2_SOURCE:-/root/npu-eval-tools/GenEval2/evaluation.py}
     mkdir -p "$RUN"
     QUEUE=$(mktemp -d "$RUN/judges_XXXXXXXX")
     PIDS=()
     cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
     trap cleanup EXIT INT TERM
     for WORKER in 0 1; do
       DEVICE=$((14+WORKER))
       ASCEND_RT_VISIBLE_DEVICES=$DEVICE "$JUDGE_PYTHON" -m qwen_latent_cot.interaction.rewards \
         --queue "$QUEUE" --worker "$WORKER" --judge "$JUDGE" --official-source "$SOURCE" \
         >"$QUEUE/judge_$WORKER.log" 2>&1 &
       PIDS+=("$!")
     done
     for ((ATTEMPT=0; ATTEMPT<600; ATTEMPT++)); do
       [[ ! -f "$QUEUE/judge_0/ready.json" || ! -f "$QUEUE/judge_1/ready.json" ]] || break
       for PID in "${PIDS[@]}"; do
         kill -0 "$PID" 2>/dev/null || { cat "$QUEUE"/*.log >&2; exit 1; }
       done
       sleep 1
     done
     [[ -f "$QUEUE/judge_0/ready.json" && -f "$QUEUE/judge_1/ready.json" ]] || { echo 'Judge startup timeout' >&2; exit 1; }
     ARGS+=(--judge-queue "$QUEUE")
     export ASCEND_RT_VISIBLE_DEVICES=${POLICY_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13} ;;
 *) echo 'stage must be sft or rl' >&2; exit 1 ;;
esac
cd "$ROOT"
if [[ "$NPROC" == 1 ]]; then
 "$PYTHON" "${ARGS[@]}" "$@"
else
 "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC" \
   "${ARGS[@]}" --distributed "$@"
fi
