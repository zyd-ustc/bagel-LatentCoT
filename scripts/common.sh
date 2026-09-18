#!/usr/bin/env bash

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

split_paths() {
  local raw=${1:-}
  raw=${raw//:/ }
  read -r -a SPLIT_PATHS <<< "$raw"
}

run_python_module() {
  local nproc_per_node=$1
  local master_port=$2
  local module=$3
  shift 3

  local nnodes=${NNODES:-1}
  local node_rank=${NODE_RANK:-0}
  local master_addr=${MASTER_ADDR:-127.0.0.1}

  if [[ "$nproc_per_node" -gt 1 || "$nnodes" -gt 1 ]]; then
    torchrun \
      --nnodes="$nnodes" \
      --node_rank="$node_rank" \
      --nproc_per_node="$nproc_per_node" \
      --master_addr="$master_addr" \
      --master_port="$master_port" \
      -m "$module" "$@"
    return 0
  fi
  python -m "$module" "$@"
}

require_value() {
  local value=$1
  local message=$2
  if [[ -z "$value" ]]; then
    echo "$message" >&2
    exit 1
  fi
}

limit_to_single_gpu_without_deepspeed() {
  if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
    return 0
  fi
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
    return 0
  fi
  if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
  fi
}
