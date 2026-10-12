#!/usr/bin/env bash
# Warm start the response100 writer; train with larger text and query pools.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export DIAGNOSTIC_DATA=${DIAGNOSTIC_DATA:-/data/zyd_workspace/bagel-LatentCoT/datasets/c1_response_text2192.jsonl}
export QUERY_DATA=${QUERY_DATA:-/data/zyd_workspace/bagel-LatentCoT/datasets/memory_response_queries132.jsonl}
export DIAGNOSTIC_CONFIG=${DIAGNOSTIC_CONFIG:-$ROOT/configs/c1_response_expand.json}
export C1_C=${C1_C:-/root/outputs/c1_response_20261012_134848/last.pt}
exec bash "$ROOT/scripts/c1_response_npu.sh" "${1:?usage: bash scripts/c1_response_expand_npu.sh fresh_output_directory}"
