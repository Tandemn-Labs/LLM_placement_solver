#!/bin/bash

API_KEY="$1"
PROVIDER="${2:-anthropic}"
LLM_MODEL="$3"

ARGS=(--model llama-70b --gpu-pool config/gpu_pool.csv --input-len 2048 --output-len 512 --api-key "$API_KEY" --provider "$PROVIDER")
if [ -n "$LLM_MODEL" ]; then
  ARGS+=(--llm-model "$LLM_MODEL")
fi

python -m llm_advisor.cli "${ARGS[@]}"


# python -m llm_advisor.cli --model llama-70b --gpu-pool config/gpu_pool.csv --input-len 512 --output-len 8192 --api-key "$1"
