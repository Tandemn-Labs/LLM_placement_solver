#!/bin/bash

python -m llm_advisor.cli --model llama-70b --gpu-pool config/gpu_pool.csv --input-len 2048 --output-len 512 --api-key "$1"