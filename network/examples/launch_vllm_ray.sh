#!/usr/bin/env bash
set -euo pipefail

# Start a Ray cluster containing exactly PIPELINE_STAGES GPU nodes first.
# This command runs once on the Ray head. Each node needs the same pinned image.
: "${HF_REVISION:?set a pinned Hugging Face commit SHA}"
: "${VLLM_API_KEY:?set a private vLLM API key}"
case "${MODEL_SIZE:?set MODEL_SIZE to 14 or 32}" in
  14) MODEL=Qwen/Qwen3-14B; STAGES=2 ;;
  32) MODEL=Qwen/Qwen3-32B; STAGES=4 ;;
  *) echo "MODEL_SIZE must be 14 or 32" >&2; exit 2 ;;
esac

vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --revision "$HF_REVISION" --tokenizer-revision "$HF_REVISION" \
  --tensor-parallel-size "${GPUS_PER_NODE:-1}" \
  --pipeline-parallel-size "$STAGES" \
  --distributed-executor-backend ray \
  --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port "${VLLM_PORT:-8000}"
