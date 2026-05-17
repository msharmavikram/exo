#!/usr/bin/env bash
set -euo pipefail

repo_root="${EXO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
codex_dir="${EXO_SGLANG_CODEX_DIR:-$repo_root/.codex}"
image="${EXO_SGLANG_DOCKER_IMAGE:-nvcr.io/nvidia/sglang:26.02-py3}"
shm_size="${EXO_SGLANG_DOCKER_SHM_SIZE:-32g}"

hf_cache="${HF_HOME:-$codex_dir/huggingface}"
tiktoken_dir="${TIKTOKEN_ENCODINGS_BASE:-$codex_dir/tiktoken_encodings}"

mkdir -p "$codex_dir" "$hf_cache" "$tiktoken_dir"

model_path=""
previous=""
for arg in "$@"; do
  if [[ "$previous" == "--model-path" ]]; then
    model_path="$arg"
    break
  fi
  previous="$arg"
done

docker_args=(
  run
  --rm
  --gpus all
  --network host
  --ipc=host
  --shm-size "$shm_size"
  -v "$codex_dir:$codex_dir"
  -v "$hf_cache:/root/.cache/huggingface"
  -v "$tiktoken_dir:/tiktoken_encodings:ro"
  -e "HF_HOME=/root/.cache/huggingface"
  -e "TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_args+=(-e "HF_TOKEN=$HF_TOKEN")
fi

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  docker_args+=(-e "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN")
fi

if [[ "$model_path" == /* && -e "$model_path" && "$model_path" != "$codex_dir"* ]]; then
  docker_args+=(-v "$model_path:$model_path:ro")
fi

exec docker "${docker_args[@]}" "$image" python3 -m sglang.launch_server "$@"
