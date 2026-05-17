---
name: exo-sglang-dgx-spark
description: Use when Codex needs to run exo with a Dockerized SGLang backend on NVIDIA DGX Spark, especially for openai/gpt-oss-20b. Covers isolated repo-local runtime setup, Docker launch, placement, smoke tests, benchmarking, and troubleshooting.
---

# Exo SGLang DGX Spark

Use this skill when asked to run exo on NVIDIA DGX Spark with the `Sglang` backend. Keep all mutable runtime state under the repo-local ignored `.codex/` directory.

## Defaults

- Python venv: `.codex/venvs/sglang-dgxspark`
- Exo config/data/cache: `.codex/xdg-config`, `.codex/xdg-data`, `.codex/xdg-cache`
- Model cache: `.codex/xdg-data/exo/models`
- Hugging Face cache for the SGLang container: `.codex/huggingface`
- GPT-OSS tokenizer cache: `.codex/tiktoken_encodings`
- SGLang Docker image: `nvcr.io/nvidia/sglang:26.02-py3`
- Model: `openai/gpt-oss-20b`

## Key Facts

- exo owns discovery, placement, API routing, downloads, and worker orchestration.
- SGLang owns CUDA serving and NCCL/tensor-parallel execution.
- Start exo with `EXO_SGLANG_LAUNCH_CMD=$PWD/scripts/launch_sglang_dgx_spark_docker.sh`.
- GPT-OSS-20B is MXFP4. Keep `EXO_SGLANG_QUANTIZATION` unset. Use `modelopt_fp4` only for NVIDIA NVFP4 models.
- For GPT-OSS, use `EXO_SGLANG_ATTENTION_BACKEND=triton` and add `EXO_SGLANG_EXTRA_ARGS="--reasoning-parser gpt-oss --tool-call-parser gpt-oss"`.

## Setup

```bash
mkdir -p .codex/venvs .codex/xdg-config .codex/xdg-data .codex/xdg-cache
uv venv .codex/venvs/sglang-dgxspark --python 3.13
source .codex/venvs/sglang-dgxspark/bin/activate
uv sync --active --extra mlx-cuda13 --group dev
```

Prepare GPT-OSS tokenizer assets:

```bash
mkdir -p .codex/tiktoken_encodings
curl -L -o .codex/tiktoken_encodings/o200k_base.tiktoken \
  https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
curl -L -o .codex/tiktoken_encodings/cl100k_base.tiktoken \
  https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
```

Validate Docker from the current shell without changing persistent session config:

```bash
sg docker -c 'docker run --rm --gpus all nvcr.io/nvidia/sglang:26.02-py3 nvidia-smi'
```

## Start exo

Run exo from the isolated venv. The `sg docker -c` wrapper gives the exo worker access to Docker for launching SGLang.

```bash
sg docker -c 'bash -lc "
  cd /home/vmailthody/work/exo
  source .codex/venvs/sglang-dgxspark/bin/activate
  export XDG_CONFIG_HOME=/home/vmailthody/work/exo/.codex/xdg-config
  export XDG_DATA_HOME=/home/vmailthody/work/exo/.codex/xdg-data
  export XDG_CACHE_HOME=/home/vmailthody/work/exo/.codex/xdg-cache
  export EXO_DASHBOARD_DIR=/home/vmailthody/work/exo/dashboard/static
  export EXO_LIBP2P_NAMESPACE=dgx-spark-sglang-e2e
  export EXO_SGLANG_LAUNCH_CMD=/home/vmailthody/work/exo/scripts/launch_sglang_dgx_spark_docker.sh
  export EXO_SGLANG_ATTENTION_BACKEND=triton
  export EXO_SGLANG_MEM_FRACTION_STATIC=0.75
  export EXO_SGLANG_EXTRA_ARGS=\"--reasoning-parser gpt-oss --tool-call-parser gpt-oss\"
  export TIKTOKEN_ENCODINGS_BASE=/home/vmailthody/work/exo/.codex/tiktoken_encodings
  unset EXO_SGLANG_QUANTIZATION
  uv run --active exo --force-master --libp2p-port 52416 -v
"'
```

## Place GPT-OSS-20B

```bash
curl -sS http://localhost:52415/state/nodeBackends | jq

curl -sS "http://localhost:52415/instance/previews?model_id=openai/gpt-oss-20b" \
  | jq '.previews[] | select(.instance_meta == "Sglang")'
```

The benchmark can create the placement automatically:

```bash
source .codex/venvs/sglang-dgxspark/bin/activate
XDG_CONFIG_HOME=/home/vmailthody/work/exo/.codex/xdg-config \
XDG_DATA_HOME=/home/vmailthody/work/exo/.codex/xdg-data \
XDG_CACHE_HOME=/home/vmailthody/work/exo/.codex/xdg-cache \
uv run --active python bench/exo_bench.py \
  --model openai/gpt-oss-20b \
  --force-download \
  --instance-meta sglang \
  --sharding tensor \
  --min-nodes 1 \
  --max-nodes 1 \
  --pp 512 \
  --tg 128 \
  --repeat 1 \
  --warmup 1 \
  --concurrency 1 \
  --settle-timeout 180 \
  --timeout 7200 \
  --json-out bench/results_sglang_gpt_oss_20b_dgx_spark.json
```

## Smoke Test

```bash
curl -N -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "Reasoning: medium. You are concise."},
      {"role": "user", "content": "Give a one-paragraph DGX Spark SGLang health summary."}
    ],
    "stream": true,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 256
  }'
```

## Troubleshooting

- Docker permission denied: use `sg docker -c '...'`; the user may be in the docker group even if the current login session has stale groups.
- No `Sglang` placement: confirm `pynvml` imports in the isolated venv and `/state/nodeBackends` includes `SglangCuda`.
- Hugging Face auth failure: export `HF_TOKEN` before starting exo; the Docker launcher forwards it.
- GPT-OSS tokenization error: verify `.codex/tiktoken_encodings` contains `o200k_base.tiktoken` and `cl100k_base.tiktoken`.
- Quantization error: ensure `EXO_SGLANG_QUANTIZATION` is unset for `openai/gpt-oss-20b`.
- Attention backend error: GPT-OSS in the NVIDIA SGLang container rejects `flashinfer`; use `EXO_SGLANG_ATTENTION_BACKEND=triton`.
- SGLang exits during warmup: inspect `.codex/xdg-cache/exo/exo_log/runner_log/` and `.codex/xdg-cache/exo/exo_log/exo.log`.
- OOM or UMA pressure: lower `EXO_SGLANG_MEM_FRACTION_STATIC`, close other GPU users, or flush host cache with `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`.

## References

- NVIDIA DGX Spark SGLang playbook: https://build.nvidia.com/spark/sglang/overview
- GPT-OSS on DGX Spark with SGLang: https://www.lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark/
- SGLang GPT-OSS usage: https://docs.sglang.io/docs/basic_usage/gpt_oss
