<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/imgs/exo-logo-black-bg.jpg">
  <img alt="exo logo" src="/docs/imgs/exo-logo-transparent.png" width="50%" height="50%">
</picture>

exo: Run frontier AI locally. Maintained by [exo labs](https://x.com/exolabs).

<p align="center">
  <a href="https://discord.gg/TJ4P57arEm" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://x.com/exolabs" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/twitter/follow/exolabs?style=social" alt="X"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0.html" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-Apache2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

</div>

---

exo connects all your devices into an AI cluster. Not only does exo enable running models larger than would fit on a single device, but with [day-0 support for RDMA over Thunderbolt](https://x.com/exolabs/status/2001817749744476256?s=20), makes models run faster as you add more devices.

## Features

- **Automatic Device Discovery**: Devices running exo automatically discover each other - no manual configuration.
- **RDMA over Thunderbolt**: exo ships with [day-0 support for RDMA over Thunderbolt 5](https://x.com/exolabs/status/2001817749744476256?s=20), enabling 99% reduction in latency between devices.
- **Topology-Aware Auto Parallel**: exo figures out the best way to split your model across all available devices based on a realtime view of your device topology. It takes into account device resources and network latency/bandwidth between each link.
- **Tensor Parallelism**: exo supports sharding models, for up to 1.8x speedup on 2 devices and 3.2x speedup on 4 devices.
- **MLX Support**: exo uses [MLX](https://github.com/ml-explore/mlx) as an inference backend and [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) for distributed communication.
- **NVIDIA DGX Spark Support**: exo can place text-generation workloads on NVIDIA CUDA workers and launch [SGLang](https://github.com/sgl-project/sglang) for single-node or multi-worker tensor-parallel serving.
- **Multiple API Compatibility**: Compatible with OpenAI Chat Completions API, Claude Messages API, OpenAI Responses API, and Ollama API - use your existing tools and clients.
- **Custom Model Support**: Load custom models from HuggingFace hub to expand the range of available models.

## Dashboard

exo includes a built-in dashboard for managing your cluster and chatting with models.

<p align="center">
  <img src="docs/imgs/dashboard-cluster-view.png" alt="exo dashboard - cluster view showing 4 x M3 Ultra Mac Studio with DeepSeek v3.1 and Kimi-K2-Thinking loaded" width="80%" />
</p>
<p align="center"><em>4 × 512GB M3 Ultra Mac Studio running DeepSeek v3.1 (8-bit) and Kimi-K2-Thinking (4-bit)</em></p>

## Benchmarks

<details>
  <summary>Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-1-qwen3-235b.jpeg" alt="Benchmark - Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

<details>
  <summary>DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-2-deepseek-3.1-671b.jpeg" alt="Benchmark - DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

<details>
  <summary>Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-3-kimi-k2-thinking.jpeg" alt="Benchmark - Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

---

## Quick Start

Devices running exo automatically discover each other, without needing any manual configuration. Each device provides an API and a dashboard for interacting with your cluster (runs at `http://localhost:52415`).

There are two ways to run exo:

### Run from Source (macOS)

If you have [Nix](https://nixos.org/) installed, you can skip most of the steps below and run exo directly:

```bash
nix run .#exo
```

**Note:** To accept the Cachix binary cache (and avoid the Xcode Metal ToolChain), add to `/etc/nix/nix.conf`:
```
trusted-users = root    (or your username)
experimental-features = nix-command flakes
```
Then restart the Nix daemon: `sudo launchctl kickstart -k system/org.nixos.nix-daemon`

**Prerequisites:**
- [Xcode](https://developer.apple.com/xcode/) (provides the Metal ToolChain required for MLX compilation)
- [brew](https://github.com/Homebrew/brew) (for simple package management on macOS)

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- [uv](https://github.com/astral-sh/uv) (for Python dependency management)
- [node](https://github.com/nodejs/node) (for building the dashboard)

  ```bash
  brew install uv node
  ```
- [rust](https://github.com/rust-lang/rustup) (to build Rust bindings, nightly for now)

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup toolchain install nightly
  ```
- [macmon](https://github.com/vladkens/macmon) (for hardware monitoring on Apple Silicon)

  Install the pinned fork revision used by this repo instead of Homebrew `macmon`.
  Homebrew `macmon 0.6.1` still crashes on Apple M5.

  ```bash
  cargo install --git https://github.com/vladkens/macmon \
    --rev a1cd06b6cc0d5e61db24fd8832e74cd992097a7d \
    macmon \
    --force
  ```

Clone the repo, build the dashboard, and run exo:

```bash
# Clone exo
git clone https://github.com/exo-explore/exo

# Build dashboard
cd exo/dashboard && npm install && npm run build && cd ..

# Run exo
uv run exo
```

This starts the exo dashboard and API at http://localhost:52415/


*Please view the section on RDMA to enable this feature on MacOS >=26.2!*


### Run from Source (Linux)

**Prerequisites:**

- [uv](https://github.com/astral-sh/uv) (for Python dependency management)
- [node](https://github.com/nodejs/node) (for building the dashboard) - version 18 or higher
- [rust](https://github.com/rust-lang/rustup) (to build Rust bindings, nightly for now)

**Installation methods:**

**Option 1: Using system package manager (Ubuntu/Debian example):**
```bash
# Install Node.js and npm
sudo apt update
sudo apt install nodejs npm

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust (using rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

**Option 2: Using Homebrew on Linux (if preferred):**
```bash
# Install Homebrew on Linux
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install uv node

# Install Rust (using rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

**Note:** The `macmon` package is macOS-only and not required for Linux.

Clone the repo, build the dashboard, and run exo:

```bash
# Clone exo
git clone https://github.com/exo-explore/exo

# Build dashboard
cd exo/dashboard && npm install && npm run build && cd ..

# Run exo
uv run exo
```

This starts the exo dashboard and API at http://localhost:52415/

**Important note for Linux users:** Generic Linux acceleration support is still hardware-specific. NVIDIA DGX Spark systems can use the SGLang backend described below.

#### Run on NVIDIA DGX Spark with SGLang

The DGX Spark path keeps exo responsible for discovery, placement, API compatibility, and task orchestration, while SGLang runs the CUDA serving process on each selected worker. In a multi-worker placement, exo creates one `SglangInstance`, assigns one rank per DGX Spark node, starts SGLang on every rank, and sends chat-completion requests to rank 0. SGLang handles CUDA/NCCL tensor parallel execution across the workers.

**Prerequisites:**

- NVIDIA driver and CUDA visible to `nvidia-smi`
- Python 3.13 through `uv`
- SGLang installed in the same environment that runs exo, or a custom launcher set with `EXO_SGLANG_LAUNCH_CMD`
- `nvidia-ml-py`, installed by the `mlx-cuda13` extra, so exo can detect CUDA workers and advertise `SglangCuda`

Install the exo CUDA dependencies and verify that SGLang is importable:

```bash
uv sync --extra mlx-cuda13

# Install SGLang for your CUDA/PyTorch build using the upstream
# SGLang installation docs: https://docs.sglang.io/get_started/install.html
# At minimum, this command must work:
uv run python -m sglang.launch_server --help
```

Start exo on every DGX Spark. Use the same namespace on all nodes so only these workers join the same cluster.

```bash
export EXO_LIBP2P_NAMESPACE=dgx-spark-sglang
export EXO_SGLANG_ATTENTION_BACKEND=flashinfer
export EXO_SGLANG_MEM_FRACTION_STATIC=0.75

# Required for NVIDIA NVFP4/FP4 model checkpoints.
# Leave unset for bf16/fp16/int8 models unless SGLang requires a quantization flag.
export EXO_SGLANG_QUANTIZATION=modelopt_fp4

# First DGX Spark: keep the API/dashboard stable on this node.
uv run --extra mlx-cuda13 exo --force-master --libp2p-port 52416
```

On every additional DGX Spark, run the same command without `--force-master`:

```bash
export EXO_LIBP2P_NAMESPACE=dgx-spark-sglang
export EXO_SGLANG_ATTENTION_BACKEND=flashinfer
export EXO_SGLANG_MEM_FRACTION_STATIC=0.75
export EXO_SGLANG_QUANTIZATION=modelopt_fp4

uv run --extra mlx-cuda13 exo --libp2p-port 52416
```

If multicast discovery is unavailable on your network, start the first node with a fixed libp2p port as shown above and pass its libp2p multiaddr to the other nodes with `EXO_BOOTSTRAP_PEERS` or `--bootstrap-peers`.

Once all workers are visible in the dashboard at `http://<master-node-ip>:52415`, create an SGLang placement from the master node. Use `min_nodes: 1` for a single DGX Spark or increase it to the number of DGX Spark workers you want in the tensor-parallel group.

```bash
curl -X POST http://localhost:52415/place_instance \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "<huggingface-model-id>",
    "sharding": "Tensor",
    "instance_meta": "Sglang",
    "min_nodes": 2
  }'
```

Preview placements first if you want to confirm exo sees `Sglang` as a valid option:

```bash
curl "http://localhost:52415/instance/previews?model_id=<huggingface-model-id>" \
  | jq '.previews[] | select(.instance_meta == "Sglang")'
```

Send requests through the normal OpenAI-compatible endpoint:

```bash
curl -N -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<huggingface-model-id>",
    "messages": [
      {"role": "user", "content": "Write a short DGX Spark status check."}
    ],
    "stream": true,
    "max_tokens": 128
  }'
```

Useful SGLang overrides:

- `EXO_SGLANG_LAUNCH_CMD`: launcher prefix, default `python -m sglang.launch_server`
- `EXO_SGLANG_EXTRA_ARGS`: appended verbatim to the SGLang launch command
- `EXO_SGLANG_CLIENT_HOST`: host exo uses to call rank 0 locally, default `127.0.0.1`
- `EXO_SGLANG_STARTUP_TIMEOUT`: seconds to wait for `/health`, default `600`

For model storage, use `EXO_MODELS_DIRS` or `EXO_MODELS_READ_ONLY_DIRS` as described below. The SGLang builder will pass the local exo model path when the checkpoint is already present, otherwise it passes the Hugging Face model ID to SGLang.

**Configuration Options:**

- `--no-worker`: Run exo without the worker component. Useful for coordinator-only nodes that handle networking and orchestration but don't execute inference tasks. This is helpful for machines without sufficient GPU resources but with good network connectivity.

  ```bash
  uv run exo --no-worker
  ```

**File Locations (Linux):**

exo follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) on Linux:

- **Configuration files**: `~/.config/exo/` (or `$XDG_CONFIG_HOME/exo/`)
- **Data files**: `~/.local/share/exo/` (or `$XDG_DATA_HOME/exo/`)
- **Cache files**: `~/.cache/exo/` (or `$XDG_CACHE_HOME/exo/`)
- **Log files**: `~/.cache/exo/exo_log/` (with automatic log rotation)
- **Custom model cards**: `~/.local/share/exo/custom_model_cards/`

You can override these locations by setting the corresponding XDG environment variables.

### macOS App

exo ships a macOS app that runs in the background on your Mac.

<img src="docs/imgs/macos-app-one-macbook.png" alt="exo macOS App - running on a MacBook" width="35%" />

The macOS app requires macOS Tahoe 26.2 or later.

Download the latest build here: [EXO-latest.dmg](https://assets.exolabs.net/EXO-latest.dmg).

The app will ask for permission to modify system settings and install a new Network profile. Improvements to this are being worked on.

**Custom Namespace for Cluster Isolation:**

The macOS app includes a custom namespace feature that allows you to isolate your exo cluster from others on the same network. This is configured through the `EXO_LIBP2P_NAMESPACE` setting:

- **Use cases**:
  - Running multiple separate exo clusters on the same network
  - Isolating development/testing clusters from production clusters
  - Preventing accidental cluster joining

- **Configuration**: Access this setting in the app's Advanced settings (or set the `EXO_LIBP2P_NAMESPACE` environment variable when running from source)

The namespace is logged on startup for debugging purposes.

#### Uninstalling the macOS App

The recommended way to uninstall is through the app itself: click the menu bar icon → Advanced → Uninstall. This cleanly removes all system components.

If you've already deleted the app, you can run the standalone uninstaller script:

```bash
sudo ./app/EXO/uninstall-exo.sh
```

This removes:
- Network setup LaunchDaemon
- Network configuration script
- Log files
- The "exo" network location

**Note:** You'll need to manually remove EXO from Login Items in System Settings → General → Login Items.

---

### Enabling RDMA on macOS

RDMA is a new capability added to macOS 26.2. It works on any Mac with Thunderbolt 5 (M4 Pro Mac Mini, M4 Max Mac Studio, M4 Max MacBook Pro, M3 Ultra Mac Studio).

Please refer to the caveats for immediate troubleshooting.

To enable RDMA on macOS, follow these steps:

1. Shut down your Mac.
2. Hold down the power button for 10 seconds until the boot menu appears.
3. Select "Options" to enter Recovery mode.
4. When the Recovery UI appears, open the Terminal from the Utilities menu.
5. In the Terminal, type:
   ```
   rdma_ctl enable
   ```
   and press Enter.
6. Reboot your Mac.

After that, RDMA will be enabled in macOS and exo will take care of the rest.

**Important Caveats**

1. Devices that wish to be part of an RDMA cluster must be connected to all other devices in the cluster.
2. The cables must support TB5.
3. On a Mac Studio, you cannot use the Thunderbolt 5 port next to the Ethernet port.
4. If running from source, please use the script found at `tmp/set_rdma_network_config.sh`, which will disable Thunderbolt Bridge and set dhcp on each RDMA port.
5. RDMA ports may be unable to discover each other on different versions of MacOS. Please ensure that OS versions match exactly (even beta version numbers) on all devices.

---

## Environment Variables

exo supports several environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `EXO_DEFAULT_MODELS_DIR` | Default directory for model downloads and caches. Always first in the writable dirs list. | `~/.local/share/exo/models` (Linux) or `~/.exo/models` (macOS) |
| `EXO_MODELS_DIRS` | Colon-separated additional writable directories for model downloads. Checked in order after the default; first with enough free space is used. | None |
| `EXO_MODELS_READ_ONLY_DIRS` | Colon-separated read-only directories to search for pre-downloaded models (e.g., NFS mounts, shared storage). Models here cannot be deleted. | None |
| `EXO_OFFLINE` | Run without internet connection (uses only local models) | `false` |
| `EXO_ENABLE_IMAGE_MODELS` | Enable image model support | `false` |
| `EXO_LIBP2P_NAMESPACE` | Custom namespace for cluster isolation | None |
| `EXO_FAST_SYNCH` | Control MLX_METAL_FAST_SYNCH behavior (for JACCL backend) | Auto |
| `EXO_TRACING_ENABLED` | Enable distributed tracing for performance analysis | `false` |
| `EXO_SGLANG_LAUNCH_CMD` | Command prefix used to start SGLang for `Sglang` instances | `python -m sglang.launch_server` |
| `EXO_SGLANG_HOST` | Host address SGLang binds inside each worker process | `0.0.0.0` |
| `EXO_SGLANG_CLIENT_HOST` | Host address exo uses when calling the local rank-0 SGLang server | `127.0.0.1` |
| `EXO_SGLANG_ATTENTION_BACKEND` | SGLang attention backend | `flashinfer` |
| `EXO_SGLANG_MEM_FRACTION_STATIC` | SGLang static memory fraction | `0.75` |
| `EXO_SGLANG_QUANTIZATION` | Optional SGLang quantization argument, such as `modelopt_fp4` for NVIDIA FP4 checkpoints | Auto for NVFP4 cards, otherwise unset |
| `EXO_SGLANG_EXTRA_ARGS` | Extra arguments appended to the SGLang launch command | None |
| `EXO_SGLANG_STARTUP_TIMEOUT` | Seconds to wait for SGLang health checks during worker warmup | `600` |
| `EXO_SGLANG_SKIP_HEALTH_WAIT` | Skip SGLang `/health` polling during warmup, mainly for tests/debugging | unset |

**Example usage:**

```bash
# Use pre-downloaded models from NFS mount (read-only)
EXO_MODELS_READ_ONLY_DIRS=/mnt/nfs/models:/opt/ai-models uv run exo

# Download models to an external SSD (falls back to default dir if full)
EXO_MODELS_DIRS=/Volumes/ExternalSSD/exo-models uv run exo

# Run in offline mode
EXO_OFFLINE=true uv run exo

# Enable image models
EXO_ENABLE_IMAGE_MODELS=true uv run exo

# Use custom namespace for cluster isolation
EXO_LIBP2P_NAMESPACE=my-dev-cluster uv run exo
```

---

### Using the API

exo provides multiple API-compatible interfaces for maximum compatibility with existing tools:

- **OpenAI Chat Completions API** - Compatible with OpenAI clients
- **Claude Messages API** - Compatible with Anthropic's Claude format
- **OpenAI Responses API** - Compatible with OpenAI's Responses format
- **Ollama API** - Compatible with Ollama and tools like OpenWebUI

If you prefer to interact with exo via the API, here is an example creating an instance of a small model (`mlx-community/Llama-3.2-1B-Instruct-4bit`), sending a chat completions request and deleting the instance.

---

**1. Preview instance placements**

The `/instance/previews` endpoint will preview all valid placements for your model.

```bash
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b"
```

Sample response:

```json
{
  "previews": [
    {
      "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "sharding": "Pipeline",
      "instance_meta": "MlxRing",
      "instance": {...},
      "memory_delta_by_node": {"local": 729808896},
      "error": null
    }
    // ...possibly more placements...
  ]
}
```

This will return all valid placements for this model. Pick a placement that you like.
To pick the first one, pipe into `jq`:

```bash
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b" | jq -c '.previews[] | select(.error == null) | .instance' | head -n1
```

---

**2. Create a model instance**

Send a POST to `/instance` with your desired placement in the `instance` field (the full payload must match types as in `CreateInstanceParams`), which you can copy from step 1:

```bash
curl -X POST http://localhost:52415/instance \
  -H 'Content-Type: application/json' \
  -d '{
    "instance": {...}
  }'
```


Sample response:

```json
{
  "message": "Command received.",
  "command_id": "e9d1a8ab-...."
}
```

---

**3. Send a chat completion**

Now, make a POST to `/v1/chat/completions` (the same format as OpenAI's API):

```bash
curl -N -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "What is Llama 3.2 1B?"}
    ],
    "stream": true
  }'
```

---

**4. Delete the instance**

When you're done, delete the instance by its ID (find it via `/state` or `/instance` endpoints):

```bash
curl -X DELETE http://localhost:52415/instance/YOUR_INSTANCE_ID
```

### Claude Messages API Compatibility

Use the Claude Messages API format with the `/v1/messages` endpoint:

```bash
curl -N -X POST http://localhost:52415/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 1024,
    "stream": true
  }'
```

### OpenAI Responses API Compatibility

Use the OpenAI Responses API format with the `/v1/responses` endpoint:

```bash
curl -N -X POST http://localhost:52415/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": true
  }'
```

### Ollama API Compatibility

exo supports Ollama API endpoints for compatibility with tools like OpenWebUI:

```bash
# Ollama chat
curl -X POST http://localhost:52415/ollama/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": false
  }'

# List models (Ollama format)
curl http://localhost:52415/ollama/api/tags
```

### Custom Model Loading from HuggingFace

You can add custom models from the HuggingFace hub:

```bash
curl -X POST http://localhost:52415/models/add \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "mlx-community/my-custom-model"
  }'
```

**Security Note:**

Custom models requiring `trust_remote_code` in their configuration must be explicitly enabled (default is false) for security. Only enable this if you trust the model's remote code execution. Models are fetched from HuggingFace and stored locally as custom model cards.

**Other useful API endpoints*:**

- List all models: `curl http://localhost:52415/models`
- List downloaded models only: `curl http://localhost:52415/models?status=downloaded`
- Search HuggingFace: `curl "http://localhost:52415/models/search?query=llama&limit=10"`
- Inspect instance IDs and deployment state: `curl http://localhost:52415/state`

For further details, see:

- API documentation in [docs/api.md](docs/api.md).
- API types and endpoints in [src/exo/master/api.py](src/exo/master/api.py).

---

## Benchmarking

The `exo-bench` tool measures model prefill and token generation speed across different placement configurations. This helps you optimize model performance and validate improvements.

**Prerequisites:**
- Nodes should be running with `uv run exo` before benchmarking
- The tool uses the `/bench/chat/completions` endpoint

**Basic usage:**

```bash
uv run bench/exo_bench.py \
  --model Llama-3.2-1B-Instruct-4bit \
  --pp 128,256,512 \
  --tg 128,256
```

**Key parameters:**

- `--model`: Model to benchmark (short ID or HuggingFace ID)
- `--pp`: Prompt size hints (comma-separated integers)
- `--tg`: Generation lengths (comma-separated integers)
- `--max-nodes`: Limit placements to N nodes (default: 4)
- `--instance-meta`: Filter by `ring`, `jaccl`, or `both` (default: both)
- `--sharding`: Filter by `pipeline`, `tensor`, or `both` (default: both)
- `--repeat`: Number of repetitions per configuration (default: 1)
- `--warmup`: Warmup runs per placement (default: 0)
- `--json-out`: Output file for results (default: bench/results.json)

**Example with filters:**

```bash
uv run bench/exo_bench.py \
  --model Llama-3.2-1B-Instruct-4bit \
  --pp 128,512 \
  --tg 128 \
  --max-nodes 2 \
  --sharding tensor \
  --repeat 3 \
  --json-out my-results.json
```

The tool outputs performance metrics including prompt tokens per second (prompt_tps), generation tokens per second (generation_tps), and peak memory usage for each configuration.

---

## Hardware Accelerator Support

On macOS, exo uses the GPU. On Linux, exo currently runs on CPU. We are working on extending hardware accelerator support. If you'd like support for a new hardware platform, please [search for an existing feature request](https://github.com/exo-explore/exo/issues) and add a thumbs up so we know what hardware is important to the community.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to exo.
