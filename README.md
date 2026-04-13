# federated-inference

Run large LLMs across multiple heterogeneous edge devices using **pipeline parallelism** via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s RPC backend.

Distribute models across a Raspberry Pi, Android phone (Termux), Mac, x86 workstation, Jetson board, or any combination. The **coordinator** (your main machine) automatically distributes model layers to workers proportional to their available memory—weak devices get fewer layers, powerful devices get more.

**Example**: Llama-3.1-70B on 3 devices
```
Workstation (32 GB)   → 50 layers
Raspberry Pi (8 GB)   → 15 layers  
Android Phone (4 GB)  → 5 layers
                        ──────────
                        70 total
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Your Machine (Laptop/Workstation) — the Coordinator         │
│                                                              │
│  FastAPI Server (port 8080)                                 │
│  ↓                                                           │
│  llama-server  ←──  --rpc  worker1:8765,worker2:8765,...    │
│  ↓                                                           │
│  Manages workers via gRPC:                                  │
│    • Health checks every 10 seconds                         │
│    • Starts/stops llama-rpc-server on each worker           │
│    • Monitors free RAM/VRAM to balance layer assignment     │
└─────────────────────────────────────────────────────────────┘
        gRPC (port 50051)
        │      │      │
        ↓      ↓      ↓
    ┌──────┐ ┌──────┐ ┌──────────────┐
    │ Pi   │ │ x86  │ │ Android      │
    │(gRPC)│ │(gRPC)│ │ Termux (gRPC)│
    │      │ │      │ │              │
    │ llama│ │ llama│ │ llama        │
    │-rpc │ │-rpc  │ │-rpc-server   │
    │     │ │      │ │              │
    └──────┘ └──────┘ └──────────────┘
```

---

## Installation

### Prerequisites

- **Python 3.11+**
- **Git**
- **CMake** (auto-installed on macOS via Homebrew if missing)

### Step 1: Install Python Package

On **every device** (coordinator + all workers):

```bash
# Option A: Virtual environment (recommended)
python3 -m venv ~/.venvs/federated
source ~/.venvs/federated/bin/activate

# Option B: Or just use --break-system-packages if you're comfortable
pip install --break-system-packages federated-inference[coordinator]
```

**On workers only** (skip `[coordinator]` extras on weak devices like Termux):
```bash
pip install federated-inference  # minimal install, no uvicorn/FastAPI
```

### Step 2: Build llama.cpp (Required on EVERY device)

```bash
bash scripts/install_llama_cpp.sh
```

This builds `llama-server` and `llama-rpc-server` from llama.cpp source, with optimizations:
- **macOS**: Metal GPU acceleration (M1/M2/M3/M4)
- **Linux + NVIDIA**: CUDA GPU support
- **Linux + AMD**: ROCm GPU support
- **All platforms**: CPU-only fallback

**Progress indicator**: First run takes ~5–15 minutes (compiling C++). Subsequent runs are quick.

---

## Setup Workflow

### A. Get a Model (GGUF File)

Use **LM Studio** (simplest) or any GGUF downloader:

1. Download [LM Studio](https://lmstudio.ai/)
2. Search & download a model (e.g. `Llama-3.2-3B-Instruct`)
3. Find the GGUF file:
   ```
   ~/Library/Application Support/LM Studio/models/
   └── lmstudio-community/Llama-3.2-3B-Instruct-GGUF/
       └── Llama-3.2-3B-Instruct-Q4_K_M.gguf
   ```

**Recommended starter models** (good balance for distributed inference):
- `Llama-3.2-3B-Instruct Q4_K_M` — ~2 GB, runs on anything
- `Llama-3.1-8B-Instruct Q4_K_M` — ~5 GB, fast + good quality
- `Mistral-7B-Instruct Q4_K_M` — ~4 GB, solid general-purpose

### B. Coordinator Setup (on your main machine)

```bash
# Copy and edit config files
cp config/topology.example.yaml topology.yaml
cp config/model.example.yaml model.yaml

# Edit topology.yaml: add your workers' WLAN IPs
# Edit model.yaml: point to your GGUF file
```

**topology.yaml example:**
```yaml
coordinator:
  host: "0.0.0.0"
  port: 8080

workers:
  - id: "raspberry-pi"
    host: "192.168.1.101"      # Find this with: ip addr show wlan0
    grpc_port: 50051
    rpc_port: 8765
    enabled: true
    mem_limit_mb: 0            # auto (use free RAM)

  - id: "x86-workstation"
    host: "192.168.1.102"
    enabled: true
    mem_limit_mb: 0
```

**model.yaml example:**
```yaml
model:
  path: "~/Library/Application Support/LM Studio/models/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  context_length: 4096
  n_gpu_layers: -1             # offload as many as possible
  batch_size: 512
  threads: 4

llama_server_extra_flags:
  - "--flash-attn"
```

### C. Start Workers (on each edge device)

**On Raspberry Pi, Android, x86 workstation, etc:**

```bash
# Activate venv if you made one
source ~/.venvs/federated/bin/activate

# Start the worker
federated-worker start --grpc-port 50051 --grpc-host 0.0.0.0
```

**For Termux on Android**, also ensure the device is on the same WLAN as your coordinator.

### D. Start Coordinator (on your main machine)

```bash
source ~/.venvs/federated/bin/activate
federated-coordinator start --topology config/topology.yaml --model config/model.yaml
```

Watch the logs:
```
[INFO] Worker 'raspberry-pi': CONFIGURED → CONNECTING → HEALTHY → ACTIVE
[INFO] Worker 'x86-workstation': CONFIGURED → CONNECTING → HEALTHY → ACTIVE
[INFO] llama-server READY at http://127.0.0.1:8181
```

### E. Run Inference

**HTTP (OpenAI-compatible):**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "What is federated inference?"}]
  }'
```

**Python:**
```python
from federated_inference import FederatedInferenceClient
import asyncio

async def main():
    client = FederatedInferenceClient("http://localhost:8080")
    response = await client.chat.completions.create(
        model="llama",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

## Platform-Specific Notes

### macOS

✅ **Yes, run the install script on macOS too.**

```bash
bash scripts/install_llama_cpp.sh
```

This builds `llama-server` with Metal GPU acceleration. Your MacBook becomes the coordinator (or a powerful worker).

### Raspberry Pi

```bash
# 1. Install Python 3.11+ (check with: python3 --version)
curl https://raw.githubusercontent.com/deadsnakes/ubuntu-toolchain-r/master/install.sh | bash

# 2. Install package
pip install federated-inference

# 3. Build llama.cpp (first time ~15 min)
bash scripts/install_llama_cpp.sh

# 4. Start worker
federated-worker start
```

### Android (Termux)

See `scripts/install_llama_cpp_android.md` for detailed steps. Quick summary:

```bash
pkg install python-grpcio     # pre-built wheel (avoids 1-hour compile)
pip install federated-inference --no-deps
pip install protobuf pyyaml click

bash scripts/install_llama_cpp.sh  # builds llama.cpp (~20 min)
federated-worker start
```

---

## Load Balancing & Capacity Management

**How it works:**
- Each worker reports its free RAM and VRAM to the coordinator
- `llama-server` automatically distributes model layers **proportional to reported memory**
- No manual tuning needed — a 32 GB workstation automatically gets ~8× more layers than a 4 GB phone

**Fine-tuning (optional):**
```yaml
workers:
  - id: "weak-device"
    host: "192.168.1.100"
    mem_limit_mb: 2048         # cap at 2 GB (e.g., leave headroom for OS)
    
  - id: "powerful-device"
    host: "192.168.1.101"
    mem_limit_mb: 0            # auto (use all available)
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'uvicorn'` on Worker

**Cause**: Installing `[coordinator]` extras on a worker device.

**Fix**:
```bash
pip install federated-inference  # ← without [coordinator]
```

### `fatal: could not create work tree dir '/tmp/llama.cpp'`

**Cause**: `/tmp/llama.cpp` exists but is corrupted.

**Fix**:
```bash
rm -rf /tmp/llama.cpp
bash scripts/install_llama_cpp.sh
```

### Worker never goes `HEALTHY`

**Check:**
- Worker is running: `ps aux | grep llama`
- Firewall allows port 50051: `netstat -an | grep 50051`
- Coordinator can reach worker: `ping <worker-ip>`
- Worker logs: check coordinator's gRPC error messages

### Coordinator won't start llama-server

**Likely cause**: Model GGUF path doesn't exist or is wrong.

**Debug:**
```bash
# Check the path
ls -lh "~/Library/Application Support/LM Studio/models/.../model.gguf"

# Test llama-server directly
llama-server --model /path/to/model.gguf --port 8181
```

---

## Performance Tips

1. **Use quantized models** (Q4_K_M, Q5_K_M) — they're 4–8× smaller and almost the same quality
2. **Reserve headroom**: Set `mem_limit_mb` on weak devices to leave room for OS
3. **Use `--flash-attn`** in `llama_server_extra_flags` if your GPU supports it
4. **CPU threads**: Adjust `threads` in `model.yaml` based on your coordinator's cores
5. **Network**: Keep devices on the same LAN (WiFi is fine, but Ethernet is faster)

---

## Supported Platforms

| Platform | Coordinator | Worker | Notes |
|---|---|---|---|
| macOS M1/M2/M3/M4 | ✅ | ✅ | Metal GPU acceleration |
| Linux x86_64 | ✅ | ✅ | CPU, NVIDIA CUDA, AMD ROCm |
| Linux ARM64 (Pi) | ✅ | ✅ | CPU-only, ~150+ devices possible |
| Android (Termux) | ❌ | ✅ | See `scripts/install_llama_cpp_android.md` |
| Windows (WSL2) | ✅ | ✅ | Via WSL Linux subsystem |

---

## FAQ

**Q: Does LM Studio or Ollama work as the coordinator?**
A: No. LM Studio and Ollama don't support llama.cpp's `--rpc` flag. You must use `llama-server` (from the install script). But you can freely reuse GGUF files that LM Studio downloaded.

**Q: What's the maximum number of workers?**
A: No hard limit. Performance degrades gracefully with network latency. 10–20 devices on a home LAN work great.

**Q: Can I run coordinator + worker on the same machine?**
A: Yes. Just configure `topology.yaml` with `host: "127.0.0.1"` for a local worker.

**Q: How do I monitor performance?**
A: Check coordinator logs for health updates. curl the `/health` endpoint.

---

## Development

```bash
# Clone and install editable
git clone https://github.com/yourusername/federated-inference
cd federated-inference
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[coordinator,dev]"

# Run tests
pytest

# Build docs (if needed)
# sphinx-build docs docs/_build
```

---

## License

MIT
