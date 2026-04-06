# federated-inference

Run large LLMs across multiple heterogeneous edge devices (Raspberry Pi, Android, x86, Jetson) using pipeline parallelism via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s built-in RPC backend.

## Architecture

```
Client → REST API (OpenAI-compatible, port 8080)
              │
         Coordinator
         ├── WorkerRegistry  (tracks device health)
         ├── HealthLoop      (gRPC polling every 10s)
         ├── LlamaManager    (manages llama-server with --rpc <workers>)
         └── gRPC → Worker 1 (Raspberry Pi)
                  → Worker 2 (x86 workstation)
                  → Worker N (Android phone via Termux)
                       │
                  llama-rpc-server (llama.cpp)
```

## Quick Start

### 1. Install on every device

```bash
pip install federated-inference
bash scripts/install_llama_cpp.sh
```

### 2. Start workers on edge devices

```bash
federated-worker start --grpc-port 50051
```

### 3. Configure and start coordinator

```bash
cp config/topology.example.yaml topology.yaml
cp config/model.example.yaml model.yaml
# Edit topology.yaml with your device IPs
federated-coordinator start --topology topology.yaml --model model.yaml
```

### 4. Run inference

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3","messages":[{"role":"user","content":"Hello!"}]}'
```

Or use the Python client:

```python
from federated_inference import FederatedInferenceClient
import asyncio

async def main():
    client = FederatedInferenceClient("http://localhost:8080")
    resp = await client.chat.completions.create(
        model="llama-3",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp.choices[0].message.content)

asyncio.run(main())
```

## Supported Platforms

| Platform | Worker | Notes |
|---|---|---|
| Linux x86_64 | ✅ | Full support |
| Linux ARM64 (Pi) | ✅ | Full support |
| macOS ARM64 (M1+) | ✅ | Metal acceleration |
| Android (Termux) | ✅ | See `scripts/install_llama_cpp_android.md` |
| Windows (WSL2) | ✅ | Via WSL |
