# Installing federated-inference Worker on Android (Termux)

## Prerequisites

- Android 8.0+ (ARM64 recommended)
- [Termux](https://f-droid.org/packages/com.termux/) installed (use F-Droid version, **not** Play Store)

## Step 1: Install base packages in Termux

```bash
pkg update && pkg upgrade -y
pkg install -y python clang cmake git ninja binutils
```

## Step 2: Install the federated-inference Python package

```bash
pip install federated-inference
```

## Step 3: Build and install llama.cpp

Run the shared install script (it auto-detects Termux):

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/federated-inference/main/scripts/install_llama_cpp.sh)
```

Or manually:

```bash
git clone --depth 1 https://github.com/ggerganov/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build \
  -DLLAMA_BUILD_SERVER=ON \
  -DGGML_RPC=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build --target llama-server llama-rpc-server -j$(nproc)
cp ~/llama.cpp/build/bin/llama-rpc-server $PREFIX/bin/
```

## Step 4: Start the worker

```bash
federated-worker start --grpc-port 50051
```

Or create a simple startup script `~/start-worker.sh`:

```bash
#!/data/data/com.termux/files/usr/bin/bash
federated-worker start \
  --grpc-port 50051 \
  --worker-id "android-$(hostname)"
```

## Step 5: Keep Termux running in the background

Install the [Termux:Boot](https://f-droid.org/packages/com.termux.boot/) add-on to auto-start the worker on device boot:

```bash
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/start-worker.sh <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock
federated-worker start --grpc-port 50051
EOF
chmod +x ~/.termux/boot/start-worker.sh
```

## Notes

- **Performance**: Android devices use CPU inference (no CUDA). Expect ~2-8 tok/s depending on the device and model size.
- **RAM**: A typical Android phone has 8-12 GB RAM. With Q4_K_M quantization, a 7B model fits in ~4 GB.
- **Battery**: The device will consume significant battery while running inference. Consider keeping it plugged in.
- **Network**: Coordinator and workers must be on the same local network (or VPN). The coordinator connects to the worker's gRPC port (default 50051) to manage it.
