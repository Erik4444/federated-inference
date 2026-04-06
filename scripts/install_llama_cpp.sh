#!/usr/bin/env bash
# install_llama_cpp.sh
# Installs llama-server and llama-rpc-server on the current machine.
# Supports: Linux x86_64, Linux aarch64, macOS ARM64, Android (Termux/aarch64).
#
# Usage:
#   bash install_llama_cpp.sh [--prefix /usr/local]

set -euo pipefail

PREFIX="${1:-/usr/local}"
LLAMA_VERSION="b5622"   # pin to a known-good release; update as needed

ARCH=$(uname -m)
OS=$(uname -s)

echo "==> Detecting platform: OS=$OS ARCH=$ARCH"

# ── Termux detection ────────────────────────────────────────────────────────
IS_TERMUX=false
if [ -n "${TERMUX_VERSION:-}" ] || [ -d "/data/data/com.termux" ]; then
  IS_TERMUX=true
  echo "==> Termux environment detected"
fi

# ── macOS ───────────────────────────────────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
  echo "==> Building for macOS (Metal acceleration)"
  if ! command -v cmake &>/dev/null; then
    echo "  Installing cmake via Homebrew..."
    brew install cmake
  fi
  git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp 2>/dev/null || \
    (cd /tmp/llama.cpp && git pull)
  cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
    -DGGML_METAL=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build /tmp/llama.cpp/build --config Release -j"$(sysctl -n hw.logicalcpu)"
  install -m755 /tmp/llama.cpp/build/bin/llama-server "$PREFIX/bin/"
  install -m755 /tmp/llama.cpp/build/bin/llama-rpc-server "$PREFIX/bin/"
  echo "==> Done (macOS Metal)"
  exit 0
fi

# ── Linux / Termux ──────────────────────────────────────────────────────────
HAS_CUDA=false
HAS_ROCM=false

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  HAS_CUDA=true
  echo "==> NVIDIA GPU detected"
elif command -v rocminfo &>/dev/null; then
  HAS_ROCM=true
  echo "==> AMD ROCm GPU detected"
else
  echo "==> No GPU detected, building CPU-only"
fi

# Install build deps
if $IS_TERMUX; then
  pkg install -y cmake clang git ninja 2>/dev/null || true
elif command -v apt-get &>/dev/null; then
  apt-get install -y --no-install-recommends cmake build-essential git ninja-build 2>/dev/null || true
elif command -v dnf &>/dev/null; then
  dnf install -y cmake gcc-c++ git ninja-build 2>/dev/null || true
fi

# Clone or update
LLAMA_DIR="/tmp/llama.cpp"
if [ -d "$LLAMA_DIR/.git" ]; then
  echo "==> Updating existing llama.cpp clone..."
  git -C "$LLAMA_DIR" pull --ff-only
else
  echo "==> Cloning llama.cpp..."
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

# Configure CMake flags
CMAKE_FLAGS=(
  "-DLLAMA_BUILD_SERVER=ON"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DGGML_RPC=ON"
)

if $HAS_CUDA; then
  CMAKE_FLAGS+=("-DGGML_CUDA=ON")
elif $HAS_ROCM; then
  CMAKE_FLAGS+=("-DGGML_HIPBLAS=ON")
fi

# Build
JOBS=$(nproc 2>/dev/null || echo 2)
echo "==> Building with ${JOBS} jobs (flags: ${CMAKE_FLAGS[*]})"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" "${CMAKE_FLAGS[@]}"
cmake --build "$LLAMA_DIR/build" --config Release -j"$JOBS" \
  --target llama-server llama-rpc-server

# Install
BINDIR="$PREFIX/bin"
if $IS_TERMUX; then
  BINDIR="$PREFIX/bin"
fi
mkdir -p "$BINDIR"
install -m755 "$LLAMA_DIR/build/bin/llama-server" "$BINDIR/"
install -m755 "$LLAMA_DIR/build/bin/llama-rpc-server" "$BINDIR/"

echo ""
echo "==> Installed:"
echo "    $BINDIR/llama-server"
echo "    $BINDIR/llama-rpc-server"
echo ""
echo "==> Verify with:"
echo "    llama-rpc-server --version"
