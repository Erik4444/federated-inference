#!/usr/bin/env bash
# install_llama_cpp.sh
# Installs llama-server and llama-rpc-server on the current machine.
# Supports: Linux x86_64, Linux aarch64, macOS ARM64, Android (Termux/aarch64).
#
# Usage:
#   bash install_llama_cpp.sh [--prefix /usr/local]

set -euo pipefail

LLAMA_VERSION="${LLAMA_VERSION:-latest}"   # override with: LLAMA_VERSION=b5622 bash install_llama_cpp.sh

ARCH=$(uname -m)
OS=$(uname -s)

echo "==> Detecting platform: OS=$OS ARCH=$ARCH"

# ── Termux detection ────────────────────────────────────────────────────────
IS_TERMUX=false
if [ -n "${TERMUX_VERSION:-}" ] || [ -d "/data/data/com.termux" ]; then
  IS_TERMUX=true
  echo "==> Termux environment detected"
fi

# On Termux /usr/local doesn't exist — default to the Termux prefix.
# On macOS/Linux default to ~/.local so no sudo is required.
if $IS_TERMUX; then
  PREFIX="${1:-/data/data/com.termux/files/usr}"
else
  PREFIX="${1:-$HOME/.local}"
fi

# ── Install build deps ───────────────────────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
  if ! command -v cmake &>/dev/null; then
    echo "==> Installing cmake via Homebrew..."
    brew install cmake
  fi
elif $IS_TERMUX; then
  pkg install -y cmake clang git ninja 2>/dev/null || true
elif command -v apt-get &>/dev/null; then
  apt-get install -y --no-install-recommends cmake build-essential git ninja-build 2>/dev/null || true
elif command -v dnf &>/dev/null; then
  dnf install -y cmake gcc-c++ git ninja-build 2>/dev/null || true
fi

# ── Clone or update ──────────────────────────────────────────────────────────
# On Android/Termux /tmp is not writable — use $TMPDIR or $HOME instead
if $IS_TERMUX; then
  LLAMA_DIR="${TMPDIR:-$HOME}/llama.cpp"
else
  LLAMA_DIR="/tmp/llama.cpp"
fi
# Resolve "latest" to the actual newest release tag
if [ "$LLAMA_VERSION" = "latest" ]; then
  echo "==> Fetching latest llama.cpp release tag..."
  LLAMA_VERSION=$(curl -fsSL "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest" \
    | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
  if [ -z "$LLAMA_VERSION" ]; then
    echo "ERROR: Could not determine latest release tag. Set LLAMA_VERSION manually."
    exit 1
  fi
  echo "==> Latest release: $LLAMA_VERSION"
fi

if [ -d "$LLAMA_DIR/.git" ]; then
  echo "==> Switching existing clone to $LLAMA_VERSION ..."
  git -C "$LLAMA_DIR" fetch --depth 1 origin tag "$LLAMA_VERSION"
  git -C "$LLAMA_DIR" checkout "$LLAMA_VERSION"
elif [ -e "$LLAMA_DIR" ]; then
  echo "==> $LLAMA_DIR exists but is not a git repo — removing and re-cloning..."
  rm -rf "$LLAMA_DIR"
  git clone --depth 1 --branch "$LLAMA_VERSION" https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
else
  echo "==> Cloning llama.cpp $LLAMA_VERSION ..."
  git clone --depth 1 --branch "$LLAMA_VERSION" https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

# ── macOS ────────────────────────────────────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
  echo "==> Building for macOS (Metal acceleration)"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DGGML_METAL=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" --config Release \
    -j"$(sysctl -n hw.logicalcpu)"

  mkdir -p "$PREFIX/bin"
  if [ -f "$LLAMA_DIR/build/bin/llama-server" ]; then
    install -m755 "$LLAMA_DIR/build/bin/llama-server" "$PREFIX/bin/"
    echo "    ✓ installed llama-server"
  fi
  if [ -f "$LLAMA_DIR/build/bin/llama-rpc-server" ]; then
    install -m755 "$LLAMA_DIR/build/bin/llama-rpc-server" "$PREFIX/bin/"
    echo "    ✓ installed llama-rpc-server"
  else
    echo "    ℹ llama-rpc-server not found (RPC built into llama-server)"
  fi
  echo "==> Done (macOS Metal)"
  exit 0
fi

# ── Linux / Termux ───────────────────────────────────────────────────────────
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
cmake --build "$LLAMA_DIR/build" --config Release -j"$JOBS"

# Install
BINDIR="$PREFIX/bin"
mkdir -p "$BINDIR"

INSTALLED_COUNT=0

if [ -f "$LLAMA_DIR/build/bin/llama-server" ]; then
  cp "$LLAMA_DIR/build/bin/llama-server" "$BINDIR/llama-server"
  chmod 755 "$BINDIR/llama-server"
  echo "    ✓ installed llama-server"
  ((INSTALLED_COUNT++))
else
  echo "    ✗ llama-server not found in build output"
fi

if [ -f "$LLAMA_DIR/build/bin/llama-rpc-server" ]; then
  cp "$LLAMA_DIR/build/bin/llama-rpc-server" "$BINDIR/llama-rpc-server"
  chmod 755 "$BINDIR/llama-rpc-server"
  echo "    ✓ installed llama-rpc-server"
  ((INSTALLED_COUNT++))
else
  echo "    ℹ llama-rpc-server not found (RPC built into llama-server in this version)"
fi

if [ "$INSTALLED_COUNT" -eq 0 ]; then
  echo "ERROR: No binaries were installed. Build may have failed."
  exit 1
fi

# If llama-rpc-server doesn't exist but llama-server does, symlink it
# (modern llama.cpp combines both into llama-server)
if [ ! -f "$BINDIR/llama-rpc-server" ] && [ -f "$BINDIR/llama-server" ]; then
  echo "==> Creating symlink: llama-rpc-server → llama-server"
  ln -sf "$BINDIR/llama-server" "$BINDIR/llama-rpc-server"
  chmod 755 "$BINDIR/llama-server"   # ensure target is executable
fi

echo ""
echo "==> Installed to: $BINDIR"
echo "==> Verify with:"
echo "    llama-server --version"
echo "    llama-rpc-server --version"
