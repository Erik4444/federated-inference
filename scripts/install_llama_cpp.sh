#!/usr/bin/env bash
# install_llama_cpp.sh — Build and install llama-server + rpc-server from source.
# Supports: macOS (Metal), Linux x86_64/aarch64 (CUDA, ROCm, CPU), Android (Termux).
#
# Usage:
#   bash install_llama_cpp.sh [PREFIX]
#
# Environment:
#   LLAMA_VERSION  — git tag to build (default: latest release)
#   LLAMA_JOBS     — parallel build jobs (default: auto-detect)

set -euo pipefail

# ── Helpers ─────────────────────────────────────────────────────────────────

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }
ok()   { echo "    ✓ $*"; }
warn() { echo "    ✗ $*"; }

# ── Platform detection ──────────────────────────────────────────────────────

ARCH=$(uname -m)
OS=$(uname -s)
IS_TERMUX=false
[ -n "${TERMUX_VERSION:-}" ] || [ -d "/data/data/com.termux" ] && IS_TERMUX=true

info "Platform: OS=$OS ARCH=$ARCH${IS_TERMUX:+ (Termux)}"

# ── Prefix (install destination) ────────────────────────────────────────────

if $IS_TERMUX; then
  PREFIX="${1:-/data/data/com.termux/files/usr}"
else
  PREFIX="${1:-$HOME/.local}"
fi
BINDIR="$PREFIX/bin"

# ── Source directory ────────────────────────────────────────────────────────

if $IS_TERMUX; then
  LLAMA_DIR="${TMPDIR:-$HOME}/llama.cpp"
else
  LLAMA_DIR="/tmp/llama.cpp"
fi

# ── Version resolution ──────────────────────────────────────────────────────

LLAMA_VERSION="${LLAMA_VERSION:-latest}"

if [ "$LLAMA_VERSION" = "latest" ]; then
  info "Fetching latest llama.cpp release tag..."
  LLAMA_VERSION=$(curl -fsSL "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest" \
    | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
  [ -n "$LLAMA_VERSION" ] || die "Could not determine latest release tag. Set LLAMA_VERSION manually."
  info "Latest release: $LLAMA_VERSION"
fi

# ── Build dependencies ──────────────────────────────────────────────────────

install_deps() {
  if [ "$OS" = "Darwin" ]; then
    command -v cmake &>/dev/null || { info "Installing cmake via Homebrew..."; brew install cmake; }
  elif $IS_TERMUX; then
    pkg install -y cmake clang git ninja 2>/dev/null || true
  elif command -v apt-get &>/dev/null; then
    apt-get install -y --no-install-recommends cmake build-essential git ninja-build 2>/dev/null || true
  elif command -v dnf &>/dev/null; then
    dnf install -y cmake gcc-c++ git ninja-build 2>/dev/null || true
  fi
}

install_deps

# ── Clone / update source ──────────────────────────────────────────────────

if [ -d "$LLAMA_DIR/.git" ]; then
  info "Updating existing clone to $LLAMA_VERSION ..."
  git -C "$LLAMA_DIR" fetch --depth 1 origin tag "$LLAMA_VERSION"
  git -C "$LLAMA_DIR" checkout "$LLAMA_VERSION"
elif [ -e "$LLAMA_DIR" ]; then
  info "$LLAMA_DIR exists but is not a git repo — re-cloning..."
  rm -rf "$LLAMA_DIR"
  git clone --depth 1 --branch "$LLAMA_VERSION" https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
else
  info "Cloning llama.cpp $LLAMA_VERSION ..."
  git clone --depth 1 --branch "$LLAMA_VERSION" https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

rm -rf "$LLAMA_DIR/build"

# ── CMake configuration ────────────────────────────────────────────────────

CMAKE_FLAGS=(
  -DLLAMA_BUILD_SERVER=ON
  -DGGML_RPC=ON
  -DCMAKE_BUILD_TYPE=Release
)

# On Termux (low-RAM Android devices) reduce compilation memory usage:
#   - GGML_NATIVE=OFF   skips native CPU-feature detection → fewer heavy variants
#   - O1 instead of O3  cuts per-TU RAM by ~40–60 % (ggml-cpu is the worst offender)
if $IS_TERMUX; then
  CMAKE_FLAGS+=(
    -DGGML_NATIVE=OFF
    -DCMAKE_CXX_FLAGS_RELEASE="-O1"
    -DCMAKE_C_FLAGS_RELEASE="-O1"
  )
fi

detect_gpu() {
  if [ "$OS" = "Darwin" ]; then
    CMAKE_FLAGS+=(-DGGML_METAL=ON)
    info "GPU: Metal"
  elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    CMAKE_FLAGS+=(-DGGML_CUDA=ON)
    info "GPU: NVIDIA CUDA"
  elif command -v rocminfo &>/dev/null; then
    CMAKE_FLAGS+=(-DGGML_HIPBLAS=ON)
    info "GPU: AMD ROCm"
  else
    info "GPU: none (CPU-only build)"
  fi
}

detect_gpu

# ── Build ───────────────────────────────────────────────────────────────────

# Determine available RAM in MB, then cap jobs to ~1 per 1.5 GB so
# the OOM-killer does not abort heavy C++ translation units (e.g. httplib).
ram_cap_jobs() {
  local ram_mb=0
  if [ "$OS" = "Darwin" ]; then
    ram_mb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))
  elif [ -r /proc/meminfo ]; then
    ram_mb=$(awk '/^MemTotal:/ { print int($2/1024) }' /proc/meminfo)
  fi
  if [ "$ram_mb" -gt 0 ]; then
    echo $(( ram_mb / 1536 < 1 ? 1 : ram_mb / 1536 ))
  else
    echo ""   # unknown — caller falls back to CPU count
  fi
}

if [ -n "${LLAMA_JOBS:-}" ]; then
  JOBS="$LLAMA_JOBS"
elif $IS_TERMUX; then
  # Always single-threaded on Termux: ggml-cpu TUs can consume >1.5 GB each,
  # so parallel jobs reliably trigger the Android OOM-killer.
  JOBS=1
  info "Termux build: forcing -j1 to avoid OOM on ggml-cpu (set LLAMA_JOBS to override)"
else
  if [ "$OS" = "Darwin" ]; then
    CPU_JOBS=$(sysctl -n hw.logicalcpu)
  else
    CPU_JOBS=$(nproc 2>/dev/null || echo 2)
  fi
  RAM_JOBS=$(ram_cap_jobs)
  if [ -n "$RAM_JOBS" ] && [ "$RAM_JOBS" -lt "$CPU_JOBS" ]; then
    JOBS="$RAM_JOBS"
    info "RAM-limited build: capping to $JOBS job(s) (set LLAMA_JOBS to override)"
  else
    JOBS="$CPU_JOBS"
  fi
fi

info "Building with $JOBS job(s) ..."
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" "${CMAKE_FLAGS[@]}"
cmake --build "$LLAMA_DIR/build" --config Release -j"$JOBS"

# ── Install binaries ───────────────────────────────────────────────────────

mkdir -p "$BINDIR"
INSTALLED=()

install_bin() {
  local name="$1"
  local src="$LLAMA_DIR/build/bin/$name"

  # Direct path check first, then search
  if [ ! -f "$src" ]; then
    src=$(find "$LLAMA_DIR/build" -name "$name" -not -name "*.o" -not -name "*.d" -type f 2>/dev/null | head -1)
  fi

  if [ -n "$src" ] && [ -f "$src" ]; then
    install -m755 "$src" "$BINDIR/$name"
    ok "installed $name"
    INSTALLED+=("$name")
  else
    warn "$name not found in build output"
  fi
}

install_bin llama-server
install_bin rpc-server

[ ${#INSTALLED[@]} -gt 0 ] || die "No binaries installed. Build may have failed."

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
info "Installed ${#INSTALLED[*]} binaries to $BINDIR"
info "Version: $LLAMA_VERSION"
info "Verify:"
for bin in "${INSTALLED[@]}"; do
  echo "    $bin --help"
done
