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
#   LLAMA_FORCE    — set to 1 to rebuild even if version is already installed

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

# ── Skip if already installed at requested version ──────────────────────────

VERSION_FILE="$BINDIR/.llama_cpp_version"
INSTALLED_VERSION=""
[ -f "$VERSION_FILE" ] && INSTALLED_VERSION=$(cat "$VERSION_FILE")

if [ "${LLAMA_FORCE:-0}" != "1" ] && \
   [ "$INSTALLED_VERSION" = "$LLAMA_VERSION" ] && \
   [ -f "$BINDIR/llama-server" ] && \
   [ -f "$BINDIR/rpc-server" ]; then
  info "llama.cpp $LLAMA_VERSION already installed — skipping build."
  info "Set LLAMA_FORCE=1 to force a rebuild."
  exit 0
fi

# ── Build dependencies ──────────────────────────────────────────────────────

install_deps() {
  if [ "$OS" = "Darwin" ]; then
    command -v cmake &>/dev/null || { info "Installing cmake via Homebrew..."; brew install cmake; }
    # ccache speeds up repeated builds significantly
    command -v ccache &>/dev/null || { brew install ccache 2>/dev/null || true; }
  elif $IS_TERMUX; then
    pkg install -y cmake clang git ninja ccache 2>/dev/null || true
  elif command -v apt-get &>/dev/null; then
    apt-get install -y --no-install-recommends cmake build-essential git ninja-build ccache 2>/dev/null || true
  elif command -v dnf &>/dev/null; then
    dnf install -y cmake gcc-c++ git ninja-build ccache 2>/dev/null || true
  fi
}

install_deps

# ── ccache setup ────────────────────────────────────────────────────────────

CCACHE_CMAKE_FLAGS=()
if command -v ccache &>/dev/null; then
  ok "ccache found — repeated builds will be significantly faster"
  CCACHE_CMAKE_FLAGS=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
fi

# ── Clone / update source ──────────────────────────────────────────────────

if [ -d "$LLAMA_DIR/.git" ]; then
  CURRENT_TAG=$(git -C "$LLAMA_DIR" describe --tags --exact-match 2>/dev/null || echo "")
  if [ "$CURRENT_TAG" = "$LLAMA_VERSION" ]; then
    info "Source already at $LLAMA_VERSION — reusing existing clone"
  else
    info "Updating existing clone to $LLAMA_VERSION ..."
    git -C "$LLAMA_DIR" fetch --depth 1 origin tag "$LLAMA_VERSION"
    git -C "$LLAMA_DIR" checkout "$LLAMA_VERSION"
    # Source changed: wipe build dir so cmake reconfigures cleanly
    rm -rf "$LLAMA_DIR/build"
  fi
elif [ -e "$LLAMA_DIR" ]; then
  info "$LLAMA_DIR exists but is not a git repo — re-cloning..."
  rm -rf "$LLAMA_DIR"
  git clone --depth 1 --branch "$LLAMA_VERSION" https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
else
  info "Cloning llama.cpp $LLAMA_VERSION ..."
  git clone --depth 1 --branch "$LLAMA_VERSION" https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

# ── CMake configuration ────────────────────────────────────────────────────

CMAKE_FLAGS=(
  -DLLAMA_BUILD_SERVER=ON
  -DGGML_RPC=ON
  -DCMAKE_BUILD_TYPE=Release
  "${CCACHE_CMAKE_FLAGS[@]}"
)

# Detect available RAM in MB (best-effort, returns 0 if unknown).
available_ram_mb() {
  if [ "$OS" = "Darwin" ]; then
    echo $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))
  elif [ -r /proc/meminfo ]; then
    awk '/^MemTotal:/ { print int($2/1024) }' /proc/meminfo
  else
    echo 0
  fi
}

RAM_MB=$(available_ram_mb)

# Low-RAM builds: ggml-cpu.cpp is the worst offender — even a single compiler
# process can consume >1.5 GB at -O2/-O3 due to the llamafile SIMD dispatch
# variants and CPU-all-variants codegen.
#
# Thresholds (total device RAM):
#   < 2 GB  → ultra-low: -O0, disable ALL optional codegen (Galaxy Tab E/A etc.)
#   < 4 GB  → low:       -O1, disable CPU-all-variants and llamafile dispatch
if [ "$RAM_MB" -gt 0 ] && [ "$RAM_MB" -lt 2048 ]; then
  info "Ultra-low-RAM device detected (${RAM_MB} MB) — disabling heavy codegen"
  CMAKE_FLAGS+=(
    -DGGML_NATIVE=OFF
    -DGGML_CPU_ALL_VARIANTS=OFF
    -DGGML_LLAMAFILE=OFF
    "-DCMAKE_C_FLAGS=-O0 -g0 -fno-lto"
    "-DCMAKE_CXX_FLAGS=-O0 -g0 -fno-lto"
  )
elif [ "$RAM_MB" -gt 0 ] && [ "$RAM_MB" -lt 4096 ]; then
  info "Low-RAM device detected (${RAM_MB} MB) — reducing compile-time memory usage"
  CMAKE_FLAGS+=(
    -DGGML_NATIVE=OFF
    -DGGML_CPU_ALL_VARIANTS=OFF
    -DGGML_LLAMAFILE=OFF
    "-DCMAKE_C_FLAGS=-O1 -g0 -fno-lto"
    "-DCMAKE_CXX_FLAGS=-O1 -g0 -fno-lto"
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

if [ -n "${LLAMA_JOBS:-}" ]; then
  JOBS="$LLAMA_JOBS"
elif [ "$RAM_MB" -gt 0 ] && [ "$RAM_MB" -lt 4096 ]; then
  # Low-RAM devices (< 4 GB): single-threaded to avoid OOM-killer.
  # Note: Termux with >= 4 GB RAM uses the standard parallel detection below.
  JOBS=1
  info "Low-RAM build: forcing -j1 (${RAM_MB} MB RAM, set LLAMA_JOBS to override)"
else
  if [ "$OS" = "Darwin" ]; then
    CPU_JOBS=$(sysctl -n hw.logicalcpu)
  else
    CPU_JOBS=$(nproc 2>/dev/null || echo 2)
  fi
  # Cap to 1 job per 1.5 GB to avoid OOM.
  if [ "$RAM_MB" -gt 0 ]; then
    RAM_JOBS=$(( RAM_MB / 1536 < 1 ? 1 : RAM_MB / 1536 ))
    JOBS=$(( RAM_JOBS < CPU_JOBS ? RAM_JOBS : CPU_JOBS ))
  else
    JOBS="$CPU_JOBS"
  fi
fi

info "Building with $JOBS job(s) ..."
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" "${CMAKE_FLAGS[@]}"
cmake --build "$LLAMA_DIR/build" --config Release -j"$JOBS" \
  --target llama-server rpc-server

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

# ── Stamp installed version ──────────────────────────────────────────────────

echo "$LLAMA_VERSION" > "$VERSION_FILE"

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
info "Installed ${#INSTALLED[*]} binaries to $BINDIR"
info "Version: $LLAMA_VERSION"
info "Verify:"
for bin in "${INSTALLED[@]}"; do
  echo "    $bin --help"
done
