from __future__ import annotations

import platform
import shutil
import subprocess


def probe_ram() -> tuple[int, int]:
    """Return (total_bytes, free_bytes) using psutil (cross-platform)."""
    import psutil
    vm = psutil.virtual_memory()
    return vm.total, vm.available


def probe_vram() -> tuple[int, int]:
    """Return (total_bytes, free_bytes). Returns (0, 0) if no GPU is found."""
    # Try NVIDIA first
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
            total_mib, free_mib = (int(x.strip()) for x in out.split(","))
            return total_mib * 1024 * 1024, free_mib * 1024 * 1024
        except Exception:
            pass

    # Try AMD ROCm via sysfs
    try:
        import glob
        total_path = glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
        free_path = glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")
        if total_path and free_path:
            total = int(open(total_path[0]).read().strip())
            used = int(open(free_path[0]).read().strip())
            return total, max(0, total - used)
    except Exception:
        pass

    return 0, 0


def get_os_info() -> str:
    return f"{platform.system()} {platform.release()} {platform.machine()}"


def get_arch() -> str:
    return platform.machine()


def probe_llama_version(binary: str) -> str:
    try:
        out = subprocess.check_output(
            [binary, "--version"],
            stderr=subprocess.STDOUT,
            timeout=5,
        ).decode().strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"
