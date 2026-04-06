import platform
from unittest.mock import MagicMock, patch

import pytest

from federated_inference.worker import device_info as di


def test_probe_ram_returns_positive():
    total, free = di.probe_ram()
    assert total > 0
    assert free >= 0
    assert free <= total


def test_probe_vram_graceful_fallback():
    # On CI / no-GPU machine this should return (0, 0) without crashing
    with patch("shutil.which", return_value=None):
        total, free = di.probe_vram()
    assert total >= 0
    assert free >= 0


def test_probe_vram_nvidia(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/nvidia-smi" if x == "nvidia-smi" else None)
    with patch("subprocess.check_output", return_value=b"8192, 6000\n"):
        total, free = di.probe_vram()
    assert total == 8192 * 1024 * 1024
    assert free == 6000 * 1024 * 1024


def test_get_arch():
    arch = di.get_arch()
    assert arch == platform.machine()


def test_get_os_info():
    info = di.get_os_info()
    assert platform.system() in info


def test_probe_llama_version_missing_binary():
    version = di.probe_llama_version("/nonexistent/binary")
    assert version == "unknown"
