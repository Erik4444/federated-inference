from __future__ import annotations

import logging
import socket
import subprocess
import time

logger = logging.getLogger(__name__)


class RPCManager:
    """Manages the lifecycle of a llama-rpc-server subprocess."""

    def __init__(self, binary: str = "llama-rpc-server") -> None:
        self._binary = binary
        self._process: subprocess.Popen | None = None
        self._port: int | None = None

    def start(self, host: str, port: int, mem_limit_mb: int = 0) -> str:
        """Start llama-rpc-server. Returns 'host:port' on success."""
        if self.is_running():
            logger.info("llama-rpc-server already running on port %d", self._port)
            return f"{host}:{self._port}"

        cmd = [self._binary, "--host", host, "--port", str(port)]
        logger.info("Starting llama-rpc-server: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._port = port

        check_host = "127.0.0.1" if host == "0.0.0.0" else host
        if not self._wait_for_port(check_host, port, timeout=30):
            self.stop(force=True)
            # Retry once with captured output to diagnose the failure
            diag_proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
            )
            stderr_tail = diag_proc.stderr.decode(errors="replace").strip()[-500:]
            msg = f"llama-rpc-server did not start in time on port {port}"
            if stderr_tail:
                msg += f"\nstderr: {stderr_tail}"
            raise RuntimeError(msg)

        logger.info("llama-rpc-server listening on %s:%d", host, port)
        return f"{host}:{port}"

    def stop(self, force: bool = False) -> None:
        if self._process is None:
            return
        if self._process.poll() is not None:
            self._process = None
            self._port = None
            return
        try:
            if force:
                self._process.kill()
            else:
                self._process.terminate()
            self._process.wait(timeout=10)
        except Exception as e:
            logger.warning("Error stopping llama-rpc-server: %s", e)
            try:
                self._process.kill()
            except Exception:
                pass
        finally:
            self._process = None
            self._port = None

    @property
    def port(self) -> int | None:
        return self._port

    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    @staticmethod
    def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except OSError:
                time.sleep(0.5)
        return False
