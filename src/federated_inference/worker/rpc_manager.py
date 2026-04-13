from __future__ import annotations

import asyncio
import logging
import socket
import time

logger = logging.getLogger(__name__)


class RPCManager:
    """Manages the lifecycle of a llama-rpc-server subprocess."""

    def __init__(self, binary: str = "llama-rpc-server") -> None:
        self._binary = binary
        self._process: asyncio.subprocess.Process | None = None
        self._port: int | None = None

    async def start(self, host: str, port: int, mem_limit_mb: int = 0) -> str:
        """Start llama-rpc-server. Returns 'host:port' on success."""
        if self.is_running():
            logger.info("llama-rpc-server already running on port %d", self._port)
            return f"{host}:{self._port}"

        cmd = [self._binary, "--host", host, "--port", str(port)]

        logger.info("Starting llama-rpc-server: %s", " ".join(cmd))
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._port = port

        # Wait until the port is accepting connections (up to 30s)
        if not await self._wait_for_port(host if host != "0.0.0.0" else "127.0.0.1", port, timeout=30):
            await self.stop(force=True)
            raise RuntimeError(f"llama-rpc-server did not start in time on port {port}")

        logger.info("llama-rpc-server listening on %s:%d", host, port)
        return f"{host}:{port}"

    async def stop(self, force: bool = False) -> None:
        if self._process is None:
            return
        if self._process.returncode is not None:
            self._process = None
            return
        try:
            if force:
                self._process.kill()
            else:
                self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=10)
        except Exception as e:
            logger.warning("Error stopping llama-rpc-server: %s", e)
        finally:
            self._process = None
            self._port = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @staticmethod
    async def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except OSError:
                await asyncio.sleep(0.5)
        return False
