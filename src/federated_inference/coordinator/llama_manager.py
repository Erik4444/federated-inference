from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto

from federated_inference.coordinator.config import CoordinatorSettings, ModelConfig
from federated_inference.coordinator.registry import WorkerRegistry, WorkerState

logger = logging.getLogger(__name__)


class CoordinatorState(Enum):
    IDLE = auto()
    STARTING = auto()
    READY = auto()
    RESTARTING = auto()
    STOPPING = auto()


class LlamaManager:
    """
    Manages the llama-server subprocess on the coordinator.

    Builds the --rpc argument from ACTIVE workers and restarts
    the server when worker topology changes.
    """

    def __init__(
        self,
        settings: CoordinatorSettings,
        model_config: ModelConfig,
        registry: WorkerRegistry,
    ) -> None:
        self._settings = settings
        self._model = model_config
        self._registry = registry
        self._process: asyncio.subprocess.Process | None = None
        self.state = CoordinatorState.IDLE
        self._restart_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    def request_restart(self) -> None:
        """Signal that the worker set has changed and llama-server should restart."""
        self._restart_event.set()

    async def run(self) -> None:
        """Main lifecycle loop – call this as a background task."""
        self._task = asyncio.current_task()
        while True:
            active_addresses = self._registry.active_rpc_addresses()
            if not active_addresses:
                logger.info("No ACTIVE workers, waiting...")
                self.state = CoordinatorState.IDLE
                await asyncio.sleep(5)
                continue

            await self._start_server(active_addresses)

            # Wait until a restart is requested (e.g. worker comes/goes)
            self._restart_event.clear()
            await self._restart_event.wait()

            logger.info("Worker topology changed, restarting llama-server...")
            self.state = CoordinatorState.RESTARTING
            await self._stop_server()

    async def _start_server(self, rpc_addresses: list[str]) -> None:
        self.state = CoordinatorState.STARTING
        m = self._model.model
        cmd = [
            self._settings.llama_server_binary,
            "--model", m.path,
            "--host", self._settings.llama_server_host,
            "--port", str(self._settings.llama_server_port),
            "--ctx-size", str(m.context_length),
            "--n-gpu-layers", str(m.n_gpu_layers),
            "--batch-size", str(m.batch_size),
            "--threads", str(m.threads),
            "--rpc", ",".join(rpc_addresses),
            *self._model.llama_server_extra_flags,
        ]
        logger.info("Starting llama-server: %s", " ".join(cmd))
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for llama-server to be ready
        if await self._wait_ready():
            self.state = CoordinatorState.READY
            logger.info(
                "llama-server READY at http://%s:%d",
                self._settings.llama_server_host,
                self._settings.llama_server_port,
            )
        else:
            logger.error("llama-server failed to start")
            await self._stop_server()

    async def _stop_server(self) -> None:
        self.state = CoordinatorState.STOPPING
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=15)
            except asyncio.TimeoutError:
                self._process.kill()
        self._process = None
        self.state = CoordinatorState.IDLE

    async def _wait_ready(self, timeout: float = 120.0) -> bool:
        import time
        import socket as sock

        host = self._settings.llama_server_host
        port = self._settings.llama_server_port
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                return False
            try:
                with sock.create_connection((host, port), timeout=1):
                    return True
            except OSError:
                await asyncio.sleep(1)
        return False

    async def stop(self) -> None:
        await self._stop_server()
