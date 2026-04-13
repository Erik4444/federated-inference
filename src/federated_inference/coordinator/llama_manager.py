from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
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
        notify_all: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        self._settings = settings
        self._model = model_config
        self._registry = registry
        self._notify_all = notify_all
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

            # If the server failed to reach READY state, back off and retry.
            if self.state != CoordinatorState.READY:
                logger.info("llama-server did not reach READY, retrying in 10s...")
                await asyncio.sleep(10)
                continue

            # Wait until a restart is requested:
            # - worker topology change (request_restart called externally)
            # - llama-server process died (_monitor_process sets the event)
            self._restart_event.clear()
            await self._restart_event.wait()

            logger.info("Worker topology changed or process died, restarting llama-server...")
            self.state = CoordinatorState.RESTARTING
            await self._stop_server()

    async def _drain_pipe(self, stream: asyncio.StreamReader) -> None:
        """Read lines from a subprocess pipe and log them at INFO level."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                logger.info("llama-server: %s", line.decode(errors="replace").rstrip())
        except Exception:
            pass

    async def _monitor_process(self) -> None:
        """Set restart_event when llama-server exits unexpectedly."""
        if self._process is None:
            return
        await self._process.wait()
        if self.state == CoordinatorState.READY:
            logger.warning(
                "llama-server exited unexpectedly (code %s), requesting restart",
                self._process.returncode,
            )
            self._restart_event.set()

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

        # Drain stdout/stderr continuously so the pipe buffer never fills up.
        # Logged at INFO so startup output is visible without --log-level DEBUG.
        drain_stdout = asyncio.create_task(self._drain_pipe(self._process.stdout))
        drain_stderr = asyncio.create_task(self._drain_pipe(self._process.stderr))

        if await self._wait_ready():
            self.state = CoordinatorState.READY
            logger.info(
                "llama-server READY at http://%s:%d",
                self._settings.llama_server_host,
                self._settings.llama_server_port,
            )
            # Monitor for unexpected crashes while READY.
            asyncio.create_task(self._monitor_process())
            if self._notify_all:
                await self._notify_all(
                    "inference_ready",
                    f"Coordinator is ready. llama-server listening on "
                    f"{self._settings.llama_server_host}:{self._settings.llama_server_port}",
                )
        else:
            # Give drain tasks a moment to flush error output before cancelling.
            await asyncio.sleep(0.5)
            drain_stdout.cancel()
            drain_stderr.cancel()
            returncode = self._process.returncode if self._process else None
            logger.error(
                "llama-server failed to start (exit code: %s). "
                "Check output above for details.",
                returncode,
            )
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
        host = self._settings.llama_server_host
        port = self._settings.llama_server_port
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout

        while loop.time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                return False
            try:
                # Non-blocking connect check via executor so drain tasks keep running.
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=1.0,
                )
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
                return True
            except (OSError, asyncio.TimeoutError):
                await asyncio.sleep(1)
        return False

    async def stop(self) -> None:
        await self._stop_server()
