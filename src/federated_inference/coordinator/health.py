from __future__ import annotations

import asyncio
import logging

import grpc

from federated_inference._generated import worker_pb2, worker_pb2_grpc
from federated_inference.coordinator.config import CoordinatorSettings
from federated_inference.coordinator.registry import WorkerRegistry, WorkerState

logger = logging.getLogger(__name__)


# gRPC channel options tuned for unreliable WiFi networks.
# Sends keepalive pings every 30s, times out after 10s with no response.
_GRPC_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.http2.max_pings_without_data", 0),
]

# Number of consecutive failures before a gRPC channel is recycled.
_CHANNEL_REFRESH_THRESHOLD = 3


class HealthLoop:
    """Periodically health-checks all configured workers via gRPC."""

    def __init__(self, registry: WorkerRegistry, settings: CoordinatorSettings) -> None:
        self._registry = registry
        self._settings = settings
        self._task: asyncio.Task | None = None
        self._channels: dict[str, grpc.Channel] = {}
        self._channel_failures: dict[str, int] = {}

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def _get_channel(self, address: str) -> grpc.Channel:
        """Return a cached gRPC channel, recycling stale ones after repeated failures."""
        failures = self._channel_failures.get(address, 0)
        if address in self._channels and failures >= _CHANNEL_REFRESH_THRESHOLD:
            logger.info("Recycling stale gRPC channel for %s (%d failures)", address, failures)
            self._channels[address].close()
            del self._channels[address]
            self._channel_failures[address] = 0

        if address not in self._channels:
            self._channels[address] = grpc.insecure_channel(address, options=_GRPC_CHANNEL_OPTIONS)

        return self._channels[address]

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for ch in self._channels.values():
            ch.close()
        self._channels.clear()

    async def _loop(self) -> None:
        # Initial connection attempt
        for entry in self._registry.all():
            self._registry.transition(entry.id, WorkerState.CONNECTING)
            await self._check_worker(entry.id)

        while True:
            await asyncio.sleep(self._settings.health_check_interval_seconds)
            for entry in self._registry.all():
                await self._check_worker(entry.id)

    async def _check_worker(self, worker_id: str) -> None:
        entry = self._registry.get(worker_id)
        if entry is None:
            return
        node = entry.node
        address = f"{node.host}:{node.grpc_port}"

        channel = self._get_channel(address)
        stub = worker_pb2_grpc.WorkerServiceStub(channel)
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: stub.HealthCheck(
                    worker_pb2.HealthRequest(),
                    timeout=self._settings.worker_timeout_seconds,
                ),
            )
            entry.consecutive_failures = 0
            self._channel_failures[address] = 0

            # Update device info
            di = response.device_info
            entry.device_info = {
                "arch": di.arch,
                "total_ram_bytes": di.total_ram_bytes,
                "free_ram_bytes": di.free_ram_bytes,
                "total_vram_bytes": di.total_vram_bytes,
                "free_vram_bytes": di.free_vram_bytes,
                "os_info": di.os_info,
                "llama_cpp_version": di.llama_cpp_version,
                "cpu_percent": di.cpu_percent,
            }

            if response.rpc_server_running and response.rpc_address:
                entry.rpc_address = f"{entry.node.host}:{entry.node.rpc_port}"
                self._registry.transition(worker_id, WorkerState.ACTIVE)
            else:
                # Include UNREACHABLE and DEGRADED so a recovered worker
                # triggers HEALTHY → start_rpc_on_worker again.
                if entry.state in (
                    WorkerState.ACTIVE,
                    WorkerState.CONNECTING,
                    WorkerState.CONFIGURED,
                    WorkerState.UNREACHABLE,
                    WorkerState.DEGRADED,
                ):
                    self._registry.transition(worker_id, WorkerState.HEALTHY)

        except grpc.RpcError as e:
            entry.consecutive_failures += 1
            self._channel_failures[address] = self._channel_failures.get(address, 0) + 1
            logger.warning("HealthCheck failed for %s: %s", worker_id, e.details() if hasattr(e, 'details') else e)
            self._registry.transition(worker_id, WorkerState.UNREACHABLE)

    async def check_worker_now(self, worker_id: str) -> None:
        """Immediately health-check a worker (e.g. right after discovery)."""
        await self._check_worker(worker_id)

    async def start_rpc_on_worker(self, worker_id: str) -> bool:
        """Ask a worker to start its llama-rpc-server. Returns True on success."""
        entry = self._registry.get(worker_id)
        if entry is None:
            return False

        self._registry.transition(worker_id, WorkerState.RPC_STARTING)
        address = f"{entry.node.host}:{entry.node.grpc_port}"

        channel = self._get_channel(address)
        stub = worker_pb2_grpc.WorkerServiceStub(channel)
        mem_limit_mb = entry.effective_mem_limit_mb()
        if mem_limit_mb > 0:
            logger.info(
                "Worker %s: using mem_limit_mb=%d (free_ram=%s MiB, free_vram=%s MiB)",
                worker_id,
                mem_limit_mb,
                entry.device_info.get("free_ram_bytes", 0) // (1024 * 1024),
                entry.device_info.get("free_vram_bytes", 0) // (1024 * 1024),
            )

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: stub.StartRPC(
                    worker_pb2.StartRPCRequest(
                        port=entry.node.rpc_port,
                        mem_limit_mb=mem_limit_mb,
                    ),
                    timeout=60,
                ),
            )
            if response.success:
                entry.rpc_address = f"{entry.node.host}:{entry.node.rpc_port}"
                self._registry.transition(worker_id, WorkerState.ACTIVE)
                await self.notify_worker(
                    worker_id, "worker_connected",
                    f"Worker {worker_id} is connected and RPC is active."
                )
                return True
            else:
                logger.error("StartRPC failed for %s: %s", worker_id, response.error)
                self._registry.transition(worker_id, WorkerState.HEALTHY)
                return False
        except grpc.RpcError as e:
            logger.error("StartRPC gRPC error for %s: %s", worker_id, e)
            self._registry.transition(worker_id, WorkerState.UNREACHABLE)
            return False

    async def notify_worker(self, worker_id: str, event: str, message: str) -> None:
        entry = self._registry.get(worker_id)
        if entry is None:
            return
        address = f"{entry.node.host}:{entry.node.grpc_port}"
        channel = self._get_channel(address)
        stub = worker_pb2_grpc.WorkerServiceStub(channel)
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: stub.Notify(
                    worker_pb2.NotifyRequest(event=event, message=message),
                    timeout=5,
                ),
            )
        except Exception as e:
            logger.warning("Notify failed for %s: %s", worker_id, e)

    async def notify_all_active(self, event: str, message: str) -> None:
        for entry in self._registry.workers_in_state(WorkerState.ACTIVE):
            await self.notify_worker(entry.id, event, message)

    async def stop_rpc_on_worker(self, worker_id: str) -> None:
        entry = self._registry.get(worker_id)
        if entry is None:
            return
        address = f"{entry.node.host}:{entry.node.grpc_port}"
        if address not in self._channels:
            return
        stub = worker_pb2_grpc.WorkerServiceStub(self._get_channel(address))
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: stub.StopRPC(worker_pb2.StopRPCRequest(force=False), timeout=15),
            )
        except Exception as e:
            logger.warning("StopRPC error for %s: %s", worker_id, e)
