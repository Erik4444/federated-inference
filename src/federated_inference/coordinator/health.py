from __future__ import annotations

import asyncio
import logging

import grpc

from federated_inference._generated import worker_pb2, worker_pb2_grpc
from federated_inference.coordinator.config import CoordinatorSettings
from federated_inference.coordinator.registry import WorkerRegistry, WorkerState

logger = logging.getLogger(__name__)


class HealthLoop:
    """Periodically health-checks all configured workers via gRPC."""

    def __init__(self, registry: WorkerRegistry, settings: CoordinatorSettings) -> None:
        self._registry = registry
        self._settings = settings
        self._task: asyncio.Task | None = None
        self._channels: dict[str, grpc.Channel] = {}

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for ch in self._channels.values():
            ch.close()

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

        if address not in self._channels:
            self._channels[address] = grpc.insecure_channel(address)

        stub = worker_pb2_grpc.WorkerServiceStub(self._channels[address])
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: stub.HealthCheck(
                    worker_pb2.HealthRequest(),
                    timeout=self._settings.worker_timeout_seconds,
                ),
            )
            entry.consecutive_failures = 0

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
            }

            if response.rpc_server_running and response.rpc_address:
                entry.rpc_address = response.rpc_address
                self._registry.transition(worker_id, WorkerState.ACTIVE)
            else:
                if entry.state == WorkerState.ACTIVE:
                    self._registry.transition(worker_id, WorkerState.HEALTHY)
                elif entry.state in (WorkerState.CONNECTING, WorkerState.CONFIGURED):
                    self._registry.transition(worker_id, WorkerState.HEALTHY)

        except grpc.RpcError as e:
            entry.consecutive_failures += 1
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

        if address not in self._channels:
            self._channels[address] = grpc.insecure_channel(address)

        stub = worker_pb2_grpc.WorkerServiceStub(self._channels[address])
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
            response = await asyncio.get_event_loop().run_in_executor(
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
                entry.rpc_address = response.address
                self._registry.transition(worker_id, WorkerState.ACTIVE)
                return True
            else:
                logger.error("StartRPC failed for %s: %s", worker_id, response.error)
                self._registry.transition(worker_id, WorkerState.HEALTHY)
                return False
        except grpc.RpcError as e:
            logger.error("StartRPC gRPC error for %s: %s", worker_id, e)
            self._registry.transition(worker_id, WorkerState.UNREACHABLE)
            return False

    async def stop_rpc_on_worker(self, worker_id: str) -> None:
        entry = self._registry.get(worker_id)
        if entry is None:
            return
        address = f"{entry.node.host}:{entry.node.grpc_port}"
        if address not in self._channels:
            return
        stub = worker_pb2_grpc.WorkerServiceStub(self._channels[address])
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: stub.StopRPC(worker_pb2.StopRPCRequest(force=False), timeout=15),
            )
        except Exception as e:
            logger.warning("StopRPC error for %s: %s", worker_id, e)
