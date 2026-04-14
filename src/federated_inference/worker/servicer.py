from __future__ import annotations

import logging

import grpc

from federated_inference._generated import worker_pb2, worker_pb2_grpc
from federated_inference.worker import device_info as di
from federated_inference.worker.rpc_manager import RPCManager

logger = logging.getLogger(__name__)


class WorkerServicer(worker_pb2_grpc.WorkerServiceServicer):
    def __init__(self, config, rpc_manager: RPCManager) -> None:
        self._config = config
        self._rpc = rpc_manager

    # ── Register ────────────────────────────────────────────────────────────

    def Register(self, request, context):
        logger.info(
            "Register called by coordinator for worker_id=%s", request.worker_id
        )
        return worker_pb2.RegisterResponse(accepted=True, message="OK")

    # ── HealthCheck ─────────────────────────────────────────────────────────

    def HealthCheck(self, request, context):
        total_ram, free_ram = di.probe_ram()
        total_vram, free_vram = di.probe_vram()

        info = worker_pb2.DeviceInfo(
            arch=di.get_arch(),
            total_ram_bytes=total_ram,
            free_ram_bytes=free_ram,
            total_vram_bytes=total_vram,
            free_vram_bytes=free_vram,
            os_info=di.get_os_info(),
            llama_cpp_version=di.probe_llama_version(self._config.llama_rpc_binary),
            cpu_percent=di.probe_cpu(),
        )

        running = self._rpc.is_running()

        rpc_address = ""
        if running and self._rpc.port is not None:
            rpc_address = f"{self._config.rpc_host}:{self._rpc.port}"

        return worker_pb2.HealthResponse(
            status=worker_pb2.HealthResponse.HEALTHY,
            device_info=info,
            rpc_server_running=running,
            rpc_address=rpc_address,
        )

    # ── StartRPC ─────────────────────────────────────────────────────────────

    def StartRPC(self, request, context):
        try:
            address = self._rpc.start(
                host=self._config.rpc_host,
                port=request.port,
                mem_limit_mb=request.mem_limit_mb,
            )
            return worker_pb2.StartRPCResponse(success=True, address=address)
        except Exception as e:
            logger.error("Failed to start llama-rpc-server: %s", e)
            return worker_pb2.StartRPCResponse(success=False, error=str(e))

    # ── StopRPC ──────────────────────────────────────────────────────────────

    def StopRPC(self, request, context):
        try:
            self._rpc.stop(force=request.force)
            return worker_pb2.StopRPCResponse(success=True)
        except Exception as e:
            logger.error("Failed to stop llama-rpc-server: %s", e)
            return worker_pb2.StopRPCResponse(success=False, error=str(e))

    # ── Notify ───────────────────────────────────────────────────────────────

    def Notify(self, request, context):
        logger.info("[Coordinator] %s: %s", request.event, request.message)
        return worker_pb2.NotifyResponse(acknowledged=True)

    # ── StreamMetrics ────────────────────────────────────────────────────────

    def StreamMetrics(self, request, context):
        import time

        interval = max(1, request.interval_seconds)
        while context.is_active():
            _, free_ram = di.probe_ram()
            _, free_vram = di.probe_vram()
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=None)
            except Exception:
                cpu = 0.0

            yield worker_pb2.MetricsSnapshot(
                timestamp_ms=int(time.time() * 1000),
                free_ram_bytes=free_ram,
                free_vram_bytes=free_vram,
                cpu_percent=cpu,
            )
            time.sleep(interval)
