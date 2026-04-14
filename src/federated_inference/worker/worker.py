from __future__ import annotations

import asyncio
import logging
import shutil
import socket
from concurrent import futures

import grpc

from federated_inference._generated import worker_pb2_grpc
from federated_inference.worker.config import WorkerConfig
from federated_inference.worker.rpc_manager import RPCManager
from federated_inference.worker.servicer import WorkerServicer

logger = logging.getLogger(__name__)


class Worker:
    """
    Federated inference worker.

    Runs a gRPC server that the coordinator connects to in order to:
    - Check health and device capabilities
    - Start / stop the llama-rpc-server subprocess

    Usage::

        worker = Worker(grpc_port=50051)
        await worker.start()   # blocks until stopped
    """

    def __init__(
        self,
        grpc_port: int = 50051,
        grpc_host: str = "0.0.0.0",
        worker_id: str = "",
        llama_rpc_binary: str = "rpc-server",
        rpc_host: str = "0.0.0.0",
        config: WorkerConfig | None = None,
        discovery: bool = False,
        discovery_port: int = 50052,
        rpc_port: int = 8765,
        tags: list[str] | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = WorkerConfig(
                worker_id=worker_id or socket.gethostname(),
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                rpc_host=rpc_host,
                llama_rpc_binary=llama_rpc_binary,
            )
        if not self._config.worker_id:
            self._config.worker_id = socket.gethostname()

        if not shutil.which(self._config.llama_rpc_binary):
            raise FileNotFoundError(
                f"RPC binary '{self._config.llama_rpc_binary}' not found in PATH. "
                f"Run: bash scripts/install_llama_cpp.sh"
            )
        self._rpc_manager = RPCManager(binary=self._config.llama_rpc_binary)
        self._server: grpc.Server | None = None

        self._broadcaster = None
        if discovery:
            from federated_inference.worker.discovery import DiscoveryBroadcaster
            self._broadcaster = DiscoveryBroadcaster(
                worker_id=self._config.worker_id,
                grpc_port=self._config.grpc_port,
                rpc_port=rpc_port,
                tags=tags or [],
                discovery_port=discovery_port,
            )

    @classmethod
    def from_config(cls, path: str) -> "Worker":
        return cls(config=WorkerConfig.from_file(path))

    async def start(self) -> None:
        """Start the gRPC server and block until stopped."""
        servicer = WorkerServicer(self._config, self._rpc_manager)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        worker_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, self._server)
        host = self._config.grpc_host
        # On Android/Termux, binding 0.0.0.0 can fail when the kernel doesn't
        # support dual-stack sockets. [::] (IPv6 any) with dual-stack is more
        # reliable; it also accepts IPv4 connections on all Android versions.
        if host in ("0.0.0.0", ""):
            grpc_address = f"[::]:{self._config.grpc_port}"
        else:
            grpc_address = f"{host}:{self._config.grpc_port}"
        address = f"{host}:{self._config.grpc_port}"
        bound = self._server.add_insecure_port(grpc_address)
        if not bound:
            raise RuntimeError(
                f"gRPC server could not bind to '{address}'. "
                "Possible causes:\n"
                "  1. Port is already in use (kill any existing worker process)\n"
                "  2. The host address does not exist on this device "
                "(use '0.0.0.0' to bind all interfaces)"
            )
        self._server.start()
        logger.info(
            "Worker '%s' gRPC server listening on %s",
            self._config.worker_id,
            address,
        )
        if self._broadcaster:
            self._broadcaster.start()
        # Run until server is stopped — use executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._server.wait_for_termination)

    async def stop(self) -> None:
        """Gracefully stop the worker."""
        if self._broadcaster:
            await self._broadcaster.stop()
        self._rpc_manager.stop()
        if self._server is not None:
            self._server.stop(grace=5)
            self._server = None
