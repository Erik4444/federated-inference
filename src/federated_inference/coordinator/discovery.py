from __future__ import annotations

import asyncio
import json
import logging
import socket

from federated_inference.coordinator.config import WorkerNode
from federated_inference.coordinator.registry import WorkerRegistry

logger = logging.getLogger(__name__)

DISCOVERY_PORT = 50052


class WorkerDiscovery:
    """
    Listens for UDP broadcast announcements from workers and registers
    them dynamically in the WorkerRegistry.

    Works alongside static topology.yaml workers — both can be active at
    the same time.  If a worker is already registered (same worker_id),
    its host is updated in case it moved to a different IP (e.g. DHCP).
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        port: int = DISCOVERY_PORT,
        on_new_worker=None,
    ) -> None:
        self._registry = registry
        self._port = port
        # Optional async callback(worker_id) fired when a new worker appears
        self._on_new_worker = on_new_worker
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _listen(self) -> None:
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self._port))
        sock.setblocking(False)

        logger.info("Discovery: listening for workers on UDP port %d", self._port)
        try:
            while True:
                try:
                    data, addr = await loop.sock_recvfrom(sock, 1024)
                    await self._handle(data, addr[0])
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug("Discovery recv error: %s", e)
        finally:
            sock.close()

    async def _handle(self, data: bytes, source_ip: str) -> None:
        try:
            payload = json.loads(data.decode())
            worker_id = str(payload.get("worker_id") or source_ip)
            grpc_port = int(payload.get("grpc_port", 50051))
            rpc_port = int(payload.get("rpc_port", 8765))
            tags = list(payload.get("tags", []))
        except Exception:
            return

        existing = self._registry.get(worker_id)
        if existing is None:
            node = WorkerNode(
                id=worker_id,
                host=source_ip,
                grpc_port=grpc_port,
                rpc_port=rpc_port,
                tags=tags,
                enabled=True,
            )
            self._registry.register_node(node)
            logger.info(
                "Discovery: new worker '%s' at %s (grpc=%d rpc=%d tags=%s)",
                worker_id, source_ip, grpc_port, rpc_port, tags,
            )
            if self._on_new_worker:
                await self._on_new_worker(worker_id)
        elif existing.node.host != source_ip:
            logger.info(
                "Discovery: worker '%s' changed IP %s → %s",
                worker_id, existing.node.host, source_ip,
            )
            existing.node.host = source_ip
