from __future__ import annotations

import asyncio
import json
import logging
import socket

logger = logging.getLogger(__name__)

DISCOVERY_PORT = 50052


class DiscoveryBroadcaster:
    """
    Periodically announces this worker via UDP broadcast so coordinators
    can find it without manual IP configuration.

    Sends a small JSON packet to 255.255.255.255:<discovery_port> every
    ``interval`` seconds.  The coordinator's WorkerDiscovery counterpart
    listens for these packets and registers workers dynamically.
    """

    def __init__(
        self,
        worker_id: str,
        grpc_port: int,
        rpc_port: int,
        tags: list[str],
        discovery_port: int = DISCOVERY_PORT,
        interval: float = 10.0,
    ) -> None:
        self._payload = json.dumps({
            "worker_id": worker_id,
            "grpc_port": grpc_port,
            "rpc_port": rpc_port,
            "tags": tags,
        }).encode()
        self._port = discovery_port
        self._interval = interval
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _broadcast_loop(self) -> None:
        loop = asyncio.get_event_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        logger.info(
            "Discovery: broadcasting on UDP port %d every %.0fs",
            self._port, self._interval,
        )
        try:
            while True:
                try:
                    await loop.sock_sendto(
                        sock, self._payload, ("255.255.255.255", self._port)
                    )
                except Exception as e:
                    logger.debug("Discovery broadcast error: %s", e)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            raise
        finally:
            sock.close()
