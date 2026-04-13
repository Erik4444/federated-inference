from __future__ import annotations

import asyncio
import logging
import signal

import uvicorn

from federated_inference.coordinator.config import ModelConfig, TopologyConfig
from federated_inference.coordinator.health import HealthLoop
from federated_inference.coordinator.llama_manager import CoordinatorState, LlamaManager
from federated_inference.coordinator.proxy import LlamaProxy
from federated_inference.coordinator.registry import WorkerRegistry, WorkerState

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Federated inference coordinator.

    Manages worker nodes, starts llama-server with the distributed
    --rpc flag, and exposes an OpenAI-compatible REST API.

    Usage::

        coordinator = Coordinator.from_config("topology.yaml", "model.yaml")
        await coordinator.start()   # blocks until stopped
    """

    def __init__(
        self,
        topology: TopologyConfig,
        model_config: ModelConfig,
    ) -> None:
        self.topology = topology
        self.model_config = model_config
        settings = topology.coordinator

        # Build registry
        self.registry = WorkerRegistry()
        for node in topology.enabled_workers:
            self.registry.register_node(node)

        # Proxy to local llama-server
        llama_base = f"http://{settings.llama_server_host}:{settings.llama_server_port}"
        self.proxy = LlamaProxy(llama_base)

        # LlamaManager
        self.llama_manager = LlamaManager(settings, model_config, self.registry)

        # Health loop
        self._health_loop = HealthLoop(self.registry, settings)

        # Optional UDP discovery
        self._discovery = None
        if settings.discovery:
            from federated_inference.coordinator.discovery import WorkerDiscovery
            self._discovery = WorkerDiscovery(
                registry=self.registry,
                port=settings.discovery_port,
                on_new_worker=self._on_discovered_worker,
            )

        # Wire state changes to llama_manager restart requests
        self.registry.on_state_change(self._on_worker_state_change)

    @classmethod
    def from_config(cls, topology_path: str, model_path: str) -> "Coordinator":
        return cls(
            topology=TopologyConfig.from_file(topology_path),
            model_config=ModelConfig.from_file(model_path),
        )

    def _on_worker_state_change(self, entry, new_state: WorkerState) -> None:
        if new_state in (WorkerState.ACTIVE, WorkerState.UNREACHABLE):
            self.llama_manager.request_restart()

    async def start(self) -> None:
        """Start coordinator services and serve the REST API. Blocks until stopped."""
        from federated_inference.coordinator.api import build_app

        app = build_app(self)
        settings = self.topology.coordinator

        # Start llama-server lifecycle in background
        llama_task = asyncio.create_task(self._run_llama_pipeline())
        # Start health loop
        self._health_loop.start()

        config = uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        logger.info(
            "Coordinator API listening on http://%s:%d",
            settings.host,
            settings.port,
        )

        try:
            await server.serve()
        finally:
            await self.stop()
            llama_task.cancel()

    async def _run_llama_pipeline(self) -> None:
        """Wait for enough workers to be ACTIVE, then start llama-server."""
        settings = self.topology.coordinator
        # Give the health loop a moment to connect to workers
        await asyncio.sleep(settings.health_check_interval_seconds * 1.5)
        # Start RPC servers on HEALTHY workers
        healthy = self.registry.workers_in_state(WorkerState.HEALTHY)
        if len(healthy) < settings.min_healthy_workers:
            logger.warning(
                "Only %d healthy workers, need %d. Waiting...",
                len(healthy),
                settings.min_healthy_workers,
            )
        for entry in healthy:
            await self._health_loop.start_rpc_on_worker(entry.id)
        # Hand off to LlamaManager
        await self.llama_manager.run()

    async def stop(self) -> None:
        logger.info("Shutting down coordinator...")
        await self._health_loop.stop()
        for entry in self.registry.all():
            await self._health_loop.stop_rpc_on_worker(entry.id)
        await self.llama_manager.stop()
        await self.proxy.close()
