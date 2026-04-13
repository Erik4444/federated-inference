"""
federated-inference
===================

Run large LLMs across multiple heterogeneous edge devices via pipeline parallelism.

Quick start
-----------

Coordinator (one machine)::

    from federated_inference import Coordinator
    import asyncio

    coordinator = Coordinator.from_config("topology.yaml", "model.yaml")
    asyncio.run(coordinator.start())

Worker (every edge device)::

    from federated_inference import Worker
    import asyncio

    worker = Worker(grpc_port=50051)
    asyncio.run(worker.start())

Client::

    from federated_inference import FederatedInferenceClient
    import asyncio

    async def main():
        client = FederatedInferenceClient("http://coordinator-host:8080")
        response = await client.chat.completions.create(
            model="llama-3-70b",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.choices[0].message.content)

    asyncio.run(main())

CLI
---

    federated-coordinator start --topology topology.yaml --model model.yaml
    federated-worker start --grpc-port 50051
"""

from __future__ import annotations

# Imports are deferred so that optional dependencies (uvicorn, fastapi, openai)
# are only required when the relevant class is actually used.  This lets the
# worker run on minimal installs (e.g. Android/Termux) without the coordinator
# or client extras being present.

__all__ = [
    "Coordinator",
    "TopologyConfig",
    "ModelConfig",
    "Worker",
    "WorkerConfig",
    "FederatedInferenceClient",
]


def __getattr__(name: str):
    if name in ("Coordinator", "TopologyConfig", "ModelConfig"):
        from federated_inference.coordinator import coordinator as _coord_mod
        from federated_inference.coordinator import config as _coord_cfg
        globals()["Coordinator"] = _coord_mod.Coordinator
        globals()["TopologyConfig"] = _coord_cfg.TopologyConfig
        globals()["ModelConfig"] = _coord_cfg.ModelConfig
        return globals()[name]

    if name in ("Worker", "WorkerConfig"):
        from federated_inference.worker.worker import Worker as _Worker
        from federated_inference.worker.config import WorkerConfig as _WorkerConfig
        globals()["Worker"] = _Worker
        globals()["WorkerConfig"] = _WorkerConfig
        return globals()[name]

    if name == "FederatedInferenceClient":
        from federated_inference.client.client import FederatedInferenceClient as _Client
        globals()["FederatedInferenceClient"] = _Client
        return _Client

    raise AttributeError(f"module 'federated_inference' has no attribute {name!r}")
