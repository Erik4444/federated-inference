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

from federated_inference.coordinator.coordinator import Coordinator
from federated_inference.coordinator.config import TopologyConfig, ModelConfig
from federated_inference.worker.worker import Worker
from federated_inference.worker.config import WorkerConfig
from federated_inference.client.client import FederatedInferenceClient

__all__ = [
    "Coordinator",
    "TopologyConfig",
    "ModelConfig",
    "Worker",
    "WorkerConfig",
    "FederatedInferenceClient",
]
