import pytest
from federated_inference.coordinator.config import (
    CoordinatorSettings,
    ModelConfig,
    ModelSettings,
    TopologyConfig,
    WorkerNode,
)
from federated_inference.coordinator.registry import WorkerRegistry


@pytest.fixture
def worker_node():
    return WorkerNode(id="test-worker", host="127.0.0.1", grpc_port=50099, rpc_port=8799)


@pytest.fixture
def topology_config(worker_node):
    return TopologyConfig(
        coordinator=CoordinatorSettings(
            host="0.0.0.0",
            port=8080,
            min_healthy_workers=1,
            health_check_interval_seconds=5,
        ),
        workers=[worker_node],
    )


@pytest.fixture
def model_config():
    return ModelConfig(
        model=ModelSettings(path="/models/test.gguf"),
        llama_server_extra_flags=[],
    )


@pytest.fixture
def registry(worker_node):
    reg = WorkerRegistry()
    reg.register_node(worker_node)
    return reg
