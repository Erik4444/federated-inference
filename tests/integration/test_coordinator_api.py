"""
Integration test: tests the FastAPI coordinator endpoints using
httpx TestClient. LlamaManager is mocked to return READY state.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from federated_inference.coordinator.api import build_app
from federated_inference.coordinator.config import (
    CoordinatorSettings,
    ModelConfig,
    ModelSettings,
    TopologyConfig,
    WorkerNode,
)
from federated_inference.coordinator.llama_manager import CoordinatorState, LlamaManager
from federated_inference.coordinator.proxy import LlamaProxy
from federated_inference.coordinator.registry import WorkerRegistry, WorkerState


@pytest.fixture
def coordinator_mock():
    node = WorkerNode(id="w1", host="127.0.0.1", rpc_port=8765)
    registry = WorkerRegistry()
    registry.register_node(node)
    registry.transition("w1", WorkerState.ACTIVE)

    settings = CoordinatorSettings()
    model_cfg = ModelConfig(model=ModelSettings(path="/models/test.gguf"))
    topology = TopologyConfig(coordinator=settings, workers=[node])

    llama_mgr = MagicMock(spec=LlamaManager)
    llama_mgr.state = CoordinatorState.READY

    proxy = MagicMock(spec=LlamaProxy)

    coord = MagicMock()
    coord.registry = registry
    coord.llama_manager = llama_mgr
    coord.proxy = proxy
    coord.model_config = model_cfg
    return coord


@pytest.fixture
def client(coordinator_mock):
    app = build_app(coordinator_mock)
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["coordinator_state"] == "READY"
    assert len(data["workers"]) == 1
    assert data["workers"][0]["id"] == "w1"


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1


def test_chat_completions_503_when_not_ready(coordinator_mock):
    coordinator_mock.llama_manager.state = CoordinatorState.STARTING
    from federated_inference.coordinator.api import build_app
    app = build_app(coordinator_mock)
    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post("/v1/chat/completions", json={"model": "test", "messages": []})
    assert resp.status_code == 503
