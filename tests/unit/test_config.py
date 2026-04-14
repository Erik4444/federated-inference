import pytest
import yaml
from pathlib import Path

from federated_inference.coordinator.config import TopologyConfig, ModelConfig, WorkerNode
from federated_inference.worker.config import WorkerConfig


def test_topology_defaults():
    cfg = TopologyConfig()
    assert cfg.coordinator.port == 8080
    assert cfg.workers == []


def test_topology_enabled_workers():
    cfg = TopologyConfig(workers=[
        WorkerNode(id="a", host="1.2.3.4", enabled=True),
        WorkerNode(id="b", host="1.2.3.5", enabled=False),
    ])
    enabled = cfg.enabled_workers
    assert len(enabled) == 1
    assert enabled[0].id == "a"


def test_topology_from_file(tmp_path):
    data = {
        "coordinator": {"port": 9000},
        "workers": [{"id": "w1", "host": "10.0.0.1"}],
    }
    p = tmp_path / "topology.yaml"
    p.write_text(yaml.dump(data))
    cfg = TopologyConfig.from_file(p)
    assert cfg.coordinator.port == 9000
    assert cfg.workers[0].id == "w1"


def test_model_config_from_file(tmp_path):
    data = {"model": {"path": "/models/test.gguf", "context_length": 2048}}
    p = tmp_path / "model.yaml"
    p.write_text(yaml.dump(data))
    cfg = ModelConfig.from_file(p)
    assert cfg.model.path == "/models/test.gguf"
    assert cfg.model.context_length == 2048


def test_worker_config_defaults():
    cfg = WorkerConfig()
    assert cfg.grpc_port == 50051
    assert cfg.rpc_host == "0.0.0.0"
    assert cfg.llama_rpc_binary == "rpc-server"


def test_worker_config_from_file_defaults_to_installed_binary(tmp_path):
    p = tmp_path / "worker.yaml"
    p.write_text("{}")
    cfg = WorkerConfig.from_file(p)
    assert cfg.llama_rpc_binary == "rpc-server"
