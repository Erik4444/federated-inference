import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from federated_inference.coordinator.config import (
    CoordinatorSettings,
    ModelConfig,
    ModelSettings,
)
from federated_inference.coordinator.llama_manager import CoordinatorState, LlamaManager
from federated_inference.coordinator.registry import WorkerRegistry, WorkerState
from federated_inference.coordinator.config import WorkerNode


@pytest.fixture
def registry_with_active_worker():
    reg = WorkerRegistry()
    node = WorkerNode(id="w1", host="127.0.0.1", rpc_port=8765)
    reg.register_node(node)
    reg.transition("w1", WorkerState.ACTIVE)
    reg.get("w1").rpc_address = "127.0.0.1:8765"
    return reg


@pytest.fixture
def llama_manager(registry_with_active_worker, model_config):
    settings = CoordinatorSettings(llama_server_binary="/usr/local/bin/llama-server")
    return LlamaManager(settings, model_config, registry_with_active_worker)


@pytest.fixture
def model_config():
    return ModelConfig(model=ModelSettings(path="/models/test.gguf"))


def test_initial_state(llama_manager):
    assert llama_manager.state == CoordinatorState.IDLE


def test_request_restart_sets_event(llama_manager):
    llama_manager.request_restart()
    assert llama_manager._restart_event.is_set()


@pytest.mark.asyncio
async def test_stop_when_no_process(llama_manager):
    # Should not raise
    await llama_manager.stop()
    assert llama_manager.state == CoordinatorState.IDLE


@pytest.mark.asyncio
async def test_stop_terminates_process(llama_manager):
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)
    llama_manager._process = mock_proc
    llama_manager.state = CoordinatorState.READY
    await llama_manager.stop()
    mock_proc.terminate.assert_called_once()
