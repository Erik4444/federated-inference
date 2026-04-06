"""
Integration test: spins up a real gRPC WorkerServicer and calls it
via a real gRPC channel. The llama-rpc-server subprocess is mocked.
"""
import asyncio
import threading
from concurrent import futures
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from federated_inference._generated import worker_pb2, worker_pb2_grpc
from federated_inference.worker.config import WorkerConfig
from federated_inference.worker.rpc_manager import RPCManager
from federated_inference.worker.servicer import WorkerServicer

TEST_PORT = 50197


@pytest.fixture(scope="module")
def grpc_server():
    config = WorkerConfig(worker_id="test", grpc_port=TEST_PORT, llama_rpc_binary="/fake/llama-rpc-server")
    rpc_manager = RPCManager(binary="/fake/llama-rpc-server")
    servicer = WorkerServicer(config, rpc_manager)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{TEST_PORT}")
    server.start()
    yield server, servicer, rpc_manager
    server.stop(grace=0)


@pytest.fixture(scope="module")
def stub():
    channel = grpc.insecure_channel(f"127.0.0.1:{TEST_PORT}")
    return worker_pb2_grpc.WorkerServiceStub(channel)


def test_register(grpc_server, stub):
    response = stub.Register(
        worker_pb2.RegisterRequest(worker_id="coordinator", grpc_host="127.0.0.1", grpc_port=50099)
    )
    assert response.accepted is True


def test_health_check(grpc_server, stub):
    response = stub.HealthCheck(worker_pb2.HealthRequest())
    assert response.status == worker_pb2.HealthResponse.HEALTHY
    assert response.device_info.total_ram_bytes > 0


def test_health_check_rpc_not_running(grpc_server, stub):
    _, _, rpc_manager = grpc_server
    response = stub.HealthCheck(worker_pb2.HealthRequest())
    assert response.rpc_server_running is False
