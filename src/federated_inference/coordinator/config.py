from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class WorkerNode(BaseModel):
    id: str
    host: str
    grpc_port: int = 50051
    rpc_port: int = 8765
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    # Optional memory cap for llama-rpc-server on this worker (MiB).
    # 0 = auto: coordinator derives this from the worker's reported free
    # RAM/VRAM at the time it issues StartRPC.  Set explicitly to reserve
    # headroom or to limit a weak device (e.g. mem_limit_mb: 2048).
    mem_limit_mb: int = 0


class CoordinatorSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    llama_server_host: str = "127.0.0.1"
    llama_server_port: int = 8181
    llama_server_binary: str = "llama-server"
    health_check_interval_seconds: int = 10
    worker_timeout_seconds: int = 30
    min_healthy_workers: int = 1
    # UDP discovery: workers announce themselves so no static IPs are needed.
    # Set to true to enable; discovery_port must match the workers' setting.
    discovery: bool = False
    discovery_port: int = 50052


class TopologyConfig(BaseModel):
    coordinator: CoordinatorSettings = Field(default_factory=CoordinatorSettings)
    workers: list[WorkerNode] = Field(default_factory=list)

    @property
    def enabled_workers(self) -> list[WorkerNode]:
        return [w for w in self.workers if w.enabled]

    @classmethod
    def from_file(cls, path: str | Path) -> "TopologyConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})


class ModelSettings(BaseModel):
    path: str
    context_length: int = 4096
    n_gpu_layers: int = -1
    batch_size: int = 512
    # ubatch_size controls how many tokens flow through each RPC call.
    # Smaller values reduce per-transfer size at the cost of more round trips.
    # For WiFi RPC pipelines 128–256 is often faster than the default 512.
    # 0 = use llama-server default (same as batch_size).
    ubatch_size: int = 0
    threads: int = 4


class ModelConfig(BaseModel):
    model: ModelSettings
    llama_server_extra_flags: list[str] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})
