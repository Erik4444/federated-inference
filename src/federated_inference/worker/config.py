from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WorkerConfig:
    worker_id: str = ""
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    rpc_host: str = "0.0.0.0"
    llama_rpc_binary: str = "llama-rpc-server"

    @classmethod
    def from_file(cls, path: str | Path) -> "WorkerConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            worker_id=str(data.get("worker_id", "")),
            grpc_host=str(data.get("grpc_host", "0.0.0.0")),
            grpc_port=int(data.get("grpc_port", 50051)),
            rpc_host=str(data.get("rpc_host", "0.0.0.0")),
            llama_rpc_binary=str(data.get("llama_rpc_binary", "llama-rpc-server")),
        )
