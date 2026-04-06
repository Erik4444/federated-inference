from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class WorkerConfig(BaseModel):
    worker_id: str = ""          # defaults to hostname if empty
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    rpc_host: str = "0.0.0.0"   # bind address for llama-rpc-server
    llama_rpc_binary: str = "llama-rpc-server"

    @classmethod
    def from_file(cls, path: str | Path) -> "WorkerConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})
