from __future__ import annotations

import asyncio
import logging
import sys

import click


@click.group()
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
def main(log_level: str) -> None:
    """federated-worker – Edge device worker for federated LLM inference."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


@main.command()
@click.option("--grpc-port", default=50051, show_default=True, help="gRPC listen port")
@click.option("--grpc-host", default="0.0.0.0", show_default=True)
@click.option("--worker-id", default="", help="Stable worker ID (defaults to hostname)")
@click.option("--rpc-host", default="0.0.0.0", show_default=True,
              help="Bind address for llama-rpc-server")
@click.option("--llama-rpc-binary", default="llama-rpc-server", show_default=True)
@click.option("--config", "config_file", default=None, type=click.Path(exists=True),
              help="Optional YAML worker config file")
def start(
    grpc_port: int,
    grpc_host: str,
    worker_id: str,
    rpc_host: str,
    llama_rpc_binary: str,
    config_file: str | None,
) -> None:
    """Start the worker gRPC server."""
    from federated_inference.worker.worker import Worker

    if config_file:
        worker = Worker.from_config(config_file)
    else:
        worker = Worker(
            grpc_port=grpc_port,
            grpc_host=grpc_host,
            worker_id=worker_id,
            rpc_host=rpc_host,
            llama_rpc_binary=llama_rpc_binary,
        )

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        click.echo("\nWorker stopped.")
