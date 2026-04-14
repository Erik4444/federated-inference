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
              help="Bind address for RPC server")
@click.option("--llama-rpc-binary", default="rpc-server", show_default=True,
              help="GGML RPC server binary (rpc-server)")
@click.option("--config", "config_file", default=None, type=click.Path(exists=True),
              help="Optional YAML worker config file")
@click.option("--discover/--no-discover", default=False, show_default=True,
              help="Broadcast presence via UDP so the coordinator can find this worker automatically")
@click.option("--discovery-port", default=50052, show_default=True,
              help="UDP port for discovery broadcast (must match coordinator)")
@click.option("--rpc-port", default=8765, show_default=True,
              help="Port for llama-rpc-server (included in discovery payload)")
@click.option("--tags", default="", help="Comma-separated tags, e.g. arm64,termux")
def start(
    grpc_port: int,
    grpc_host: str,
    worker_id: str,
    rpc_host: str,
    llama_rpc_binary: str,
    config_file: str | None,
    discover: bool,
    discovery_port: int,
    rpc_port: int,
    tags: str,
) -> None:
    """Start the worker gRPC server."""
    from federated_inference.worker.worker import Worker

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    if config_file:
        worker = Worker.from_config(config_file)
    else:
        worker = Worker(
            grpc_port=grpc_port,
            grpc_host=grpc_host,
            worker_id=worker_id,
            rpc_host=rpc_host,
            llama_rpc_binary=llama_rpc_binary,
            discovery=discover,
            discovery_port=discovery_port,
            rpc_port=rpc_port,
            tags=tag_list,
        )

    import signal

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)

        start_task = asyncio.create_task(worker.start())
        await stop_event.wait()
        await worker.stop()
        start_task.cancel()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    click.echo("\nWorker stopped.")
