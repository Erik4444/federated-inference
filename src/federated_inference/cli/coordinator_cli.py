from __future__ import annotations

import asyncio
import logging
import sys

import click


@click.group()
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
def main(log_level: str) -> None:
    """federated-coordinator – Coordinator for federated LLM inference."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


@main.command()
@click.option("--topology", default=None, type=click.Path(exists=True),
              help="Path to topology.yaml (optional when --discover is used)")
@click.option("--model", required=True, type=click.Path(exists=True),
              help="Path to model.yaml")
@click.option("--discover/--no-discover", default=False, show_default=True,
              help="Auto-discover workers via UDP broadcast (no topology.yaml needed)")
@click.option("--discovery-port", default=50052, show_default=True,
              help="UDP port to listen on for worker announcements")
def start(topology: str | None, model: str, discover: bool, discovery_port: int) -> None:
    """Start the coordinator (REST API + worker management)."""
    from federated_inference.coordinator.coordinator import Coordinator
    from federated_inference.coordinator.config import (
        TopologyConfig, CoordinatorSettings, ModelConfig
    )

    if topology:
        topo = TopologyConfig.from_file(topology)
    else:
        topo = TopologyConfig()   # empty: no static workers

    if discover:
        topo.coordinator.discovery = True
        topo.coordinator.discovery_port = discovery_port

    if not topology and not discover:
        raise click.UsageError("Provide --topology and/or --discover.")

    model_config = ModelConfig.from_file(model)
    coordinator = Coordinator(topology=topo, model_config=model_config)

    import signal

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)

        start_task = asyncio.create_task(coordinator.start())
        await stop_event.wait()
        await coordinator.stop()
        start_task.cancel()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    click.echo("\nCoordinator stopped.")


@main.command()
@click.option("--topology", required=True, type=click.Path(exists=True))
def status(topology: str) -> None:
    """Print the status of all configured workers."""
    import httpx
    from federated_inference.coordinator.config import TopologyConfig

    cfg = TopologyConfig.from_file(topology)
    url = f"http://{cfg.coordinator.host if cfg.coordinator.host != '0.0.0.0' else '127.0.0.1'}:{cfg.coordinator.port}/health"
    try:
        resp = httpx.get(url, timeout=5)
        import json
        data = resp.json()
        click.echo(f"Coordinator state: {data['coordinator_state']}")
        for w in data.get("workers", []):
            click.echo(f"  [{w['state']:12}] {w['id']}  rpc={w['rpc_address'] or '-'}")
    except Exception as e:
        click.echo(f"Could not reach coordinator at {url}: {e}", err=True)
        raise SystemExit(1)
