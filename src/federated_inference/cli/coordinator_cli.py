from __future__ import annotations

import asyncio
import logging
import sys
import time

import click


def _coordinator_health_url(
    topology: str | None,
    host: str | None = None,
    port: int | None = None,
) -> str:
    coord_host = host
    coord_port = port

    if topology:
        from federated_inference.coordinator.config import TopologyConfig

        cfg = TopologyConfig.from_file(topology)
        coord_host = coord_host or cfg.coordinator.host
        coord_port = coord_port or cfg.coordinator.port

    coord_host = coord_host or "127.0.0.1"
    coord_port = coord_port or 8080
    if coord_host == "0.0.0.0":
        coord_host = "127.0.0.1"
    return f"http://{coord_host}:{coord_port}/health"


def _dashboard_components():
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text
        from rich import box
    except ImportError:
        click.echo("ERROR: 'rich' is not installed. Run: pip install 'federated-inference[coordinator]'", err=True)
        raise SystemExit(1)
    return Console, Live, Table, Text, box


def _ram_bar(Text, free: int, total: int):
    if total == 0:
        return Text("—", style="dim")
    used = total - free
    pct = used / total
    filled = int(pct * 10)
    bar = "█" * filled + "░" * (10 - filled)
    used_gb = used / 1024 ** 3
    total_gb = total / 1024 ** 3
    label = f"{bar} {used_gb:.1f}/{total_gb:.1f} GB"
    if total < 2 * 1024 ** 3:
        style = "bold red"
    elif total < 4 * 1024 ** 3:
        style = "yellow"
    elif pct > 0.9:
        style = "bold red"
    elif pct > 0.75:
        style = "yellow"
    else:
        style = "green"
    return Text(label, style=style)


def _cpu_text(Text, cpu: float):
    if cpu <= 0:
        return Text("—", style="dim")
    label = f"{cpu:5.1f}%"
    if cpu >= 80:
        return Text(label, style="bold red")
    if cpu >= 50:
        return Text(label, style="yellow")
    return Text(label, style="green")


_STATE_STYLE = {
    "ACTIVE": ("● ACTIVE", "bold green"),
    "HEALTHY": ("● HEALTHY", "green"),
    "RPC_STARTING": ("◎ RPC_STARTING", "cyan"),
    "CONNECTING": ("◌ CONNECTING", "blue"),
    "CONFIGURED": ("◌ CONFIGURED", "dim"),
    "DEGRADED": ("⚠ DEGRADED", "yellow"),
    "UNREACHABLE": ("✗ UNREACHABLE", "bold red"),
    "IDLE": ("○ IDLE", "dim"),
    "STARTING": ("◎ STARTING", "cyan"),
    "READY": ("● READY", "bold green"),
    "RESTARTING": ("↻ RESTARTING", "yellow"),
    "STOPPING": ("◌ STOPPING", "dim"),
}


def _state_text(Text, state: str):
    label, style = _STATE_STYLE.get(state, (state, "white"))
    return Text(label, style=style)


def _device_text(Text, arch: str, total_ram: int, state: str):
    if total_ram == 0:
        return Text(arch, style="dim" if state in ("UNREACHABLE", "CONFIGURED", "IDLE") else "")
    if total_ram < 2 * 1024 ** 3:
        return Text.from_markup(f"{arch} [bold red]⚠ VERY LOW RAM[/]")
    if total_ram < 4 * 1024 ** 3:
        return Text.from_markup(f"{arch} [yellow]⚠ LOW RAM[/]")
    return Text(arch, style="dim" if state in ("UNREACHABLE", "CONFIGURED", "IDLE") else "")


def _build_dashboard_table(Table, Text, box, data: dict | None, error: str | None, last_ok: float, interval: int):
    ts = time.strftime("%H:%M:%S")
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white",
        title=f"[bold]Federated Inference — Cluster[/]  [dim]{ts}[/]",
        caption=f"[dim]Refreshing every {interval}s — Ctrl+C to quit[/]",
        expand=True,
    )
    table.add_column("Node", style="bold", min_width=14)
    table.add_column("State", min_width=16)
    table.add_column("RAM (used/total)", min_width=26)
    table.add_column("CPU", min_width=8, justify="right")
    table.add_column("Device", min_width=12)
    table.add_column("Endpoint", min_width=18)

    if error:
        table.add_row(
            Text("—", style="dim"),
            Text(f"Cannot reach coordinator: {error}", style="bold red"),
            Text("—", style="dim"),
            Text("—", style="dim"),
            Text("—", style="dim"),
            Text("—", style="dim"),
        )
        return table

    if data is None:
        return table

    coordinator_state = data.get("coordinator_state", "?")
    coordinator = data.get("coordinator") or {}
    workers = data.get("workers", [])

    active_count = 0
    total_cap_gb = 0.0

    if coordinator:
        di = coordinator.get("device_info") or {}
        total_ram = di.get("total_ram_bytes", 0)
        free_ram = di.get("free_ram_bytes", 0)
        cpu_pct = di.get("cpu_percent", 0.0)
        arch = di.get("arch", "")
        endpoint = coordinator.get("llama_address") or coordinator.get("api_address") or "—"
        table.add_row(
            coordinator.get("id", "orchestrator"),
            _state_text(Text, coordinator.get("state", coordinator_state)),
            _ram_bar(Text, free_ram, total_ram),
            _cpu_text(Text, cpu_pct),
            _device_text(Text, arch, total_ram, coordinator.get("state", coordinator_state)),
            Text(endpoint, style="dim"),
        )
        table.add_section()

    if not workers:
        table.add_row(
            Text("—", style="dim"),
            Text("No workers registered yet", style="yellow"),
            Text("Waiting for health checks or discovery", style="dim"),
            Text("—", style="dim"),
            Text("—", style="dim"),
            Text("—", style="dim"),
        )

    for w in workers:
        wid = w["id"]
        state = w["state"]
        di = w.get("device_info") or {}
        rpc = w.get("rpc_address") or "—"

        total_ram = di.get("total_ram_bytes", 0)
        free_ram = di.get("free_ram_bytes", 0)
        cpu_pct = di.get("cpu_percent", 0.0)
        arch = di.get("arch", "")

        if state == "ACTIVE":
            active_count += 1
            total_cap_gb += total_ram / 1024 ** 3

        table.add_row(
            wid,
            _state_text(Text, state),
            _ram_bar(Text, free_ram, total_ram) if state not in ("UNREACHABLE", "CONFIGURED") else Text("—", style="dim"),
            _cpu_text(Text, cpu_pct) if state not in ("UNREACHABLE", "CONFIGURED") else Text("—", style="dim"),
            _device_text(Text, arch, total_ram, state),
            Text(rpc, style="dim"),
        )

    table.add_section()
    table.add_row(
        Text("TOTAL", style="bold"),
        Text(f"llama: {coordinator_state}", style="bold cyan"),
        Text(f"{total_cap_gb:.1f} GB active capacity", style="bold"),
        Text(""),
        Text(f"{active_count}/{len(workers)} active", style="bold"),
        Text(""),
    )
    return table


async def _run_dashboard_live(interval: int, fetch_data) -> None:
    Console, Live, Table, Text, box = _dashboard_components()
    console = Console()
    last_data: dict | None = None
    last_error: str | None = None
    last_ok: float = 0.0

    async def refresh() -> None:
        nonlocal last_data, last_error, last_ok
        try:
            last_data = await fetch_data()
            last_error = None
            last_ok = time.time()
        except Exception as e:
            last_error = str(e)

    await refresh()
    with Live(
        _build_dashboard_table(Table, Text, box, last_data, last_error, last_ok, interval),
        console=console,
        refresh_per_second=1,
        screen=False,
    ) as live:
        try:
            while True:
                await asyncio.sleep(interval)
                await refresh()
                live.update(_build_dashboard_table(Table, Text, box, last_data, last_error, last_ok, interval))
        except asyncio.CancelledError:
            raise


@click.group()
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
def main(log_level: str) -> None:
    """federated-coordinator – Coordinator for federated LLM inference."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
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
@click.option("--dashboard/--no-dashboard", default=False, show_default=True,
              help="Render the live status table in the same terminal")
@click.option("--dashboard-interval", default=5, show_default=True,
              help="Dashboard refresh interval in seconds")
def start(
    topology: str | None,
    model: str,
    discover: bool,
    discovery_port: int,
    dashboard: bool,
    dashboard_interval: int,
) -> None:
    """Start the coordinator (REST API + worker management)."""
    from federated_inference.coordinator.coordinator import Coordinator
    from federated_inference.coordinator.config import (
        TopologyConfig, CoordinatorSettings, ModelConfig
    )
    from federated_inference.coordinator.status import build_health_snapshot

    if topology:
        topo = TopologyConfig.from_file(topology)
    else:
        topo = TopologyConfig()   # empty: no static workers

    if discover:
        topo.coordinator.discovery = True
        topo.coordinator.discovery_port = discovery_port

    if not topology and not discover:
        raise click.UsageError("Provide --topology and/or --discover.")

    if dashboard:
        logging.disable(logging.CRITICAL)

    model_config = ModelConfig.from_file(model)
    coordinator = Coordinator(topology=topo, model_config=model_config)

    import signal

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)

        start_task = asyncio.create_task(coordinator.start())
        dashboard_task = None
        if dashboard:
            async def fetch_local():
                return build_health_snapshot(coordinator)

            dashboard_task = asyncio.create_task(
                _run_dashboard_live(dashboard_interval, fetch_local)
            )
        await stop_event.wait()
        if dashboard_task:
            dashboard_task.cancel()
            try:
                await dashboard_task
            except asyncio.CancelledError:
                pass
        await coordinator.stop()
        start_task.cancel()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    click.echo("\nCoordinator stopped.")


@main.command()
@click.option("--topology", default=None, type=click.Path(exists=True),
              help="Path to topology.yaml (optional when --host/--port are used)")
@click.option("--host", default=None,
              help="Coordinator host (defaults to topology.yaml or 127.0.0.1)")
@click.option("--port", default=None, type=int,
              help="Coordinator port (defaults to topology.yaml or 8080)")
def status(topology: str | None, host: str | None, port: int | None) -> None:
    """Print the current status of all workers known to the coordinator."""
    import httpx

    url = _coordinator_health_url(topology, host, port)
    try:
        resp = httpx.get(url, timeout=5)
        data = resp.json()
        click.echo(f"Coordinator state: {data['coordinator_state']}")
        workers = data.get("workers", [])
        if not workers:
            click.echo("  No workers registered yet.")
        for w in workers:
            click.echo(f"  [{w['state']:12}] {w['id']}  rpc={w['rpc_address'] or '-'}")
    except Exception as e:
        click.echo(f"Could not reach coordinator at {url}: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--topology", default=None, type=click.Path(exists=True),
              help="Path to topology.yaml (optional when --host/--port are used)")
@click.option("--interval", default=5, show_default=True,
              help="Refresh interval in seconds")
@click.option("--host", default=None,
              help="Override coordinator host (default: read from topology.yaml or 127.0.0.1)")
@click.option("--port", default=None, type=int,
              help="Override coordinator port (default: read from topology.yaml or 8080)")
def dashboard(topology: str | None, interval: int, host: str | None, port: int | None) -> None:
    """Live terminal dashboard — shows RAM, CPU, and state of all workers."""
    # Silence all loggers so they don't break the rich Live display.
    logging.disable(logging.CRITICAL)

    url = _coordinator_health_url(topology, host, port)
    async def _run() -> None:
        import httpx

        async with httpx.AsyncClient(timeout=4) as client:
            async def fetch_remote():
                resp = await client.get(url)
                return resp.json()

            await _run_dashboard_live(interval, fetch_remote)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
