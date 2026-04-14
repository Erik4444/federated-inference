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
        data = resp.json()
        click.echo(f"Coordinator state: {data['coordinator_state']}")
        for w in data.get("workers", []):
            click.echo(f"  [{w['state']:12}] {w['id']}  rpc={w['rpc_address'] or '-'}")
    except Exception as e:
        click.echo(f"Could not reach coordinator at {url}: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--topology", required=True, type=click.Path(exists=True),
              help="Path to topology.yaml (to find the coordinator address)")
@click.option("--interval", default=5, show_default=True,
              help="Refresh interval in seconds")
@click.option("--host", default=None,
              help="Override coordinator host (default: read from topology.yaml)")
@click.option("--port", default=None, type=int,
              help="Override coordinator port (default: read from topology.yaml)")
def dashboard(topology: str, interval: int, host: str | None, port: int | None) -> None:
    """Live terminal dashboard — shows RAM, CPU, and state of all workers."""
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text
        from rich import box
    except ImportError:
        click.echo("ERROR: 'rich' is not installed. Run: pip install 'federated-inference[coordinator]'", err=True)
        raise SystemExit(1)

    import httpx
    import time
    from federated_inference.coordinator.config import TopologyConfig

    cfg = TopologyConfig.from_file(topology)
    coord_host = host or (cfg.coordinator.host if cfg.coordinator.host != "0.0.0.0" else "127.0.0.1")
    coord_port = port or cfg.coordinator.port
    url = f"http://{coord_host}:{coord_port}/health"

    console = Console()

    # ── RAM thresholds for low-RAM warnings ─────────────────────────────────
    LOW_RAM_BYTES  = 4 * 1024 ** 3   # < 4 GB  → caution
    VERY_LOW_BYTES = 2 * 1024 ** 3   # < 2 GB  → warning

    def _ram_bar(free: int, total: int) -> Text:
        """Render a 10-char progress bar + GB label."""
        if total == 0:
            return Text("—", style="dim")
        used = total - free
        pct = used / total
        filled = int(pct * 10)
        bar = "█" * filled + "░" * (10 - filled)
        used_gb  = used  / 1024 ** 3
        total_gb = total / 1024 ** 3
        label = f"{bar} {used_gb:.1f}/{total_gb:.1f} GB"
        if total < VERY_LOW_BYTES:
            style = "bold red"
        elif total < LOW_RAM_BYTES:
            style = "yellow"
        elif pct > 0.9:
            style = "bold red"
        elif pct > 0.75:
            style = "yellow"
        else:
            style = "green"
        return Text(label, style=style)

    def _cpu_text(cpu: float) -> Text:
        if cpu <= 0:
            return Text("—", style="dim")
        label = f"{cpu:5.1f}%"
        if cpu >= 80:
            return Text(label, style="bold red")
        elif cpu >= 50:
            return Text(label, style="yellow")
        return Text(label, style="green")

    _STATE_STYLE = {
        "ACTIVE":       ("● ACTIVE",       "bold green"),
        "HEALTHY":      ("● HEALTHY",      "green"),
        "RPC_STARTING": ("◎ RPC_STARTING", "cyan"),
        "CONNECTING":   ("◌ CONNECTING",   "blue"),
        "CONFIGURED":   ("◌ CONFIGURED",   "dim"),
        "DEGRADED":     ("⚠ DEGRADED",     "yellow"),
        "UNREACHABLE":  ("✗ UNREACHABLE",  "bold red"),
    }

    def _state_text(state: str) -> Text:
        label, style = _STATE_STYLE.get(state, (state, "white"))
        return Text(label, style=style)

    def _ram_flag(total: int) -> str:
        if total == 0:
            return ""
        if total < VERY_LOW_BYTES:
            return " [bold red]⚠ VERY LOW RAM[/]"
        if total < LOW_RAM_BYTES:
            return " [yellow]⚠ LOW RAM[/]"
        return ""

    def build_table(data: dict | None, error: str | None, last_ok: float) -> Table:
        age = int(time.time() - last_ok) if last_ok else 0
        ts = time.strftime("%H:%M:%S")

        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold white",
            title=f"[bold]Federated Inference — Workers[/]  [dim]{ts}[/]",
            caption=f"[dim]Refreshing every {interval}s — Ctrl+C to quit[/]",
            expand=True,
        )
        table.add_column("Worker",      style="bold", min_width=14)
        table.add_column("State",       min_width=16)
        table.add_column("RAM (used/total)",  min_width=26)
        table.add_column("CPU",         min_width=8, justify="right")
        table.add_column("Device",      min_width=12)
        table.add_column("RPC",         min_width=18)

        if error:
            table.add_row(
                Text("—", style="dim"),
                Text(f"Cannot reach coordinator: {error}", style="bold red"),
                "", "", "", "",
            )
            return table

        if data is None:
            return table

        coordinator_state = data.get("coordinator_state", "?")
        workers = data.get("workers", [])

        total_cap_gb = 0.0
        active_count = 0

        for w in workers:
            wid    = w["id"]
            state  = w["state"]
            di     = w.get("device_info") or {}
            rpc    = w.get("rpc_address") or "—"

            total_ram = di.get("total_ram_bytes", 0)
            free_ram  = di.get("free_ram_bytes",  0)
            cpu_pct   = di.get("cpu_percent", 0.0)
            arch      = di.get("arch", "")
            os_info   = di.get("os_info", "")

            ram_bar   = _ram_bar(free_ram, total_ram)
            cpu_txt   = _cpu_text(cpu_pct)
            state_txt = _state_text(state)

            # device label: arch + low-RAM flag (rendered via markup in a new Text)
            flag = _ram_flag(total_ram)
            if flag:
                device_txt = Text.from_markup(f"{arch}{flag}")
            else:
                device_txt = Text(arch, style="dim" if state == "UNREACHABLE" else "")

            if state == "ACTIVE":
                active_count += 1
                total_cap_gb += (total_ram / 1024 ** 3)

            table.add_row(
                wid,
                state_txt,
                ram_bar if state not in ("UNREACHABLE", "CONFIGURED") else Text("—", style="dim"),
                cpu_txt if state not in ("UNREACHABLE", "CONFIGURED") else Text("—", style="dim"),
                device_txt,
                Text(rpc, style="dim"),
            )

        # Footer summary row
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

    last_data: dict | None = None
    last_error: str | None = None
    last_ok: float = 0.0

    def fetch() -> None:
        nonlocal last_data, last_error, last_ok
        try:
            resp = httpx.get(url, timeout=4)
            last_data  = resp.json()
            last_error = None
            last_ok    = time.time()
        except Exception as e:
            last_error = str(e)

    fetch()  # initial fetch before Live starts
    with Live(
        build_table(last_data, last_error, last_ok),
        console=console,
        refresh_per_second=1,
        screen=False,
    ) as live:
        try:
            while True:
                time.sleep(interval)
                fetch()
                live.update(build_table(last_data, last_error, last_ok))
        except KeyboardInterrupt:
            pass
    console.print("[dim]Dashboard stopped.[/]")
