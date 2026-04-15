from __future__ import annotations

from typing import TYPE_CHECKING

from federated_inference.worker import device_info as di

if TYPE_CHECKING:
    from federated_inference.coordinator.coordinator import Coordinator


def _coordinator_bind_host(host: str) -> str:
    return "127.0.0.1" if host == "0.0.0.0" else host


def build_coordinator_snapshot(coordinator: "Coordinator") -> dict:
    settings = coordinator.topology.coordinator
    total_ram, free_ram = di.probe_ram()
    total_vram, free_vram = di.probe_vram()
    net_tx, net_rx = di.probe_net()

    return {
        "id": "orchestrator",
        "state": coordinator.llama_manager.state.name,
        "api_address": f"{_coordinator_bind_host(settings.host)}:{settings.port}",
        "llama_address": f"{settings.llama_server_host}:{settings.llama_server_port}",
        "device_info": {
            "arch": di.get_arch(),
            "total_ram_bytes": total_ram,
            "free_ram_bytes": free_ram,
            "total_vram_bytes": total_vram,
            "free_vram_bytes": free_vram,
            "os_info": di.get_os_info(),
            "cpu_percent": di.probe_cpu(),
            "net_tx_bytes_per_sec": net_tx,
            "net_rx_bytes_per_sec": net_rx,
        },
    }


def build_health_snapshot(coordinator: "Coordinator") -> dict:
    workers = [
        {
            "id": e.id,
            "state": e.state.name,
            "rpc_address": e.rpc_address,
            "device_info": e.device_info,
        }
        for e in coordinator.registry.all()
    ]
    return {
        "coordinator_state": coordinator.llama_manager.state.name,
        "coordinator": build_coordinator_snapshot(coordinator),
        "workers": workers,
    }


def build_metrics_snapshot(coordinator: "Coordinator") -> dict:
    coord = build_coordinator_snapshot(coordinator)
    coord_di = coord["device_info"]
    return {
        "coordinator": {
            "id": coord["id"],
            "state": coord["state"],
            "free_ram_bytes": coord_di.get("free_ram_bytes", 0),
            "free_vram_bytes": coord_di.get("free_vram_bytes", 0),
            "cpu_percent": coord_di.get("cpu_percent", 0.0),
        },
        "workers": [
            {
                "id": e.id,
                "state": e.state.name,
                "free_ram_bytes": e.device_info.get("free_ram_bytes", 0),
                "free_vram_bytes": e.device_info.get("free_vram_bytes", 0),
                "cpu_percent": e.device_info.get("cpu_percent", 0.0),
            }
            for e in coordinator.registry.all()
        ],
    }
