from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Callable

from federated_inference.coordinator.config import WorkerNode

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    CONFIGURED = auto()
    CONNECTING = auto()
    HEALTHY = auto()
    RPC_STARTING = auto()
    ACTIVE = auto()
    DEGRADED = auto()
    UNREACHABLE = auto()


class WorkerEntry:
    def __init__(self, node: WorkerNode) -> None:
        self.node = node
        self.state: WorkerState = WorkerState.CONFIGURED
        self.rpc_address: str = ""
        self.device_info: dict = {}
        self.consecutive_failures: int = 0

    @property
    def id(self) -> str:
        return self.node.id

    def effective_mem_limit_mb(self, headroom_fraction: float = 0.15) -> int:
        """Memory cap (MiB) to pass to llama-rpc-server on this worker.

        Priority:
        1. Explicit ``mem_limit_mb`` in topology config (if > 0)
        2. Auto: VRAM if available, else RAM — minus ``headroom_fraction``
           to leave room for the OS and other processes.
        Returns 0 if no device info is available yet (no cap).
        """
        if self.node.mem_limit_mb > 0:
            return self.node.mem_limit_mb

        vram = self.device_info.get("free_vram_bytes", 0)
        ram = self.device_info.get("free_ram_bytes", 0)
        available_bytes = vram if vram > 0 else ram
        if available_bytes <= 0:
            return 0

        usable = int(available_bytes * (1.0 - headroom_fraction))
        return max(1, usable // (1024 * 1024))


class WorkerRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, WorkerEntry] = {}
        self._state_change_callbacks: list[Callable[[WorkerEntry, WorkerState], None]] = []
        self._lock = asyncio.Lock()

    def register_node(self, node: WorkerNode) -> None:
        self._entries[node.id] = WorkerEntry(node)

    def get(self, worker_id: str) -> WorkerEntry | None:
        return self._entries.get(worker_id)

    def all(self) -> list[WorkerEntry]:
        return list(self._entries.values())

    def workers_in_state(self, state: WorkerState) -> list[WorkerEntry]:
        return [e for e in self._entries.values() if e.state == state]

    def active_rpc_addresses(self) -> list[str]:
        return [e.rpc_address for e in self.workers_in_state(WorkerState.ACTIVE) if e.rpc_address]

    def transition(self, worker_id: str, new_state: WorkerState) -> None:
        """Transition a worker to a new state and fire callbacks.

        Safe to call from both sync and async contexts.  When called from
        within a running event loop the caller should use
        ``async_transition`` instead to serialise concurrent transitions.
        """
        entry = self._entries.get(worker_id)
        if entry is None:
            return
        old = entry.state
        if old == new_state:
            return
        entry.state = new_state
        logger.info("Worker %s: %s → %s", worker_id, old.name, new_state.name)
        for cb in self._state_change_callbacks:
            try:
                cb(entry, new_state)
            except Exception as e:
                logger.warning("State change callback error: %s", e)

    async def async_transition(self, worker_id: str, new_state: WorkerState) -> None:
        """Serialised state transition — prevents concurrent race conditions."""
        async with self._lock:
            self.transition(worker_id, new_state)

    def on_state_change(self, callback: Callable[[WorkerEntry, WorkerState], None]) -> None:
        self._state_change_callbacks.append(callback)

    def healthy_count(self) -> int:
        ready = {WorkerState.HEALTHY, WorkerState.RPC_STARTING, WorkerState.ACTIVE}
        return sum(1 for e in self._entries.values() if e.state in ready)
