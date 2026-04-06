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


class WorkerRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, WorkerEntry] = {}
        self._state_change_callbacks: list[Callable[[WorkerEntry, WorkerState], None]] = []

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

    def on_state_change(self, callback: Callable[[WorkerEntry, WorkerState], None]) -> None:
        self._state_change_callbacks.append(callback)

    def healthy_count(self) -> int:
        ready = {WorkerState.HEALTHY, WorkerState.RPC_STARTING, WorkerState.ACTIVE}
        return sum(1 for e in self._entries.values() if e.state in ready)
