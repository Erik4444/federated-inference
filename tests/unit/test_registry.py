import pytest
from federated_inference.coordinator.registry import WorkerRegistry, WorkerState
from federated_inference.coordinator.config import WorkerNode


def test_register_and_get():
    reg = WorkerRegistry()
    node = WorkerNode(id="w1", host="1.2.3.4")
    reg.register_node(node)
    entry = reg.get("w1")
    assert entry is not None
    assert entry.state == WorkerState.CONFIGURED


def test_transition():
    reg = WorkerRegistry()
    reg.register_node(WorkerNode(id="w1", host="1.2.3.4"))
    reg.transition("w1", WorkerState.HEALTHY)
    assert reg.get("w1").state == WorkerState.HEALTHY


def test_state_change_callback():
    reg = WorkerRegistry()
    reg.register_node(WorkerNode(id="w1", host="1.2.3.4"))
    transitions = []
    reg.on_state_change(lambda e, s: transitions.append((e.id, s)))
    reg.transition("w1", WorkerState.CONNECTING)
    reg.transition("w1", WorkerState.HEALTHY)
    assert ("w1", WorkerState.CONNECTING) in transitions
    assert ("w1", WorkerState.HEALTHY) in transitions


def test_workers_in_state():
    reg = WorkerRegistry()
    reg.register_node(WorkerNode(id="w1", host="1.2.3.4"))
    reg.register_node(WorkerNode(id="w2", host="1.2.3.5"))
    reg.transition("w1", WorkerState.ACTIVE)
    active = reg.workers_in_state(WorkerState.ACTIVE)
    assert len(active) == 1
    assert active[0].id == "w1"


def test_active_rpc_addresses():
    reg = WorkerRegistry()
    reg.register_node(WorkerNode(id="w1", host="1.2.3.4"))
    reg.transition("w1", WorkerState.ACTIVE)
    entry = reg.get("w1")
    entry.rpc_address = "1.2.3.4:8765"
    assert reg.active_rpc_addresses() == ["1.2.3.4:8765"]


def test_healthy_count():
    reg = WorkerRegistry()
    for i in range(3):
        reg.register_node(WorkerNode(id=f"w{i}", host=f"1.2.3.{i}"))
    reg.transition("w0", WorkerState.HEALTHY)
    reg.transition("w1", WorkerState.ACTIVE)
    reg.transition("w2", WorkerState.UNREACHABLE)
    assert reg.healthy_count() == 2
