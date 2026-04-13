from __future__ import annotations

__all__ = ["Coordinator", "TopologyConfig", "ModelConfig"]


def __getattr__(name: str):
    if name == "Coordinator":
        from federated_inference.coordinator.coordinator import Coordinator
        globals()["Coordinator"] = Coordinator
        return Coordinator
    if name in ("TopologyConfig", "ModelConfig"):
        from federated_inference.coordinator.config import TopologyConfig, ModelConfig
        globals()["TopologyConfig"] = TopologyConfig
        globals()["ModelConfig"] = ModelConfig
        return globals()[name]
    raise AttributeError(f"module 'federated_inference.coordinator' has no attribute {name!r}")
