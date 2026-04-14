from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, Response
from federated_inference.coordinator.status import build_health_snapshot, build_metrics_snapshot

if TYPE_CHECKING:
    from federated_inference.coordinator.coordinator import Coordinator

logger = logging.getLogger(__name__)


def build_app(coordinator: "Coordinator") -> FastAPI:
    app = FastAPI(
        title="Federated Inference API",
        description="OpenAI-compatible API for federated multi-device LLM inference",
        version="0.1.0",
    )

    # ── Health / Status ──────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return build_health_snapshot(coordinator)

    @app.get("/metrics")
    async def metrics():
        return build_metrics_snapshot(coordinator)

    # ── OpenAI-compatible endpoints ──────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models():
        m = coordinator.model_config.model
        return {
            "object": "list",
            "data": [
                {
                    "id": m.path.split("/")[-1],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "federated-inference",
                }
            ],
        }

    async def _proxy_or_503(request: Request) -> Response:
        from federated_inference.coordinator.llama_manager import CoordinatorState

        state = coordinator.llama_manager.state
        # READY and RESTARTING are handled by the proxy (it queues during restart).
        # Only reject requests when llama-server has never been started.
        if state in (CoordinatorState.IDLE, CoordinatorState.STOPPING):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": f"Coordinator is not ready (state: {state.name}). "
                               "Check /health for worker status.",
                },
            )
        return await coordinator.proxy.forward(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _proxy_or_503(request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy_or_503(request)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _proxy_or_503(request)

    return app
