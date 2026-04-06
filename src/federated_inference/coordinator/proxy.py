from __future__ import annotations

import logging

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


class LlamaProxy:
    """Reverse proxy that forwards requests to the local llama-server."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=300.0)

    async def forward(self, request: Request) -> Response:
        path = request.url.path
        params = str(request.url.query)
        url = path + (f"?{params}" if params else "")

        body = await request.body()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }

        # Detect if the client wants streaming (SSE)
        is_streaming = False
        if body:
            import json
            try:
                payload = json.loads(body)
                is_streaming = payload.get("stream", False)
            except Exception:
                pass

        if is_streaming:
            return await self._stream(request.method, url, headers, body)
        else:
            return await self._proxy(request.method, url, headers, body)

    async def _proxy(self, method: str, url: str, headers: dict, body: bytes) -> Response:
        resp = await self._client.request(method, url, headers=headers, content=body)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    async def _stream(self, method: str, url: str, headers: dict, body: bytes) -> StreamingResponse:
        async def generator():
            async with self._client.stream(method, url, headers=headers, content=body) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(generator(), media_type="text/event-stream")

    async def close(self) -> None:
        await self._client.aclose()
