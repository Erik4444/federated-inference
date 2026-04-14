from __future__ import annotations

import asyncio
import json
import logging

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# How long a request will wait for llama-server to become ready during a restart.
_READY_WAIT_TIMEOUT = 60.0
# Number of times to retry a request that failed due to a connection error.
_MAX_RETRIES = 2
_RETRY_DELAY = 1.0


class LlamaProxy:
    """Reverse proxy that forwards requests to the local llama-server.

    During restarts the proxy holds incoming requests (up to
    ``_READY_WAIT_TIMEOUT`` seconds) instead of returning 502 immediately.
    If the backend is temporarily unreachable (e.g. transient network hiccup
    or llama-server restart race), failed requests are retried automatically.
    """

    def __init__(self, base_url: str, ready_event: asyncio.Event | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=300.0)
        # Set by LlamaManager when llama-server is READY, cleared during restarts.
        self._ready = ready_event or asyncio.Event()
        # Tracks the number of in-flight requests so we can drain before restart.
        self._in_flight = 0
        self._drained = asyncio.Event()
        self._drained.set()  # initially no requests → already drained

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def drained(self) -> asyncio.Event:
        """Signalled when all in-flight requests have completed."""
        return self._drained

    async def forward(self, request: Request) -> Response:
        # Wait for llama-server to be ready (holds requests during restart).
        if not self._ready.is_set():
            logger.debug("Request waiting for llama-server to become ready...")
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=_READY_WAIT_TIMEOUT)
            except asyncio.TimeoutError:
                return Response(
                    content=json.dumps({
                        "error": "service_unavailable",
                        "message": "llama-server is restarting. Try again shortly.",
                    }),
                    status_code=503,
                    headers={"Retry-After": "5"},
                    media_type="application/json",
                )

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
            try:
                payload = json.loads(body)
                is_streaming = payload.get("stream", False)
            except (ValueError, KeyError):
                pass

        self._in_flight += 1
        self._drained.clear()
        try:
            if is_streaming:
                return await self._stream_with_retry(request.method, url, headers, body)
            else:
                return await self._proxy_with_retry(request.method, url, headers, body)
        finally:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._drained.set()

    async def _proxy_with_retry(
        self, method: str, url: str, headers: dict, body: bytes
    ) -> Response:
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await self._client.request(method, url, headers=headers, content=body)
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers=dict(resp.headers),
                )
            except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Proxy request failed (attempt %d/%d): %s — retrying in %.0fs",
                        attempt + 1, _MAX_RETRIES + 1, e, _RETRY_DELAY,
                    )
                    # Wait for server to come back (e.g. after restart).
                    await asyncio.sleep(_RETRY_DELAY)
                    if not self._ready.is_set():
                        try:
                            await asyncio.wait_for(
                                self._ready.wait(), timeout=_READY_WAIT_TIMEOUT
                            )
                        except asyncio.TimeoutError:
                            break

        return Response(
            content=json.dumps({
                "error": "bad_gateway",
                "message": f"llama-server unreachable after {_MAX_RETRIES + 1} attempts: {last_error}",
            }),
            status_code=502,
            media_type="application/json",
        )

    async def _stream_with_retry(
        self, method: str, url: str, headers: dict, body: bytes
    ) -> StreamingResponse | Response:
        """Streaming requests can only be retried before the first byte is sent."""
        try:
            async def generator():
                async with self._client.stream(method, url, headers=headers, content=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(generator(), media_type="text/event-stream")
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            # Connection failed before streaming started — fall back to non-streaming retry.
            logger.warning("Streaming request failed to connect: %s — falling back to retry", e)
            return await self._proxy_with_retry(method, url, headers, body)

    async def close(self) -> None:
        await self._client.aclose()
