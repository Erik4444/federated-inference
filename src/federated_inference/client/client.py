from __future__ import annotations


class FederatedInferenceClient:
    """
    OpenAI-compatible client pre-configured to point at the coordinator.

    You can use this class or configure ``openai.AsyncOpenAI`` directly::

        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url="http://coordinator:8080/v1", api_key="dummy")

    Example::

        client = FederatedInferenceClient("http://coordinator:8080")
        response = await client.chat.completions.create(
            model="llama-3",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    """

    def __init__(self, coordinator_url: str, api_key: str = "dummy") -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for FederatedInferenceClient. "
                "Install it with: pip install 'federated-inference[client]'"
            ) from e

        self._base_url = coordinator_url.rstrip("/")
        self._client = AsyncOpenAI(
            base_url=self._base_url + "/v1",
            api_key=api_key,
        )

    @property
    def chat(self):
        return self._client.chat

    @property
    def completions(self):
        return self._client.completions

    @property
    def embeddings(self):
        return self._client.embeddings

    async def health(self) -> dict:
        """Return the coordinator health status."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError("httpx is required for health checks") from e

        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{self._base_url}/health", timeout=5)
            return resp.json()
