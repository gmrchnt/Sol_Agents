import httpx

from config import LLM_BASE_URL, LLM_TIMEOUT


class LLMError(Exception):
    """Raised when the model server cannot produce a usable reply."""


class LLMClient:
    """Chat client for llama-server's OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        base_url: str = LLM_BASE_URL,
        timeout: httpx.Timeout = LLM_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }

        return self._send(payload).get("content") or ""

    def complete(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]],
        temperature: float | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        return self._send(payload)

    def _send(self, payload: dict[str, object]) -> dict[str, object]:
        try:
            response = self._http.post("chat/completions", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(
                f"Cannot reach the model server at {self.base_url}. "
                f"Is llama-server running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"Model server at {self.base_url} did not reply in time "
                f"({type(exc).__name__})."
            ) from exc
        except httpx.TransportError as exc:
            raise LLMError(
                f"Connection to the model server at {self.base_url} failed: {exc}"
            ) from exc

        if response.status_code != 200:
            raise LLMError(
                f"Model server returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"Unexpected reply from model server: {response.text[:500]}"
            ) from exc

        if not isinstance(message, dict):
            raise LLMError(
                f"Unexpected reply from model server: {response.text[:500]}"
            )

        return message
