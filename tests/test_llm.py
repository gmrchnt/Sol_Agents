import json

import httpx
import pytest

from llm import LLMClient, LLMError

BASE = "http://model-server:1234/v1"
MESSAGES = [{"role": "user", "content": "hi"}]


def _client(handler) -> tuple[LLMClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return LLMClient("test-model", base_url=BASE, transport=httpx.MockTransport(record)), seen


def _reply(content: object) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_posts_to_openai_chat_completions_path() -> None:
    client, seen = _client(lambda r: _reply("ok"))
    client.chat(MESSAGES)
    assert str(seen[0].url) == f"{BASE}/chat/completions"


def test_sends_model_and_messages() -> None:
    client, seen = _client(lambda r: _reply("ok"))
    client.chat(MESSAGES)
    body = json.loads(seen[0].content)
    assert body["model"] == "test-model"
    assert body["messages"] == MESSAGES


def test_returns_message_content() -> None:
    client, _ = _client(lambda r: _reply("hello there"))
    assert client.chat(MESSAGES) == "hello there"


def test_null_content_becomes_empty_string() -> None:
    client, _ = _client(lambda r: _reply(None))
    assert client.chat(MESSAGES) == ""


def test_schema_is_sent_as_json_schema_response_format() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    client, seen = _client(lambda r: _reply("{}"))
    client.chat(MESSAGES, schema=schema)
    body = json.loads(seen[0].content)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }


def test_optional_fields_omitted_when_not_given() -> None:
    client, seen = _client(lambda r: _reply("ok"))
    client.chat(MESSAGES)
    body = json.loads(seen[0].content)
    assert "response_format" not in body
    assert "temperature" not in body


def test_temperature_zero_is_sent_not_dropped() -> None:
    client, seen = _client(lambda r: _reply("ok"))
    client.chat(MESSAGES, temperature=0)
    assert json.loads(seen[0].content)["temperature"] == 0


def test_http_error_status_raises() -> None:
    client, _ = _client(lambda r: httpx.Response(500, text="model not loaded"))
    with pytest.raises(LLMError, match="HTTP 500"):
        client.chat(MESSAGES)


def test_malformed_reply_raises() -> None:
    client, _ = _client(lambda r: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(LLMError, match="Unexpected reply"):
        client.chat(MESSAGES)


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
def test_unreachable_server_raises_with_address(error: type[httpx.TransportError]) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise error("no server", request=request)

    client, _ = _client(refuse)
    with pytest.raises(LLMError, match="Cannot reach the model server at http://model-server:1234/v1"):
        client.chat(MESSAGES)


def test_slow_model_raises_did_not_reply() -> None:
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client, _ = _client(stall)
    with pytest.raises(LLMError, match="did not reply in time"):
        client.chat(MESSAGES)
