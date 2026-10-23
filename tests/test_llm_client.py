"""Tests the real OpenAI-compatible client against a local fake HTTP server (offline)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from config import Settings
from src.generation import LLMError, OpenAICompatibleClient

MESSAGES = [{"role": "user", "content": "hello"}]


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def do_POST(self):
        key = self.headers.get("Authorization", "").replace("Bearer ", "")
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if key == "bad-key":
            status, body = 401, {"error": {"message": "Incorrect API key", "type": "auth"}}
        elif key == "limited-key":
            status, body = 429, {"error": {"message": "Rate limit reached", "type": "rate"}}
        elif key == "broken-key":
            status, body = 500, {"error": {"message": "server exploded", "type": "server"}}
        else:
            status, body = 200, {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hi there [1]"},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            }
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def server_url():
    server = HTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def _client(url: str, key: str) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        Settings(
            llm_provider="openai_compatible",
            llm_api_key=key,
            llm_base_url=url,
            llm_max_retries=0,
            llm_timeout_s=5,
        )
    )


def test_successful_call_returns_text_and_token_usage(server_url):
    response = _client(server_url, "good-key").generate(MESSAGES)
    assert response.text == "Hi there [1]"
    assert (response.prompt_tokens, response.completion_tokens) == (12, 4)


@pytest.mark.parametrize(
    "key, kind",
    [("bad-key", "auth"), ("limited-key", "rate_limit"), ("broken-key", "api")],
)
def test_http_errors_become_friendly_llm_errors_without_leaking_the_key(server_url, key, kind):
    with pytest.raises(LLMError) as info:
        _client(server_url, key).generate(MESSAGES)
    assert info.value.kind == kind
    assert key not in str(info.value)


def test_unreachable_server_is_reported_as_connection_error():
    with pytest.raises(LLMError) as info:
        _client("http://127.0.0.1:9/v1", "good-key").generate(MESSAGES)
    assert info.value.kind in {"connection", "timeout"}
