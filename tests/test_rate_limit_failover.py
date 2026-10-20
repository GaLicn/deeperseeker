import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functions


class FakeResponse:
    status = 200

    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def content(self):
        return self

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_sse_rate_limit_becomes_http_429():
    event = {
        "type": "error",
        "content": "Messages too frequent. Try again later.",
        "clear_response": True,
        "finish_reason": "rate_limit_reached",
    }
    original_pow = functions.solve_create_pow
    original_post = functions.post_with_failover

    async def fake_pow(*args, **kwargs):
        return None

    async def fake_post(*args, **kwargs):
        return FakeResponse([f"data: {json.dumps(event)}\n".encode()])

    functions.solve_create_pow = fake_pow
    functions.post_with_failover = fake_post
    try:
        with pytest.raises(Exception, match=r"HTTP 429: DeepSeek rate limit reached"):
            await functions.send_message("chat-1", "token-1", "hello", 0).__anext__()
    finally:
        functions.solve_create_pow = original_pow
        functions.post_with_failover = original_post
