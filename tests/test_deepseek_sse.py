"""Regression coverage for DeepSeek SSE protocol variants."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functions


class FakeResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

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


def collect(generator):
    async def _collect():
        return "".join([chunk async for chunk in generator])

    return asyncio.run(_collect())


def test_native_sse_payload_and_done_are_parsed():
    original_post = functions.post_with_failover

    async def fake_post(*args, **kwargs):
        return FakeResponse(
            [
                b"event: message\n",
                b'data: {"p":"response/fragments","v":[{"type":"RESPONSE","content":"hello"}]}\n',
                b"\n",
                b"data: [DONE]\n\n",
            ]
        )

    functions.post_with_failover = fake_post
    try:
        original_pow = functions.solve_create_pow

        async def fake_pow(*args, **kwargs):
            return None

        functions.solve_create_pow = fake_pow
        try:
            assert collect(functions.send_message("chat", "token", "hi", 0)) == "hello"
        finally:
            functions.solve_create_pow = original_pow
    finally:
        functions.post_with_failover = original_post


def test_openai_delta_envelope_is_parsed():
    original_post = functions.post_with_failover
    original_pow = functions.solve_create_pow

    async def fake_post(*args, **kwargs):
        event = {"choices": [{"delta": {"content": "hello"}}]}
        return FakeResponse(
            [f"data: {json.dumps(event)}\n\n".encode(), b"data: [DONE]\n\n"]
        )

    async def fake_pow(*args, **kwargs):
        return None

    functions.post_with_failover = fake_post
    functions.solve_create_pow = fake_pow
    try:
        assert collect(functions.send_message("chat", "token", "hi", 0)) == "hello"
    finally:
        functions.post_with_failover = original_post
        functions.solve_create_pow = original_pow


def test_empty_done_stream_reports_protocol_details():
    original_post = functions.post_with_failover
    original_pow = functions.solve_create_pow

    async def fake_post(*args, **kwargs):
        return FakeResponse([b": keepalive\n", b"data: [DONE]\n\n"])

    async def fake_pow(*args, **kwargs):
        return None

    functions.post_with_failover = fake_post
    functions.solve_create_pow = fake_pow
    try:
        try:
            collect(functions.send_message("chat", "token", "hi", 0))
        except Exception as exc:
            message = str(exc)
        else:
            raise AssertionError("empty stream should fail")
        assert "payloads=1" in message
        assert "done=True" in message
    finally:
        functions.post_with_failover = original_post
        functions.solve_create_pow = original_pow
