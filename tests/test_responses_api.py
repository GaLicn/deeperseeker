"""Regression coverage for OpenAI Responses API compatibility."""

import asyncio
import json
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from app import (
    _responses_input_to_messages,
    format_responses_response,
    stream_responses_response,
)
from functions import _deepseek_event_finished, _deepseek_event_fragments
from plugin_helper import extract_tools

MODEL = "v4.1flash"
MESSAGES = [{"role": "user", "content": "hi"}]


def test_responses_function_tools_use_top_level_definition():
    tools = [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]
    prompt = asyncio.run(extract_tools(tools))
    assert "Tool: read_file" in prompt
    assert '"path"' in prompt


def test_invalid_and_builtin_tools_do_not_create_blank_entries():
    tools = [
        {"type": "function", "name": "", "parameters": {}},
        {"type": "web_search", "external_web_access": False},
    ]
    assert asyncio.run(extract_tools(tools)) is None


def test_responses_input_converts_tool_calls_and_outputs():
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Run it"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "shell",
            "arguments": '{"cmd":"dir"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "file.txt"},
    ]
    messages = _responses_input_to_messages(items)
    assert messages[0]["content"][0] == {"type": "text", "text": "Run it"}
    assert messages[1]["tool_calls"][0]["function"]["name"] == "shell"
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file.txt",
        "name": "shell",
    }


def test_non_stream_response_uses_official_output_types():
    result = {
        "id": "chatcmpl-123",
        "created": 1,
        "model": MODEL,
        "choices": [{"message": {"content": "OK", "tool_calls": None}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    response = format_responses_response(result, MODEL)
    assert response["id"] == "resp_123"
    assert response["status"] == "completed"
    assert response["output"][0]["content"][0]["type"] == "output_text"
    assert response["usage"]["input_tokens"] == 2


def test_batched_deepseek_fragments_are_not_lost():
    event = {
        "o": "BATCH",
        "v": [
            {
                "p": "response/fragments",
                "o": "APPEND",
                "v": [{"type": "RESPONSE", "content": "hello"}],
            },
            {"p": "quasi_status", "v": "FINISHED"},
        ],
    }
    assert _deepseek_event_fragments(event) == [("RESPONSE", "hello")]
    assert _deepseek_event_finished(event)


def test_responses_stream_emits_semantic_events():
    async def chat_stream():
        yield "O"
        yield "K"

    async def collect():
        return "".join(
            [
                frame
                async for frame in stream_responses_response(
                    chat_stream(), MODEL, MESSAGES, 1, "sess", "sig", []
                )
            ]
        )

    with mock.patch.multiple(
        app_module,
        mark_active=mock.DEFAULT,
        save_session=mock.DEFAULT,
        generate_signature_sync=mock.DEFAULT,
        delete_sessions_for_chat=mock.DEFAULT,
        _record_usage=mock.DEFAULT,
    ):
        streamed = asyncio.run(collect())

    assert "event: response.created" in streamed
    assert "event: response.output_text.delta" in streamed
    assert '"delta": "O"' in streamed
    assert "event: response.completed" in streamed
    assert "choices" not in streamed

    events = []
    for frame in streamed.split("\n\n"):
        if not frame.strip():
            continue
        lines = frame.strip().split("\n")
        data = json.loads(lines[1][len("data: ") :])
        events.append(data)
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert events[-1]["response"]["output"][-1]["content"][0]["text"] == "OK"


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
