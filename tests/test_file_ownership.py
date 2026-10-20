"""文件归属固定（Stage 1 审计，B4）的回归测试。

/v1/files 上传此前随机挑选 token，而上游文件按账号隔离，因此聊天引用某个
file_id 时可能落到另一个账号并收到上游的 “file not found”——OpenAI 风格的
「上传后引用」流程因此在设计上就是坏的。现在上传会被钉住（files 表），首轮
聊天优先使用文件属主 token，后续轮次则把异主引用改挂到本会话自己的 token 上。
"""

import asyncio
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functions  # noqa: E402


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """把模块级数据库指向临时文件。"""
    monkeypatch.setattr(functions, "_db", str(tmp_path / "files.db"))
    functions.init_db()
    yield functions


def test_record_and_lookup_first_owner_wins(db):
    """登记后可按 file_id 反查属主，且先到者胜。"""
    db.record_file("file-1", 2)
    assert db.get_file_token("file-1") == 2
    # 别处再次上传同一个 file_id 不得夺走原属主。
    db.record_file("file-1", 5)
    assert db.get_file_token("file-1") == 2
    assert db.get_file_token("missing") is None


def test_referenced_file_ids_scan():
    """引用扫描须识别 OpenAI file 与 Anthropic file source，忽略 base64。"""
    import app as app_module

    msgs = [
        {"role": "user", "content": "plain text"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "use this"},
                {"type": "file", "file": {"file_id": "file-openai"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": "file-anthropic"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                # base64 内联数据不是文件引用。
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        },
    ]
    assert app_module._referenced_file_ids(msgs) == [
        "file-openai",
        "file-anthropic",
    ]
    assert app_module._referenced_file_ids([{"role": "user", "content": "str"}]) == []


def test_rehome_replaces_foreign_and_keeps_owned(db, monkeypatch):
    """异主引用须复制到本会话 token 上，自有与未登记的引用原样保留。"""
    import app as app_module

    db.record_file("file-owned", 1)
    db.record_file("file-foreign", 2)
    # file-legacy 早于注册表存在：属主未知。

    async def fake_get_file_content(fetch_token, file_id):
        yield "text/plain"
        yield b"data"

    async def fake_upload(file_bytes, file_name, file_content_type, auth_token):
        assert auth_token == "tok-1", "副本必须用本会话的 token 上传"
        yield ("uploaded", "ignored")
        yield (
            "success",
            {
                "file_id": "file-copy",
                "openai_timestamp": 0,
                "size": 4,
                "anthropic_timestamp": "1970-01-01T00:00:00Z",
            },
        )

    monkeypatch.setattr(
        app_module,
        "get_token",
        lambda tid: {"id": tid, "token": f"tok-{tid}", "status": "ACTIVE"},
    )
    monkeypatch.setattr(app_module, "get_file_content", fake_get_file_content)
    monkeypatch.setattr(app_module, "upload_file", fake_upload)

    tok = {"id": 1, "token": "tok-1", "status": "ACTIVE"}
    out = asyncio.run(
        app_module._rehome_foreign_files(
            ["file-owned", "file-foreign", "file-legacy"], 1, tok
        )
    )

    assert out == ["file-owned", "file-copy", "file-legacy"], out
    assert db.get_file_token("file-copy") == 1, "副本必须钉在本会话的 token 上"
    assert db.get_file_token("file-foreign") == 2, "原映射必须保持不变"


def test_rehome_keeps_reference_when_copy_fails(db, monkeypatch):
    """复制失败时保留原引用，不得静默丢弃文件引用。"""
    import app as app_module

    db.record_file("file-foreign", 2)

    async def failing_get_file_content(fetch_token, file_id):
        raise RuntimeError("upstream unavailable")
        yield  # pragma: no cover - 生成器标记

    monkeypatch.setattr(
        app_module,
        "get_token",
        lambda tid: {"id": tid, "token": f"tok-{tid}", "status": "ACTIVE"},
    )
    monkeypatch.setattr(app_module, "get_file_content", failing_get_file_content)

    tok = {"id": 1, "token": "tok-1", "status": "ACTIVE"}
    out = asyncio.run(app_module._rehome_foreign_files(["file-foreign"], 1, tok))
    assert out == ["file-foreign"]


def test_chat_prefers_file_owner_token_on_first_turn(db, monkeypatch):
    """首轮聊天引用的文件若只有一个已知属主，须改用该属主 token 运行。"""
    import app as app_module

    calls = {"tokens": [], "sessions": []}
    db.add_token("tok-1", "one")
    db.add_token("tok-2", "two")
    db.record_file("file-openai", 2)

    async def fake_sig(messages, model, scope=""):
        return "sig-file-owner"

    async def fake_create_chat(token):
        return "chat-owned"

    async def fake_files(messages, token, last_user_only=False):
        return []

    async def fake_prompt(messages, tools, model, is_first, rollover_summary=None):
        return "prompt"

    def fake_send(
        chat_id,
        auth_token,
        message,
        parent,
        thinking=False,
        search=False,
        file_ids_=None,
        search_sink=None,
    ):
        calls["tokens"].append((auth_token, tuple(file_ids_ or [])))

        async def gen():
            yield "ok"

        return gen()

    def fake_save(s, tid, sid, parent):
        calls["sessions"].append((tid, sid, parent))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what does this say?"},
                {"type": "file", "file": {"file_id": "file-openai"}},
            ],
        }
    ]

    async def fake_pick(*a, **k):
        return 1  # 调度器本会选中 token 1……

    async def direct_db(fn, *a, **kw):
        """本地 _db 走线程池，无法 await 测试用的 async 桩；这里直通并补 await。"""
        result = fn(*a, **kw)
        if inspect.isawaitable(result):
            result = await result
        return result

    monkeypatch.setattr(app_module, "_db", direct_db)
    monkeypatch.setattr(app_module, "get_auth_token", lambda: "tok")
    monkeypatch.setattr(app_module, "generate_signature", fake_sig)
    monkeypatch.setattr(app_module, "find_session", lambda s: None)
    monkeypatch.setattr(app_module, "pick_token", fake_pick)
    monkeypatch.setattr(app_module, "send_message", fake_send)
    monkeypatch.setattr(app_module, "create_new_chat", fake_create_chat)
    monkeypatch.setattr(app_module, "extract_and_upload_files", fake_files)
    monkeypatch.setattr(app_module, "build_prompt", fake_prompt)
    monkeypatch.setattr(app_module, "mark_limited", lambda tid: None)
    monkeypatch.setattr(app_module, "mark_active", lambda tid: None)
    monkeypatch.setattr(app_module, "delete_sessions_for_chat", lambda *a: None)
    monkeypatch.setattr(app_module, "save_session", fake_save)
    monkeypatch.setattr(app_module, "parse_tools", lambda t: ([], t))
    monkeypatch.setattr(
        app_module,
        "format_response",
        lambda text, model, messages, tools=None, **kw: text,
    )
    # 不需要上游调用：让流式预检直接返回原生成器。
    monkeypatch.setattr(app_module, "_preflight_stream", fake_preflight)

    try:
        result = asyncio.run(app_module.handle_chat(messages, "v4.1flash"))
    finally:
        app_module._chat_locks.clear()

    assert result == "ok"
    assert calls["tokens"] == [("tok-2", ())], (
        f"聊天必须跑在文件属主 token 上，实际为 {calls['tokens']}"
    )
    assert calls["sessions"] and calls["sessions"][0][0] == 2, (
        "会话必须记在属主 token 上"
    )


async def fake_preflight(gen):
    return gen
