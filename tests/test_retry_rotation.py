"""轮换式有界重试（B10，Stage 1 审计）的回归测试。

旧的空 SSE 重试会重新进入 handle_chat，从而重新抽取 pick_token() 的结果——
可能又落到同一个被污染的 token/会话上，正是 #33 症状（重复空响应）持续的原因。
现在重试预算为 MAX_UPSTREAM_ATTEMPTS 次，尝试之间带抖动退避，且每次重试都把
刚失败的 token id 作为排除集传下去，使 pick_token() 避开它。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402


async def _achat(session_id):
    return session_id


def _patches(calls, sig, send_impl, pick_impl):
    """构造 handle_chat 的桩集合；send_impl 决定每次 send 的行为。"""

    async def fake_sig(messages, model, scope=""):
        return sig

    def fake_get_token(tid):
        return {"id": tid, "token": f"tok-{tid}", "status": "ACTIVE"}

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
        return send_impl(calls, auth_token)

    async def fake_files(messages, token, last_user_only=False):
        return []

    async def fake_prompt(messages, tools, model, is_first, rollover_summary=None):
        return "prompt"

    return [
        ("get_auth_token", lambda: "tok"),
        ("generate_signature", fake_sig),
        ("find_session", lambda s: None),  # 每次尝试都走新建路径
        ("pick_token", pick_impl(calls)),
        ("get_token", fake_get_token),
        ("send_message", fake_send),
        ("create_new_chat", lambda tok: _achat("chat-x")),
        ("extract_and_upload_files", fake_files),
        ("build_prompt", fake_prompt),
        ("mark_limited", lambda tid: None),
        ("mark_active", lambda tid: None),
        ("delete_sessions_for_chat", lambda *a: None),
        ("save_session", lambda *a: None),
        ("record_file", lambda *a: None),
        ("parse_tools", lambda t: ([], t)),
        ("format_response", lambda text, model, messages, tools=None, **kw: text),
        ("_db", _direct_db),
    ]


async def _direct_db(fn, *a, **kw):
    """本地 _db 走线程池，无法 await 测试里的 async 桩；这里直通并补 await。"""
    result = fn(*a, **kw)
    if hasattr(result, "__await__"):
        result = await result
    return result


def _run(monkeypatch, patches, messages=None):
    saved = [(name, getattr(app_module, name)) for name, _ in patches]
    for name, fn in patches:
        monkeypatch.setattr(app_module, name, fn)

    async def scenario():
        app_module._chat_locks.clear()
        try:
            return await app_module.handle_chat(
                messages or [{"role": "user", "content": "hi"}], "test-model"
            )
        finally:
            app_module._chat_locks.clear()

    try:
        return asyncio.run(scenario())
    finally:
        for name, fn in saved:
            setattr(app_module, name, fn)


def test_retry_excludes_failed_token(monkeypatch):
    """通用上游故障必须换一个 token 重试（在还有其它可用账号时）。"""
    calls = {"send": 0, "picks": []}

    def pick_impl(c):
        def fake_pick(exclude=None):
            c["picks"].append(exclude)
            return 1 if c["send"] == 0 else 2

        return fake_pick

    def send_impl(c, auth_token):
        c["send"] += 1
        if c["send"] == 1:

            async def fail_gen():
                raise RuntimeError(
                    "Empty response from DeepSeek (no parseable SSE content)"
                )
                yield ""  # pragma: no cover

            return fail_gen()

        async def ok_gen():
            yield "recovered on token 2"

        return ok_gen()

    result = _run(
        monkeypatch, _patches(calls, "sig-retry-rotate", send_impl, pick_impl)
    )
    assert result == "recovered on token 2"
    assert calls["send"] == 2
    assert calls["picks"][0] is None, "首次选取不得排除任何 token"
    assert calls["picks"][1] == {1}, f"重试必须排除刚失败的 token：{calls['picks']}"


def test_retry_budget_is_bounded(monkeypatch):
    """持续失败必须在 MAX_UPSTREAM_ATTEMPTS 次发送后停止，返回 502 类错误。"""
    calls = {"send": 0, "picks": []}

    def pick_impl(c):
        def fake_pick(exclude=None):
            c["picks"].append(exclude)
            return 1

        return fake_pick

    def send_impl(c, auth_token):
        c["send"] += 1

        async def fail_gen():
            raise RuntimeError(
                "Empty response from DeepSeek (no parseable SSE content)"
            )
            yield ""  # pragma: no cover

        return fail_gen()

    result = _run(monkeypatch, _patches(calls, "sig-retry-bound", send_impl, pick_impl))
    assert calls["send"] == app_module.MAX_UPSTREAM_ATTEMPTS, calls
    assert getattr(result, "status_code", None) == 502, result
