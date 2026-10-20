"""PoW 预取池的回归测试。

覆盖四条关键行为：命中时不碰网络、过期条目被丢弃、不同 token / target_path 不
互相顶替、预取失败不影响主路径。池只是加速器，任何一条退化都必须表现为「变
慢」而不是「报错」——这是整个改动的安全边界。
"""

import asyncio
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functions

CHALLENGE = {
    "challenge": "ch-1",
    "salt": "salt-1",
    "signature": "sig-1",
    "expire_at": int(time.time()) + 3600,
    "difficulty": 0,
}


def _reset_pool():
    functions._POW_POOL.clear()
    for task in list(functions._pow_prefetch_tasks):
        task.cancel()
    functions._pow_prefetch_tasks.clear()


def _seed(target_path, auth_token, ttl=60.0, tag="1"):
    """直接往池里塞一个条目，绕开真实解算。

    tag 用来区分不同桶的 challenge，否则内容相同的条目会掩盖桶隔离是否真的生效。
    """
    challenge = dict(CHALLENGE, challenge=f"ch-{tag}", salt=f"salt-{tag}")
    header = functions._build_pow_header(challenge, 12345, target_path)
    functions._POW_POOL[(target_path, auth_token)] = (header, time.time() + ttl)
    return header


def test_pool_hit_skips_network_entirely():
    """池命中时必须一次网络调用都不发生。"""
    calls = []

    async def scenario():
        _reset_pool()
        expected = _seed("/api/v0/chat/completion", "tok-1")

        async def boom(*args, **kwargs):
            calls.append(args)
            raise AssertionError("池命中却打了网络")

        orig_challenge = functions.create_challange_pow
        functions.create_challange_pow = boom
        try:
            header = await functions.solve_create_pow(
                "/api/v0/chat/completion", "tok-1"
            )
        finally:
            functions.create_challange_pow = orig_challenge
        return header, expected

    header, expected = asyncio.run(scenario())
    assert header == expected
    assert calls == []


def test_expired_entry_is_discarded_and_resolved_synchronously():
    """过期条目必须被丢掉，并回落到同步求解。"""
    seen = []

    async def scenario():
        _reset_pool()
        _seed("/api/v0/chat/completion", "tok-1", ttl=-1.0)

        async def fake_challenge(target_path, auth_token):
            seen.append(target_path)
            return CHALLENGE

        async def fake_answer(pow_data):
            return 999

        orig_challenge = functions.create_challange_pow
        orig_answer = functions.find_pow_answer
        orig_prefetch = functions._POW_PREFETCH_ENABLED
        functions.create_challange_pow = fake_challenge
        functions.find_pow_answer = fake_answer
        functions._POW_PREFETCH_ENABLED = False
        try:
            return await functions.solve_create_pow("/api/v0/chat/completion", "tok-1")
        finally:
            functions.create_challange_pow = orig_challenge
            functions.find_pow_answer = orig_answer
            functions._POW_PREFETCH_ENABLED = orig_prefetch

    header = asyncio.run(scenario())
    assert seen == ["/api/v0/chat/completion"]
    decoded = json.loads(base64.b64decode(header))
    assert decoded["answer"] == 999
    assert decoded["algorithm"] == "DeepSeekHashV1"


def test_token_and_target_path_are_isolated():
    """池按 (target_path, token) 分桶，不跨桶取用。"""

    async def scenario():
        _reset_pool()
        chat_header = _seed("/api/v0/chat/completion", "tok-1", tag="tok1")
        _seed("/api/v0/file/upload_file", "tok-1", tag="upload")
        _seed("/api/v0/chat/completion", "tok-2", tag="tok2")

        return (
            functions._pow_pool_take("/api/v0/chat/completion", "tok-1"),
            functions._pow_pool_take("/api/v0/chat/completion", "tok-2"),
            functions._pow_pool_take("/api/v0/chat/completion", "tok-3"),
            chat_header,
        )

    got_chat, got_other_token, got_missing, chat_header = asyncio.run(scenario())
    assert got_chat == chat_header
    assert got_other_token is not None and got_other_token != chat_header
    assert got_missing is None


def test_entry_is_single_use():
    """一个预取结果只服务一个请求，取出即弹出。"""

    async def scenario():
        _reset_pool()
        _seed("/api/v0/chat/completion", "tok-1")
        first = functions._pow_pool_take("/api/v0/chat/completion", "tok-1")
        second = functions._pow_pool_take("/api/v0/chat/completion", "tok-1")
        return first, second

    first, second = asyncio.run(scenario())
    assert first is not None
    assert second is None


def test_prefetch_failure_is_swallowed():
    """预取失败绝不能冒泡到调用方。"""

    async def scenario():
        _reset_pool()

        async def boom(target_path, auth_token):
            raise RuntimeError("上游挂了")

        orig_challenge = functions.create_challange_pow
        functions.create_challange_pow = boom
        try:
            await functions._prefetch_pow_once("/api/v0/chat/completion", "tok-1")
        finally:
            functions.create_challange_pow = orig_challenge

    asyncio.run(scenario())
    assert functions._pow_pool_take("/api/v0/chat/completion", "tok-1") is None


def test_miss_schedules_prefetch_that_fills_the_pool():
    """未命中后必须真的预取成功，下一个请求才能命中——这是改动的全部价值。

    只验证「调度了任务」没有意义：如果预取从不成功，收益就是零。所以这里等预取
    任务真正跑完，再确认池里出现了可用条目。
    """

    async def scenario():
        _reset_pool()
        challenge_calls = []

        async def fake_challenge(target_path, auth_token):
            challenge_calls.append((target_path, auth_token))
            return dict(CHALLENGE, challenge=f"ch-{len(challenge_calls)}")

        async def fake_answer(pow_data):
            return 4242

        orig_challenge = functions.create_challange_pow
        orig_answer = functions.find_pow_answer
        orig_enabled = functions._POW_PREFETCH_ENABLED
        functions.create_challange_pow = fake_challenge
        functions.find_pow_answer = fake_answer
        functions._POW_PREFETCH_ENABLED = True
        try:
            first = await functions.solve_create_pow("/api/v0/chat/completion", "tok-1")
            # 让预取任务跑完（它内部有两次 await）。
            for _ in range(10):
                await asyncio.sleep(0)
            pending = [t for t in functions._pow_prefetch_tasks if not t.done()]
            if pending:
                await asyncio.gather(*pending)
            second = await functions.solve_create_pow(
                "/api/v0/chat/completion", "tok-1"
            )
            return first, second, challenge_calls
        finally:
            functions.create_challange_pow = orig_challenge
            functions.find_pow_answer = orig_answer
            functions._POW_PREFETCH_ENABLED = orig_enabled

    first, second, challenge_calls = asyncio.run(scenario())
    # 第一次同步求解（1 次 challenge），第二次命中池（不新增 challenge 调用）。
    assert len(challenge_calls) == 2
    assert first != second
    decoded = json.loads(base64.b64decode(second))
    assert decoded["challenge"] == "ch-2"


def test_prefetch_disabled_keeps_synchronous_path():
    """关掉开关后行为与改动前一致：每次都同步求解。"""

    async def scenario():
        _reset_pool()
        challenge_calls = []

        async def fake_challenge(target_path, auth_token):
            challenge_calls.append(target_path)
            return dict(CHALLENGE, challenge=f"ch-{len(challenge_calls)}")

        async def fake_answer(pow_data):
            return 1

        orig_challenge = functions.create_challange_pow
        orig_answer = functions.find_pow_answer
        orig_enabled = functions._POW_PREFETCH_ENABLED
        functions.create_challange_pow = fake_challenge
        functions.find_pow_answer = fake_answer
        functions._POW_PREFETCH_ENABLED = False
        try:
            await functions.solve_create_pow("/api/v0/chat/completion", "tok-1")
            await functions.solve_create_pow("/api/v0/chat/completion", "tok-1")
            return challenge_calls, len(functions._POW_POOL)
        finally:
            functions.create_challange_pow = orig_challenge
            functions.find_pow_answer = orig_answer
            functions._POW_PREFETCH_ENABLED = orig_enabled

    calls, pool_size = asyncio.run(scenario())
    assert len(calls) == 2
    assert pool_size == 0


def test_expire_at_units_are_normalized():
    """expire_at 按量级判断秒/毫秒，异常值退回 None。"""
    now = time.time()
    assert functions._pow_expire_epoch({"expire_at": int(now) + 60}) == int(now) + 60
    assert (
        functions._pow_expire_epoch({"expire_at": (int(now) + 60) * 1000})
        == int(now) + 60
    )
    assert functions._pow_expire_epoch({"expire_at": 0}) is None
    assert functions._pow_expire_epoch({"expire_at": None}) is None
    assert functions._pow_expire_epoch({}) is None


def test_store_respects_expire_at_over_ttl():
    """challenge 自带的有效期比固定 TTL 更早时，以更早的为准。"""

    async def scenario():
        _reset_pool()
        soon = {
            "challenge": "c",
            "salt": "s",
            "signature": "sig",
            "expire_at": int(time.time()) + 3,
        }
        functions._pow_pool_store("/api/v0/chat/completion", "tok-1", "header", soon)
        _header, usable_until = functions._POW_POOL[
            ("/api/v0/chat/completion", "tok-1")
        ]
        return usable_until - time.time()

    remaining = asyncio.run(scenario())
    # 3 秒有效期减去 5 秒安全余量 → 已经不可用。
    assert remaining < 1.0


def test_pool_is_bounded():
    """池满时淘汰最旧条目，不无限增长。"""

    async def scenario():
        _reset_pool()
        orig_max = functions._POW_POOL_MAX
        functions._POW_POOL_MAX = 3
        try:
            for i in range(6):
                functions._pow_pool_store(
                    "/api/v0/chat/completion", f"tok-{i}", "header", CHALLENGE
                )
            return len(functions._POW_POOL)
        finally:
            functions._POW_POOL_MAX = orig_max

    assert asyncio.run(scenario()) == 3
