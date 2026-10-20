"""Token 池分配策略测试：在途最少优先、限流冷却与租约计数。

覆盖三项契约：
1. pick_token() 按「最少在途优先」分配，并列时依次按「最久未用」与 id
   升序定序；定序确定，且每次选中回写 last_used，使并列账号之间仍能
   轮转均摊。
2. mark_limited() 记录 limited_until 冷却窗口，冷却期内该账号不进入候选
   集，到期后由 pick_token() 当场翻回 ACTIVE 并重新参与分配。
3. TokenLease 的计数语义：release() 幂等、rebind() 转移占用且不重复登记
   ——计数只增不减会让某账号永久显得最忙，破坏负载均衡。
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functions
from functions import (
    TokenLease,
    add_token,
    inflight_snapshot,
    mark_active,
    mark_limited,
    pick_token,
    token_acquire,
    token_release,
)


@pytest.fixture()
def pool(tmp_path, monkeypatch):
    """将模块级数据库指向临时文件，并清空在途计数。"""
    monkeypatch.setattr(functions, "_db", str(tmp_path / "pool.db"))
    functions._inflight.clear()
    functions.init_db()
    yield functions
    functions._inflight.clear()


def _set_cooldown(token_id, until):
    """直接改写冷却截止时间，用于构造已到期 / 未到期两种冷却状态。"""
    conn = functions.get_db()
    conn.execute(
        "UPDATE tokens SET status = 'RATE_LIMITED', limited_until = ? WHERE id = ?",
        (until, token_id),
    )
    conn.commit()
    conn.close()


def _expire_cooldown(token_id):
    """把冷却截止时间改到过去，模拟冷却窗口已过。"""
    _set_cooldown(token_id, time.time() - 1)


def _set_last_used(token_id, stamp):
    conn = functions.get_db()
    conn.execute("UPDATE tokens SET last_used = ? WHERE id = ?", (stamp, token_id))
    conn.commit()
    conn.close()


def test_pick_token_prefers_least_inflight(pool):
    a = add_token("token-a", alias="a")
    b = add_token("token-b", alias="b")
    token_acquire(a)
    token_acquire(a)
    token_acquire(b)

    assert pick_token() == b


def test_pick_token_rotates_across_equal_inflight(pool):
    """在途数并列时按「最久未用」轮转，而不是固定选中同一账号。

    并列均摊由每次选中回写 last_used 保证：被选中者立刻变为「最近使用」，
    下一轮自然让位给尚未被选中的账号。
    """
    a = add_token("token-a", alias="a")
    b = add_token("token-b", alias="b")
    c = add_token("token-c", alias="c")

    picked = [pick_token() for _ in range(3)]

    assert set(picked) == {a, b, c}, "并列时必须轮转覆盖全部账号"


def test_pick_token_prefers_least_recently_used(pool):
    """在途数并列时，最久未使用的账号优先。"""
    a = add_token("token-a", alias="a")
    b = add_token("token-b", alias="b")
    now = time.time()
    _set_last_used(a, now)
    _set_last_used(b, now - 1000)

    assert pick_token() == b


def test_pick_token_exclude_skips_poisoned_token(pool):
    """exclude 供重试轮换使用：还有别的可用账号时不回落到被排除者。"""
    a = add_token("token-a", alias="a")
    b = add_token("token-b", alias="b")

    assert pick_token() == a
    assert pick_token(exclude=[a]) == b, "被排除的账号在仍有替代时不得被选中"


def test_pick_token_exclude_never_empties_candidates(pool):
    """排除集不得掏空候选集：无替代时回落到被排除的可用账号。

    exclude 的语义是「在有替代时避开」而非硬过滤：把候选集清空会让重试
    拿到 None 并直接失败，比再用一次被污染的账号更糟。注意回落只发生在
    可用集合内——冷却中的账号仍不会被选中。
    """
    a = add_token("token-a", alias="a")
    b = add_token("token-b", alias="b")
    mark_limited(b)

    assert pick_token(exclude=[a]) == a, "排除后无其它可用时必须回落到该账号"


def test_cooldown_expiry_auto_recovers_status(pool):
    """冷却到期后 pick_token() 当场把状态翻回 ACTIVE。

    旧实现只把到期账号「视为候选」，状态仍停留在 RATE_LIMITED，恢复
    完全依赖间隔更长的定时健康检查。
    """
    a = add_token("token-a", alias="a")
    add_token("token-b", alias="b")
    _expire_cooldown(a)

    assert pick_token() == a
    conn = functions.get_db()
    status, until, last_used = conn.execute(
        "SELECT status, limited_until, last_used FROM tokens WHERE id = ?", (a,)
    ).fetchone()
    conn.close()
    assert status == "ACTIVE", "冷却到期被选中后必须翻回 ACTIVE"
    assert until is None
    assert last_used is not None, "选中必须回写 last_used"


def test_all_cooling_returns_soonest_recovery(pool):
    """全部账号仍在冷却时返回最先恢复者，而非 id 最小者（有界等待）。"""
    a = add_token("token-a", alias="a")
    b = add_token("token-b", alias="b")
    now = time.time()
    _set_cooldown(a, now + 3600)
    _set_cooldown(b, now + 60)

    assert pick_token() == b


def test_limited_token_skipped_during_cooldown(pool):
    a = add_token("token-a", alias="a")
    b = add_token("token-b", alias="b")
    mark_limited(a)

    assert pick_token() == b


def test_limited_token_returns_after_cooldown(pool):
    a = add_token("token-a", alias="a")
    add_token("token-b", alias="b")
    mark_limited(a)

    _expire_cooldown(a)

    assert pick_token() == a, "冷却到期后限流账号必须重新进入候选集"


def test_sole_limited_token_is_still_fallback(pool):
    """单账号池在冷却期内仍返回该账号，保持有界等待而非直接失败。"""
    a = add_token("token-a", alias="a")
    mark_limited(a)

    assert pick_token() == a


def test_soft_cap_deprioritizes_but_never_starves(pool):
    """在途数达到软上限的账号排序靠后，但无其它可用时仍会被选中。"""
    a = add_token("token-a", alias="a")
    b = add_token("token-b", alias="b")
    cap = functions.TOKEN_CONCURRENCY_CAP
    assert cap >= 1

    for _ in range(cap):
        token_acquire(a)

    assert pick_token() == b, "打满软上限的账号应让位于空闲账号"

    # 反向：只有 a 可用（b 被限流）时，即便 a 已打满上限也必须被选中。
    mark_limited(b)
    assert pick_token() == a, "软上限不得饿死唯一可用账号"

    for _ in range(cap):
        token_release(a)


def test_mark_active_clears_cooldown_and_stamps_last_used(pool):
    a = add_token("token-a", alias="a")
    add_token("token-b", alias="b")
    mark_limited(a)
    assert functions.get_token(a)["status"] == "RATE_LIMITED"

    mark_active(a)

    conn = functions.get_db()
    status, until, last_used = conn.execute(
        "SELECT status, limited_until, last_used FROM tokens WHERE id = ?", (a,)
    ).fetchone()
    conn.close()
    assert status == "ACTIVE"
    assert until is None
    assert last_used is not None, "mark_active 必须刷新 last_used"


def test_mark_limited_records_future_cooldown(pool):
    a = add_token("token-a", alias="a")
    before = time.time()

    mark_limited(a)

    conn = functions.get_db()
    row = conn.execute(
        "SELECT status, limited_until FROM tokens WHERE id = ?", (a,)
    ).fetchone()
    conn.close()
    assert row[0] == "RATE_LIMITED"
    assert row[1] >= before + functions.RATE_LIMIT_COOLDOWN_SEC - 1


def test_lease_release_is_idempotent(pool):
    a = add_token("token-a", alias="a")
    lease = TokenLease(a)
    assert inflight_snapshot() == {a: 1}

    lease.release()
    lease.release()

    assert inflight_snapshot() == {}


def test_lease_rebind_moves_occupancy(pool):
    a = add_token("token-a", alias="a")
    b = add_token("token-b", alias="b")
    lease = TokenLease(a)

    lease.rebind(b)
    assert inflight_snapshot() == {b: 1}

    lease.rebind(b)
    assert inflight_snapshot() == {b: 1}, "重复绑定同一账号不得重复登记"

    lease.release()
    assert inflight_snapshot() == {}


def test_token_release_drops_zero_key(pool):
    token_acquire(7)
    token_release(7)
    assert 7 not in inflight_snapshot()

    token_release(7)
    assert 7 not in inflight_snapshot(), "归零后重复释放不得产生负数计数"
