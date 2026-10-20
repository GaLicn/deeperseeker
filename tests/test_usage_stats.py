"""Token 使用统计（usage_stats.py）的回归测试。

覆盖四块容易回归的逻辑：

* 峰谷判定 —— 窗口边界与周末都落在整点，判错一个小时会让整桶 token 按
  错误档位计价；
* cost_of_hour —— 高峰必须是低峰的 2 倍，且三个分项各自独立计价；
* record_usage —— 同 (小时, 模型, token, key) 必须 UPSERT 累加而非插新行；
* aggregate / resolve_range —— 逐小时计价求和、区间补齐与预设解析。

落库测试把 functions._db 指到 tmp_path，绝不碰真实的 deeperseeker.db。

Run:  python -m pytest tests/test_usage_stats.py   (pytest-compatible)
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functions  # noqa: E402
import usage_stats  # noqa: E402

TZ = usage_stats.BEIJING_TZ
MILLION = 1_000_000

# 2026-10-19 是周一（工作日），2026-10-17 是周六。
MON_PEAK = datetime(2026, 10, 19, 10, 0, tzinfo=TZ)  # 上午高峰窗口内
MON_OFF = datetime(2026, 10, 19, 3, 0, tzinfo=TZ)  # 凌晨低峰
SAT_PEAK_CLOCK = datetime(2026, 10, 17, 10, 0, tzinfo=TZ)  # 周末同一钟点


def _use_tmp_db(monkeypatch, tmp_path):
    """把 SQLite 文件换到 tmp_path 后建表。"""
    monkeypatch.setattr(functions, "_db", str(tmp_path / "usage.db"))
    functions.init_db()


# --- 峰谷判定 ---------------------------------------------------------------


def test_peak_windows_on_weekday():
    assert usage_stats.is_peak_hour(MON_PEAK) is True
    assert usage_stats.is_peak_hour(datetime(2026, 10, 19, 14, 0, tzinfo=TZ)) is True


def test_window_edges_are_exclusive():
    """窗口是闭开区间：12:00 与 18:00 已不在高峰，09:00 与 14:00 在。"""
    for hour in (12, 18):
        assert usage_stats.is_peak_hour(
            datetime(2026, 10, 19, hour, 0, tzinfo=TZ)
        ) is False, f"{hour}:00 不应算高峰"
    for hour in (9, 14):
        assert usage_stats.is_peak_hour(
            datetime(2026, 10, 19, hour, 0, tzinfo=TZ)
        ) is True, f"{hour}:00 应算高峰"


def test_weekend_is_always_off_peak():
    assert usage_stats.is_peak_hour(SAT_PEAK_CLOCK) is False
    assert usage_stats.is_peak_hour(
        datetime(2026, 10, 18, 10, 0, tzinfo=TZ)
    ) is False  # 周日


def test_bad_hour_key_falls_back_to_off_peak():
    """坏数据按低峰处理，宁可少算也不把成本抬高。"""
    assert usage_stats.is_peak_hour_key("not-a-key") is False
    assert usage_stats.is_peak_hour_key("") is False
    assert usage_stats.parse_hour_key(None) is None
    assert usage_stats.parse_day_key("2026-13-99") is None


# --- 计价 -------------------------------------------------------------------


def test_peak_costs_double_off_peak():
    buckets = {"cache_hit": 0, "cache_miss": MILLION, "output": 0}
    off = usage_stats.cost_of_hour(
        "deepseek-v4.1-flash", buckets, usage_stats.hour_key(MON_OFF)
    )
    peak = usage_stats.cost_of_hour(
        "deepseek-v4.1-flash", buckets, usage_stats.hour_key(MON_PEAK)
    )
    assert off["tier"] == "off_peak" and peak["tier"] == "peak"
    assert off["cache_miss"] == 1.0
    assert peak["cache_miss"] == 2.0
    assert peak["total"] == off["total"] * 2


def test_cost_splits_by_bucket():
    """命中、未命中、输出三档单价不同，必须各算各的。"""
    buckets = {"cache_hit": MILLION, "cache_miss": MILLION, "output": MILLION}
    charge = usage_stats.cost_of_hour(
        "deepseek-v4.1-flash", buckets, usage_stats.hour_key(MON_OFF)
    )
    assert charge["cache_hit"] == 0.02
    assert charge["cache_miss"] == 1.0
    assert charge["output"] == 4.0
    assert charge["total"] == 5.02


def test_unknown_model_falls_back_to_default():
    assert usage_stats.price_of("no-such-model") == usage_stats.PRICING["default"]


# --- 采样落库 ---------------------------------------------------------------


def test_record_usage_accumulates_same_hour(monkeypatch, tmp_path):
    """同一小时内的多次调用累加到同一行，不新增行。"""
    _use_tmp_db(monkeypatch, tmp_path)
    for _ in range(3):
        usage_stats.record_usage(
            "deepseek-v4.1-flash",
            token_id=1,
            api_key="sk-test",
            input_tokens=100,
            output_tokens=20,
            when=MON_PEAK,
        )
    rows = usage_stats.query_hours(
        usage_stats.hour_key(MON_PEAK), usage_stats.hour_key(MON_PEAK)
    )
    assert len(rows) == 1, f"同一小时应只有一行：{rows}"
    assert rows[0]["input_tokens"] == 300
    assert rows[0]["output_tokens"] == 60
    assert rows[0]["requests"] == 3


def test_record_usage_splits_other_hours_and_keys(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    usage_stats.record_usage("m", token_id=1, api_key="k", when=MON_PEAK)
    usage_stats.record_usage("m", token_id=1, api_key="k", when=MON_OFF)
    usage_stats.record_usage("m", token_id=1, api_key="other", when=MON_PEAK)
    usage_stats.record_usage("m", token_id=2, api_key="k", when=MON_PEAK)
    rows = usage_stats.query_hours("2026-10-19T00", "2026-10-19T23")
    assert len(rows) == 4, f"小时/模型/token/key 任一不同都应分行：{rows}"


def test_record_usage_keeps_null_primary_key_distinct(monkeypatch, tmp_path):
    """token_id 缺省记 -1、api_key 缺省记空串，否则 NULL 主键会重复插入。"""
    _use_tmp_db(monkeypatch, tmp_path)
    usage_stats.record_usage("m", when=MON_PEAK)
    usage_stats.record_usage("m", when=MON_PEAK)
    rows = usage_stats.query_hours(
        usage_stats.hour_key(MON_PEAK), usage_stats.hour_key(MON_PEAK)
    )
    assert len(rows) == 1
    assert rows[0]["token_id"] == -1
    assert rows[0]["api_key"] == ""
    assert rows[0]["requests"] == 2


def test_clear_usage_empties_table(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    usage_stats.record_usage("m", token_id=1, when=MON_PEAK)
    usage_stats.clear_usage()
    assert usage_stats.query_hours("2026-01-01T00", "2026-12-31T23") == []


# --- 区间聚合 ---------------------------------------------------------------


def test_aggregate_prices_each_hour_at_its_own_tier(monkeypatch, tmp_path):
    """区间总量绝不能套一次价：高峰桶与低峰桶必须分别计价再求和。"""
    _use_tmp_db(monkeypatch, tmp_path)
    buckets = dict(input_tokens=MILLION, output_tokens=0)
    usage_stats.record_usage("deepseek-v4.1-flash", token_id=1, when=MON_PEAK, **buckets)
    usage_stats.record_usage("deepseek-v4.1-flash", token_id=1, when=MON_OFF, **buckets)

    rows = usage_stats.query_hours("2026-10-19T00", "2026-10-19T23")
    stats = usage_stats.aggregate(rows, "2026-10-19", "2026-10-19")

    assert stats["cache_miss"] == 2 * MILLION
    assert stats["cost_peak"] == 2.0
    assert stats["cost_off_peak"] == 1.0
    assert stats["cost"] == 3.0
    # 若误把区间总量（2M 未命中）套一次低峰价，会得到 2.0 而非 3.0。
    assert stats["cost"] != 2.0


def test_aggregate_fills_empty_days(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    # 必须带真实用量：tokens 为 0 的桶不计入 active_days。
    usage_stats.record_usage("m", token_id=1, input_tokens=100, when=MON_PEAK)
    rows = usage_stats.query_hours("2026-10-17T00", "2026-10-19T23")
    stats = usage_stats.aggregate(rows, "2026-10-17", "2026-10-19")
    assert [d["day"] for d in stats["daily"]] == [
        "2026-10-17",
        "2026-10-18",
        "2026-10-19",
    ]
    assert stats["daily"][0]["tokens"] == 0
    assert stats["active_days"] == 1
    assert len(stats["hourly"]) == 3 * 24


def test_aggregate_ranks_and_hit_rate(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    usage_stats.record_usage(
        "m-a", token_id=1, api_key="k1", input_tokens=500, cache_read_tokens=500,
        output_tokens=100, when=MON_PEAK,
    )
    usage_stats.record_usage(
        "m-b", token_id=2, api_key="k2", input_tokens=900, output_tokens=100,
        when=MON_PEAK,
    )
    rows = usage_stats.query_hours("2026-10-19T00", "2026-10-19T23")
    stats = usage_stats.aggregate(
        rows,
        "2026-10-19",
        "2026-10-19",
        labels={"tokens": {1: "主号"}, "keys": {}},
    )

    # tokens 口径 = 未缓存输入 + 输出 + 缓存读 + 缓存写
    assert stats["models"][0]["label"] == "m-b"
    assert stats["models"][0]["tokens"] == 1000
    assert stats["models"][1]["tokens"] == 600
    assert stats["top_model"] == "m-b"

    # 号池排行按 token 量降序：token 2（1000）在 token 1（600）之前；
    # 别名为空/未登记的成员退回 "Token #id"。
    assert [e["label"] for e in stats["pool"]] == ["Token #2", "主号"]
    assert {e["label"] for e in stats["keys"]} == {"k1", "k2"}
    # 命中 500 / (500 + 900) 输入
    assert abs(stats["cache_hit_rate"] - 500 / 1400) < 1e-9


def test_short_key_never_leaks_full_secret():
    assert usage_stats._short_key("") == "未标注"
    assert usage_stats._short_key("short") == "short"
    long_key = "sk-" + "a" * 60
    shown = usage_stats._short_key(long_key)
    assert long_key not in shown
    assert shown.startswith("sk-aaa") and shown.endswith("aaaa")


# --- 区间解析 ---------------------------------------------------------------


def test_resolve_range_presets():
    from_day, to_day, from_hour, to_hour = usage_stats.resolve_range({"range": ["7"]})
    assert from_hour is None and to_hour is None
    start = usage_stats.parse_day_key(from_day)
    end = usage_stats.parse_day_key(to_day)
    assert (end - start).days == 6


def test_resolve_range_24h_has_hour_bounds():
    """24 小时是滚动窗口，必须额外给出小时边界，否则日粒度无法还原峰谷。"""
    from_day, to_day, from_hour, to_hour = usage_stats.resolve_range(
        {"range": ["24h"]}
    )
    assert from_hour is not None and to_hour is not None
    delta = usage_stats.parse_hour_key(to_hour) - usage_stats.parse_hour_key(from_hour)
    assert delta == timedelta(hours=23)


def test_resolve_range_custom_and_fallback():
    assert usage_stats.resolve_range(
        {"range": ["custom"], "from": ["2026-10-01"], "to": ["2026-10-05"]}
    ) == ("2026-10-01", "2026-10-05", None, None)
    # 非法 custom 退回默认 30 天
    from_day, to_day, _, _ = usage_stats.resolve_range(
        {"range": ["custom"], "from": ["bad"], "to": [""]}
    )
    assert (usage_stats.parse_day_key(to_day) - usage_stats.parse_day_key(from_day)).days == 29
    # 未知预设同样退回 30 天
    from_day, to_day, _, _ = usage_stats.resolve_range({"range": ["nope"]})
    assert (usage_stats.parse_day_key(to_day) - usage_stats.parse_day_key(from_day)).days == 29


def test_parse_range_params_reads_query_string():
    params = usage_stats.parse_range_params("range=7&from=2026-10-01")
    assert params["range"] == ["7"]
    assert params["from"] == ["2026-10-01"]


def test_resolve_range_all_uses_earliest():
    """``all`` 的起点只能来自采样表；空库时退回最近 365 天兜底。"""
    frm, to, from_hour, to_hour = usage_stats.resolve_range(
        {"range": ["all"]}, earliest="2026-01-05"
    )
    assert frm == "2026-01-05"
    assert from_hour is None and to_hour is None

    frm2, to2, _, _ = usage_stats.resolve_range({"range": ["all"]})
    assert (
        usage_stats.parse_day_key(to2) - usage_stats.parse_day_key(frm2)
    ).days == 364


def test_range_labels_cover_every_preset():
    """界面档位与后端可解析的预设必须一一对应，漏一个就会出现死链。"""
    assert {k for k, _ in usage_stats.RANGE_LABELS} == usage_stats.RANGE_PRESETS


# --- 服务商归因 -------------------------------------------------------------


def test_provider_and_model_name_split():
    assert usage_stats.provider_of("anthropic/claude-3") == "anthropic"
    assert usage_stats.model_name_of("anthropic/claude-3") == "claude-3"
    # 裸名归默认服务商，显示名原样保留。
    assert usage_stats.provider_of("v4.1flash") == "deepseek"
    assert usage_stats.model_name_of("v4.1flash") == "v4.1flash"
    assert usage_stats.provider_of("") == "deepseek"
    assert usage_stats.provider_of(None) == "deepseek"


def test_raw_model_name_drives_provider_split(monkeypatch, tmp_path):
    """库里存客户端原始模型名，按服务商拆分才有意义。"""
    _use_tmp_db(monkeypatch, tmp_path)
    usage_stats.record_usage(
        "anthropic/claude-sonnet-4", token_id=1, input_tokens=100, when=MON_PEAK
    )
    usage_stats.record_usage(
        "v4.1flash", token_id=1, input_tokens=100, when=MON_PEAK
    )
    rows = usage_stats.query_hours("2026-10-19T00", "2026-10-19T23")
    assert {r["model"] for r in rows} == {
        "anthropic/claude-sonnet-4",
        "v4.1flash",
    }

    stats = usage_stats.aggregate(rows, "2026-10-19", "2026-10-19")
    by_key = {p["key"]: p for p in stats["providers"]}
    assert set(by_key) == {"anthropic", "deepseek"}
    assert by_key["anthropic"]["models"] == ["claude-sonnet-4"]
    assert by_key["deepseek"]["models"] == ["v4.1flash"]


# --- 排行补充 ---------------------------------------------------------------


def test_aggregate_reports_top_and_second_model(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    usage_stats.record_usage("m-a", token_id=1, input_tokens=100, when=MON_PEAK)
    usage_stats.record_usage("m-b", token_id=1, input_tokens=300, when=MON_PEAK)
    rows = usage_stats.query_hours("2026-10-19T00", "2026-10-19T23")
    stats = usage_stats.aggregate(rows, "2026-10-19", "2026-10-19")
    assert stats["top_model"] == "m-b"
    assert stats["second_model"] == "m-a"


def test_aggregate_single_model_has_no_second(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    usage_stats.record_usage("only", token_id=1, input_tokens=100, when=MON_PEAK)
    rows = usage_stats.query_hours("2026-10-19T00", "2026-10-19T23")
    stats = usage_stats.aggregate(rows, "2026-10-19", "2026-10-19")
    assert stats["second_model"] == ""


def test_hit_rate_series_zero_denominator_is_zero():
    series = [
        {"cache_hit": 5, "cache_miss": 5},
        {"cache_hit": 0, "cache_miss": 0},
        {"cache_hit": 9, "cache_miss": 1},
    ]
    assert usage_stats.hit_rate_series(series) == [0.5, 0.0, 0.9]


# --- 热力图 -----------------------------------------------------------------


def test_build_heatmap_aligns_weeks_to_monday():
    """网格从起始日所在周的周一到结束日所在周的周日，两端补齐。"""
    daily = [
        {"day": "2026-10-16", "tokens": 100},
        {"day": "2026-10-19", "tokens": 50},
    ]
    grid = usage_stats.build_heatmap(daily, "2026-10-14", "2026-10-19")
    weeks = grid["weeks"]
    # 10-12（周一）.. 10-25（周日）共 14 天 = 2 周
    assert len(weeks) == 2
    assert all(len(w) == 7 for w in weeks)
    assert weeks[0][0]["day"] == "2026-10-12"
    assert weeks[-1][-1]["day"] == "2026-10-25"
    # 区间外补齐的格子没有数据，档位为 0
    assert weeks[0][0]["level"] == 0 and weeks[0][0]["tokens"] == 0
    assert grid["peak"] == 100
    assert grid["active_days"] == 2


def test_build_heatmap_levels_scale_to_peak():
    daily = [
        {"day": "2026-10-16", "tokens": 100},
        {"day": "2026-10-19", "tokens": 50},
    ]
    grid = usage_stats.build_heatmap(daily, "2026-10-14", "2026-10-19")
    cells = {c["day"]: c for w in grid["weeks"] for c in w}
    # 峰值日必须落在最高档，半数落在中档，无数据日为 0 档。
    assert cells["2026-10-16"]["level"] == usage_stats.HEAT_LEVELS
    assert cells["2026-10-19"]["level"] == 2
    assert cells["2026-10-15"]["level"] == 0


def test_build_heatmap_single_day():
    grid = usage_stats.build_heatmap([], "2026-10-16", "2026-10-16")
    assert grid["peak"] == 0
    assert grid["active_days"] == 0
    # 10-16 是周五，补齐到 10-12（周一）.. 10-18（周日）恰好一周
    assert len(grid["weeks"]) == 1
    assert [c["day"] for c in grid["weeks"][0]] == [
        "2026-10-12",
        "2026-10-13",
        "2026-10-14",
        "2026-10-15",
        "2026-10-16",
        "2026-10-17",
        "2026-10-18",
    ]
    assert grid["weeks"][0][0]["level"] == 0


def test_build_heatmap_bad_range_returns_empty():
    assert usage_stats.build_heatmap([], "bad", "2026-10-19") == {
        "weeks": [],
        "peak": 0,
        "active_days": 0,
    }
    assert usage_stats.build_heatmap([], "2026-10-19", "2026-10-01") == {
        "weeks": [],
        "peak": 0,
        "active_days": 0,
    }


def test_heat_level_clamps_to_level_count():
    """相对峰值分档，极端值不得超过档数上限。"""
    assert usage_stats._heat_level(0, 100) == 0
    assert usage_stats._heat_level(100, 0) == 0
    assert usage_stats._heat_level(100, 100) == usage_stats.HEAT_LEVELS
    assert usage_stats._heat_level(200, 100) == usage_stats.HEAT_LEVELS
    assert usage_stats._heat_level(1, 10_000) == 1
    assert usage_stats.parse_range_params("") == {}