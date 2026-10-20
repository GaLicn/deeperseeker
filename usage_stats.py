"""Token 使用统计：采样落库、区间聚合、峰谷计价。

口径与参考插件（dsh-usage-statistics-panel）一致：

* 单价按每百万 token 计，单位是人民币元。
* 输入侧拆成 cache hit / cache miss 两档，输出单列。
* 高峰时段单价是低峰的 2 倍。高峰窗口按北京时间（UTC+8）判定：工作日
  09:00-12:00 与 14:00-18:00，周末与节假日全天低峰。窗口边界都落在整点，
  所以按小时采样就能完整还原每次调用落在哪一档——日粒度做不到，一天的
  总量无法反推回峰谷两个窗口。

因此 usage_hours 表以 (小时, 模型, token, API Key) 为主键累加，聚合时逐小时
计价再求和，绝不拿区间总量套一次价。

本模块单向依赖 functions.get_db（functions 不反向导入），避免循环导入。
"""

import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

from functions import get_db

# --- 计价 -------------------------------------------------------------------

# 每百万 token 的单价。off_peak 为基准价，peak 恰好是其 2 倍。
PRICING = {
    "default": {
        "off_peak": {"cache_hit": 0.02, "cache_miss": 1.0, "output": 4.0},
        "peak": {"cache_hit": 0.04, "cache_miss": 2.0, "output": 8.0},
    },
    "deepseek-v4.1-flash": {
        "off_peak": {"cache_hit": 0.02, "cache_miss": 1.0, "output": 4.0},
        "peak": {"cache_hit": 0.04, "cache_miss": 2.0, "output": 8.0},
    },
}
PER_MILLION = 1_000_000

BEIJING_TZ = timezone(timedelta(hours=8))

# 高峰窗口（北京时间，闭开区间 [start, end)）：上午与下午各一段。
PEAK_WINDOWS = ((9, 12), (14, 18))


def price_of(model):
    """取某模型的单价表；未登记的模型退回 default 档。

    注意：``model`` 列存的是客户端请求的原始模型名（可能带 ``provider/``
    前缀，见 provider_of），这类名字查不到表会退回 default 档。这不是缺陷
    ——default 与 deepseek-v4.1-flash 的单价逐字段相同，且本项目所有请求
    最终都由同一个上游模型服务。
    """
    return PRICING.get(model) or PRICING["default"]


# 客户端模型名的拆分规则与参考插件一致：``provider/model`` 取斜杠前的部分，
# 裸名（无斜杠）归到默认服务商。参考插件用 "default" 是因为它要展示任意网关
# 的接入方式；本项目只有一个上游，叫 deepseek 更直白。
DEFAULT_PROVIDER = "deepseek"


def provider_of(model_ref):
    """从客户端模型名里取服务商；``a/b`` 取 ``a``，裸名取默认服务商。"""
    if not model_ref:
        return DEFAULT_PROVIDER
    slash = model_ref.find("/")
    if slash > 0:
        return model_ref[:slash]
    return DEFAULT_PROVIDER


def model_name_of(model_ref):
    """取模型名的显示部分：``a/b`` 取 ``b``，裸名原样返回。"""
    if not model_ref:
        return model_ref
    slash = model_ref.find("/")
    if slash > 0:
        return model_ref[slash + 1:]
    return model_ref


def beijing_now():
    """当前北京时间。"""
    return datetime.now(BEIJING_TZ)


def hour_key(dt):
    """北京时间的小时键，形如 ``2026-10-16T14``。"""
    return dt.strftime("%Y-%m-%dT%H")


def day_key(dt):
    """北京时间的日键，形如 ``2026-10-16``。"""
    return dt.strftime("%Y-%m-%d")


def day_of_hour_key(key):
    """小时键所属的日键。"""
    return key[:10]


def parse_hour_key(key):
    """把小时键解析成带时区的 datetime；格式不合法返回 None。

    解析失败一律按低峰处理，坏数据不会把成本算高。
    """
    try:
        return datetime.strptime(key, "%Y-%m-%dT%H").replace(tzinfo=BEIJING_TZ)
    except (TypeError, ValueError):
        return None


def parse_day_key(key):
    """把日键解析成当天 00:00 的 datetime；格式不合法返回 None。"""
    try:
        return datetime.strptime(key, "%Y-%m-%d").replace(tzinfo=BEIJING_TZ)
    except (TypeError, ValueError):
        return None


def is_peak_hour(dt):
    """该时刻是否落在高峰窗口内（仅工作日）。"""
    if dt.weekday() >= 5:  # 周六、周日全天低峰
        return False
    for start, end in PEAK_WINDOWS:
        if start <= dt.hour < end:
            return True
    return False


def is_peak_hour_key(key):
    """小时键是否属于高峰窗口；无法解析时按低峰处理。"""
    dt = parse_hour_key(key)
    return False if dt is None else is_peak_hour(dt)


def cost_of_hour(model, buckets, key):
    """按小时键给一组 token 计价。

    @param model 模型名，用于选单价表。
    @param buckets 含 cache_hit / cache_miss / output 三个键的用量。
    @param key 北京时间小时键，决定落在高峰还是低峰。
    @returns 含 total 与三个分项的 dict，另带 tier 字段。
    """
    tier = "peak" if is_peak_hour_key(key) else "off_peak"
    price = price_of(model)[tier]
    cache_miss = buckets.get("cache_miss", 0) / PER_MILLION * price["cache_miss"]
    cache_hit = buckets.get("cache_hit", 0) / PER_MILLION * price["cache_hit"]
    output = buckets.get("output", 0) / PER_MILLION * price["output"]
    return {
        "total": cache_miss + cache_hit + output,
        "cache_miss": cache_miss,
        "cache_hit": cache_hit,
        "output": output,
        "tier": tier,
    }


# --- 采样落库 ---------------------------------------------------------------


def record_usage(
    model,
    token_id=None,
    api_key="",
    input_tokens=0,
    output_tokens=0,
    cache_read_tokens=0,
    cache_write_tokens=0,
    requests=1,
    when=None,
):
    """把一次完成的调用累加进它所属的小时桶。

    同 (小时, 模型, token, API Key) 的行做 UPSERT 累加，因此同一小时内的
    多次调用只占一行，查询端不需要再做 group by 之外的合并。

    token_id 缺失记 -1、api_key 缺失记空串：SQLite 把 NULL 视作互不相等，
    主键里一旦出现 NULL，同一个桶就能被重复插入。
    """
    dt = when or beijing_now()
    key = hour_key(dt)
    row = (
        key,
        model or "unknown",
        -1 if token_id is None else int(token_id),
        api_key or "",
        max(0, int(input_tokens or 0)),
        max(0, int(output_tokens or 0)),
        max(0, int(cache_read_tokens or 0)),
        max(0, int(cache_write_tokens or 0)),
        max(0, int(requests or 0)),
        time.time(),
    )
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT INTO usage_hours (
                hour, model, token_id, api_key,
                input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens,
                requests, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hour, model, token_id, api_key) DO UPDATE SET
                input_tokens = input_tokens + excluded.input_tokens,
                output_tokens = output_tokens + excluded.output_tokens,
                cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                requests = requests + excluded.requests,
                last_seen = excluded.last_seen
            """,
            row,
        )
        conn.commit()
    finally:
        conn.close()


def query_hours(from_key, to_key):
    """取 [from_key, to_key] 区间内的小时行，按小时升序。

    两个键都是 ``YYYY-MM-DDTHH``，字符串比较即时间比较，无需解析。
    """
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT hour, model, token_id, api_key,
                   input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens, requests
            FROM usage_hours
            WHERE hour >= ? AND hour <= ?
            ORDER BY hour
            """,
            (from_key, to_key),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def clear_usage():
    """清空采样表（面板的「重建统计」用）。"""
    conn = get_db()
    try:
        conn.execute("DELETE FROM usage_hours")
        conn.commit()
    finally:
        conn.close()


def earliest_day():
    """采样表里最早的一天（日键）；表空时返回 None。

    「全部」预设的起点只能从库里读，而 resolve_range 是纯函数——所以由调用
    方先取这个值再传进去。
    """
    conn = get_db()
    try:
        row = conn.execute("SELECT MIN(hour) AS h FROM usage_hours").fetchone()
    finally:
        conn.close()
    if row is None or not row["h"]:
        return None
    return row["h"][:10]


# --- 区间聚合 ---------------------------------------------------------------


def _empty_totals():
    return {
        "tokens": 0,
        "requests": 0,
        "input": 0,
        "output": 0,
        "cache_hit": 0,
        "cache_miss": 0,
        "cache_write": 0,
        "cost": 0.0,
        "cost_peak": 0.0,
        "cost_off_peak": 0.0,
        "cost_cache_hit": 0.0,
        "cost_cache_miss": 0.0,
        "cost_output": 0.0,
    }


def _add_sample(acc, model, cache_hit, cache_miss, output, cache_write, requests, hour):
    """把一条采样并进一个累计器，并按该小时的档位计价。"""
    charge = cost_of_hour(
        model,
        {"cache_hit": cache_hit, "cache_miss": cache_miss, "output": output},
        hour,
    )
    acc["input"] += cache_hit + cache_miss
    acc["output"] += output
    acc["cache_hit"] += cache_hit
    acc["cache_miss"] += cache_miss
    acc["cache_write"] += cache_write
    # 头部总量与供应商看板口径一致：未缓存输入 + 输出 + 缓存读 + 缓存写。
    acc["tokens"] += cache_hit + cache_miss + output + cache_write
    acc["requests"] += requests
    acc["cost"] += charge["total"]
    acc["cost_cache_hit"] += charge["cache_hit"]
    acc["cost_cache_miss"] += charge["cache_miss"]
    acc["cost_output"] += charge["output"]
    if charge["tier"] == "peak":
        acc["cost_peak"] += charge["total"]
    else:
        acc["cost_off_peak"] += charge["total"]


def _days_between(from_day, to_day):
    """[from_day, to_day] 内的全部日键；边界非法或倒置时返回空表。"""
    start = parse_day_key(from_day)
    end = parse_day_key(to_day)
    if start is None or end is None or end < start:
        return []
    out = []
    cur = start
    while cur <= end:
        out.append(day_key(cur))
        cur += timedelta(days=1)
    return out


def _hours_between(from_key, to_key):
    """[from_key, to_key] 内的全部小时键。"""
    start = parse_hour_key(from_key)
    end = parse_hour_key(to_key)
    if start is None or end is None or end < start:
        return []
    out = []
    cur = start
    while cur <= end:
        out.append(hour_key(cur))
        cur += timedelta(hours=1)
    return out


def _rank(rows, key_fn, label_fn):
    """按 token 量降序整理一份排行，附带成本。"""
    buckets = {}
    for r in rows:
        key = key_fn(r)
        entry = buckets.get(key)
        if entry is None:
            entry = _empty_totals()
            entry["key"] = key
            entry["label"] = label_fn(r, key)
            buckets[key] = entry
        cache_hit = r["cache_read_tokens"]
        cache_miss = max(0, r["input_tokens"] - cache_hit)
        _add_sample(
            entry,
            r["model"],
            cache_hit,
            cache_miss,
            r["output_tokens"],
            r["cache_write_tokens"],
            r["requests"],
            r["hour"],
        )
    ordered = sorted(buckets.values(), key=lambda e: e["tokens"], reverse=True)
    return ordered


def hit_rate_series(series):
    """按一批累计桶算逐点缓存命中率，分母为 0 的点记 0。

    命中率曲线与主图共用同一批点位，公式集中在这里，免得调用方各写一份。
    """
    return [
        (s["cache_hit"] / (s["cache_hit"] + s["cache_miss"]))
        if (s["cache_hit"] + s["cache_miss"])
        else 0.0
        for s in series
    ]


def aggregate(rows, from_day, to_day, labels=None):
    """把小时行折叠成面板需要的一份区间统计。

    @param rows query_hours 的原始行。
    @param from_day 区间起始日键（含）。
    @param to_day 区间结束日键（含）。
    @param labels 可选，token_id -> 别名、api_key -> 备注 的映射，用于显示名。
    @returns 含区间总量、逐日、逐小时与各维度排行的 dict。
    """
    labels = labels or {}
    totals = _empty_totals()

    day_acc = {}
    hour_acc = {}
    model_acc = {}

    for r in rows:
        cache_hit = r["cache_read_tokens"]
        cache_miss = max(0, r["input_tokens"] - cache_hit)
        output = r["output_tokens"]
        cache_write = r["cache_write_tokens"]
        requests = r["requests"]
        model = r["model"]
        hour = r["hour"]

        _add_sample(
            totals, model, cache_hit, cache_miss, output, cache_write, requests, hour
        )

        dk = day_of_hour_key(hour)
        d = day_acc.get(dk)
        if d is None:
            d = _empty_totals()
            d["by_model"] = {}
            day_acc[dk] = d
        _add_sample(d, model, cache_hit, cache_miss, output, cache_write, requests, hour)
        d["by_model"][model] = d["by_model"].get(model, 0) + (
            cache_hit + cache_miss + output + cache_write
        )

        h = hour_acc.get(hour)
        if h is None:
            h = _empty_totals()
            hour_acc[hour] = h
        _add_sample(h, model, cache_hit, cache_miss, output, cache_write, requests, hour)

        m = model_acc.get(model)
        if m is None:
            m = _empty_totals()
            model_acc[model] = m
        _add_sample(m, model, cache_hit, cache_miss, output, cache_write, requests, hour)

    daily = []
    for dk in _days_between(from_day, to_day):
        src = day_acc.get(dk)
        acc = src if src is not None else _empty_totals()
        acc["day"] = dk
        acc["by_model"] = src["by_model"] if src is not None else {}
        daily.append(acc)

    hourly = []
    for hk in _hours_between(from_day + "T00", to_day + "T23"):
        acc = hour_acc.get(hk) or _empty_totals()
        acc["hour"] = hk
        acc["peak"] = is_peak_hour_key(hk)
        hourly.append(acc)

    models = _rank(rows, lambda r: r["model"], lambda r, k: k)
    pool = _rank(
        rows,
        lambda r: r["token_id"],
        lambda r, k: labels.get("tokens", {}).get(k, f"Token #{k}" if k >= 0 else "未知"),
    )
    keys = _rank(
        rows,
        lambda r: r["api_key"],
        lambda r, k: labels.get("keys", {}).get(k, _short_key(k)),
    )
    # 服务商由客户端模型名的前缀推导，不额外落库：``anthropic/claude-*`` 归
    # anthropic，``v4.1flash`` 这类裸名归 deepseek。每个服务商附上它服务的
    # 模型清单，供表面板展开说明。
    providers = _rank(rows, lambda r: provider_of(r["model"]), lambda r, k: k)
    provider_models = {}
    for r in rows:
        provider_models.setdefault(provider_of(r["model"]), set()).add(
            model_name_of(r["model"])
        )
    for p in providers:
        p["models"] = sorted(provider_models.get(p["key"], ()))

    # 「活跃天」按当天是否有调用判定，而不是 tokens > 0：采样桶的 token 计数
    # 在极端情况下可能为 0（例如只落到请求数），那仍是真实使用过的一天。
    active_days = sum(1 for d in daily if d["requests"] > 0)
    denom = totals["cache_hit"] + totals["cache_miss"]
    totals["cache_hit_rate"] = (totals["cache_hit"] / denom) if denom else 0.0
    totals["active_days"] = active_days
    totals["from"] = from_day
    totals["to"] = to_day
    totals["daily"] = daily
    totals["hourly"] = hourly
    totals["models"] = models
    totals["pool"] = pool
    totals["keys"] = keys
    totals["providers"] = providers
    totals["top_model"] = models[0]["label"] if models else ""
    totals["second_model"] = models[1]["label"] if len(models) > 1 else ""
    totals["top_provider"] = providers[0]["label"] if providers else ""
    # 命中率序列不在这里给：本函数同时算好了 daily 与 hourly 两套点，选哪套
    # 要看图表最终采用的粒度（含降采样），那是 _chart_from_stats 的决定。
    return totals


# 热力图色阶档数：0 档为空数据日，1..HEAT_LEVELS 按当日 token 量相对区间
# 峰值分档。档数写死是因为 CSS 只准备了对应数量的类名（.heatLevel1..5）。
HEAT_LEVELS = 5


def _heat_level(tokens, peak):
    """按区间峰值把单日 token 量折成 0..HEAT_LEVELS 档。"""
    if tokens <= 0 or peak <= 0:
        return 0
    # 峰值日自身必须落在最高档：int(peak/peak*5) 恰好是 5，无需特判。
    level = int(tokens / peak * HEAT_LEVELS)
    return min(HEAT_LEVELS, max(1, level))


def build_heatmap(daily, from_day, to_day):
    """把逐日累计折成贡献图需要的周网格。

    网格从 ``from_day`` 所在周的周一起、到 ``to_day`` 所在周的周日止，两端
    补齐的区间外格子以 level 0 占位——每列固定 7 天，前端才能用纯 CSS grid
    按列渲染。与 aggregate 一样是纯函数，不读库。

    @param daily aggregate 返回的逐日列表（含 day / tokens）。
    @returns {"weeks": [[cell × 7], ...], "peak": 单日最大 token,
              "active_days": token 大于 0 的天数}，cell 为
              {"day", "tokens", "level"}。
    """
    start = parse_day_key(from_day)
    end = parse_day_key(to_day)
    if start is None or end is None or end < start:
        return {"weeks": [], "peak": 0, "active_days": 0}

    # Python 的 weekday() 周一为 0：回退到本周一、前进到本周日。
    grid_start = start - timedelta(days=start.weekday())
    grid_end = end + timedelta(days=6 - end.weekday())

    by_day = {d["day"]: d for d in daily}
    cells = []
    peak = 0
    active_days = 0
    cur = grid_start
    while cur <= grid_end:
        dk = day_key(cur)
        row = by_day.get(dk)
        tokens = row["tokens"] if row else 0
        if tokens > 0:
            active_days += 1
            peak = max(peak, tokens)
        cells.append({"day": dk, "tokens": tokens, "level": 0})
        cur += timedelta(days=1)

    # 峰值要扫完整张网格才能定档，故分两趟：先收集再回填 level。
    for cell in cells:
        cell["level"] = _heat_level(cell["tokens"], peak)

    weeks = [cells[i:i + 7] for i in range(0, len(cells), 7)]
    return {"weeks": weeks, "peak": peak, "active_days": active_days}


def _short_key(key):
    """API Key 的显示名：只留头尾，避免完整密钥出现在页面上。"""
    if not key:
        return "未标注"
    if len(key) <= 12:
        return key
    return f"{key[:6]}…{key[-4:]}"


# --- 区间解析 ---------------------------------------------------------------

# 面板的预设区间。``all`` 覆盖采样表里的全部历史，需要调用方把最早的一天
# 传进来——resolve_range 是纯函数，读不到库。
RANGE_PRESETS = {"today", "yesterday", "24h", "7", "14", "30", "90", "all"}
# 「最近 N 天」预设对应的天数。
RANGE_DAYS = {"7": 7, "14": 14, "30": 30, "90": 90}

# 预设键到中文标签，顺序即界面上的排列顺序。模板与后端共用这一份，
# 免得两边各写一张表、加档位时漏改一处。
RANGE_LABELS = (
    ("today", "当天"),
    ("yesterday", "昨天"),
    ("24h", "24 小时内"),
    ("7", "最近 7 天"),
    ("14", "最近 14 天"),
    ("30", "最近 30 天"),
    ("90", "最近 90 天"),
    ("all", "全部"),
)


def resolve_range(params, earliest=None):
    """把查询参数解析成 (from_day, to_day, from_hour, to_hour)。

    ``24h`` 是滚动窗口，无法只用日键表达，因此额外返回精确的小时边界；
    其余预设与自定义区间都对齐到整天，小时边界为 None。

    @param earliest ``all`` 预设下采样表里最早的一天（``YYYY-MM-DD``）。缺省
        时退回最近 365 天，免得空库上算出一个没有边界的区间。
    """
    preset = (params.get("range") or ["30"])[0]
    now = beijing_now()
    if preset == "custom":
        frm = (params.get("from") or [""])[0]
        to = (params.get("to") or [""])[0]
        if parse_day_key(frm) is None or parse_day_key(to) is None:
            preset = "30"
        else:
            return frm, to, None, None
    if preset == "today":
        d = day_key(now)
        return d, d, None, None
    if preset == "yesterday":
        d = day_key(now - timedelta(days=1))
        return d, d, None, None
    if preset == "24h":
        end = now
        start = now - timedelta(hours=23)
        return day_key(start), day_key(end), hour_key(start), hour_key(end)
    if preset == "all":
        start = parse_day_key(earliest) if earliest else None
        if start is None:
            start = now - timedelta(days=364)
        return day_key(start), day_key(now), None, None
    days = RANGE_DAYS.get(preset, 30)
    end = now
    start = now - timedelta(days=days - 1)
    return day_key(start), day_key(end), None, None


def parse_range_params(query_string):
    """把 URL 的 query string 解析成 resolve_range 需要的 dict。"""
    return parse_qs(query_string or "")