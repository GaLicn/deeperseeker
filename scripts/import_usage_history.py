"""把外部用量记录导入 usage_hours 表。

两个来源：

* dsh-usage-statistics-panel 的持久化文件 ``~/.dsh/storages/usage_history.json``
  （``tables.hours`` 里的逐小时桶）；
* 面板导出的 CSV（列：时间, 模型, 线路, 请求次数, 输入, 缓存读, 缓存写, 输出 …）。

口径换算
--------
两份来源的「输入」都是**未缓存输入**（cache miss），而 ``usage_hours.input_tokens``
按本仓库 ``_record_usage`` 的写法存的是**含缓存读的总输入**：

    input_tokens = 输入 + 缓存读
    cache_read_tokens = 缓存读
    cache_write_tokens = 缓存写
    output_tokens = 输出

不做这一步相加，``aggregate`` 里的 ``cache_miss = input_tokens - cache_read_tokens``
会把未缓存输入和缓存读都算进 cache miss，成本直接翻倍。

时间：两份来源的小时键都是北京时间，与 usage_stats 的峰谷窗口口径一致，直接落库。

幂等：按 (hour, model, token_id, api_key) 主键 UPSERT 累加，重复执行会叠加而非
去重；因此脚本默认先 dry-run 打印将写入的桶，确认无误再 ``--commit``。

Run:
  python scripts/import_usage_history.py --json <path> --csv <path>          # dry-run
  python scripts/import_usage_history.py --json <path> --csv <path> --commit  # 写入
  python scripts/import_usage_history.py --db <sqlite path> …                 # 指定库
"""

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import defaultdict

# 导入数据的来源标记，写进 api_key 列，便于与真实网关流量区分 / 事后回滚。
# 用 ``source:`` 前缀，usage_stats._short_key 会把它截成 ``source…ookup`` 之类，
# 不会和真实密钥混淆。
SOURCE_DSH = "source:dsh-panel"
SOURCE_CSV = "source:pixelsky-csv"

DEFAULT_DB = "/app/deeperseeker.db"


def _int(v):
    v = (v or "").strip() if isinstance(v, str) else v
    if v in (None, ""):
        return 0
    return int(v)


def load_dsh_hours(path):
    """读 dsh json，产出 (hour, model, token_id, api_key, in, out, cr, cw, req)。"""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    out = []
    for h in doc.get("tables", {}).get("hours", {}).values():
        provider = (h.get("provider") or "default").strip()
        model = (h.get("model") or "(unknown)").strip()
        # dsh 里 provider 与 model 是分开两列，而本仓库的 model 列存的是
        # ``provider/model`` 原始名（provider_of 依赖斜杠拆分）。裸名要补前缀，
        # 已经是 ``a/b`` 形态的不重复拼。
        ref = model if "/" in model else f"{provider}/{model}"
        cache_read = _int(h.get("cacheReadTokens"))
        miss = _int(h.get("inputTokens"))
        out.append(
            (
                h["hour"],
                ref,
                -1,
                SOURCE_DSH,
                miss + cache_read,  # 总输入
                _int(h.get("outputTokens")),
                cache_read,
                _int(h.get("cacheWriteTokens")),
                _int(h.get("requests")),
            )
        )
    return out


def load_csv_hours(path):
    """读面板导出的 CSV，产出同样形状的行。

    CSV 每行是一次会话的一段时间切片，粒度为秒；本表按小时聚合，故按
    小时键 + 模型合并后再落库。
    """
    buckets = defaultdict(lambda: [0, 0, 0, 0, 0])
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ts = (row.get("时间") or "").strip()
            if len(ts) < 13:
                continue
            hour = ts[:13].replace(" ", "T")  # "2026-10-16 16" -> "2026-10-16T16"
            model = (row.get("模型") or "(unknown)").strip()
            cache_read = _int(row.get("缓存读"))
            miss = _int(row.get("输入"))
            b = buckets[(hour, model)]
            b[0] += miss + cache_read
            b[1] += _int(row.get("输出"))
            b[2] += cache_read
            b[3] += _int(row.get("缓存写"))
            b[4] += _int(row.get("请求次数"))
    return [
        (hour, model, -1, SOURCE_CSV, b[0], b[1], b[2], b[3], b[4])
        for (hour, model), b in buckets.items()
    ]


def upsert(conn, rows):
    conn.executemany(
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
        [(*r, time.time()) for r in rows],
    )


def summarize(rows, label):
    tok = sum(r[4] + r[5] + r[6] + r[7] for r in rows)
    req = sum(r[8] for r in rows)
    print(f"  {label}: {len(rows)} 个桶, token {tok:,}, 请求 {req:,}")
    by_model = defaultdict(lambda: [0, 0])
    for r in rows:
        by_model[r[1]][0] += r[4] + r[5] + r[6] + r[7]
        by_model[r[1]][1] += r[8]
    for m in sorted(by_model, key=lambda k: -by_model[k][0]):
        print(f"      {m:<42} token {by_model[m][0]:>15,}  req {by_model[m][1]:>7,}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", dest="json_path", help="dsh usage_history.json")
    ap.add_argument("--csv", dest="csv_path", help="面板导出的 CSV")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"目标 SQLite（默认 {DEFAULT_DB}）")
    ap.add_argument("--commit", action="store_true", help="真正写入；缺省仅预览")
    args = ap.parse_args(argv)

    if not args.json_path and not args.csv_path:
        ap.error("至少给一个 --json 或 --csv")

    all_rows = []
    print("=== 将导入的数据 ===")
    if args.json_path:
        rows = load_dsh_hours(args.json_path)
        summarize(rows, "dsh json")
        all_rows += rows
    if args.csv_path:
        rows = load_csv_hours(args.csv_path)
        summarize(rows, "csv")
        all_rows += rows

    print(f"\n目标库: {args.db}")
    conn = sqlite3.connect(args.db)
    try:
        before = conn.execute("SELECT COUNT(*) FROM usage_hours").fetchone()[0]
        if not args.commit:
            print(f"当前 usage_hours 行数: {before}")
            print("dry-run：未写入。加 --commit 生效。")
            return 0
        upsert(conn, all_rows)
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM usage_hours").fetchone()[0]
        print(f"usage_hours 行数: {before} -> {after}（写入 {len(all_rows)} 个桶）")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
