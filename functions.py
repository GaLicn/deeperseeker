import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import mimetypes
import os
import random
import re
import sqlite3
import string
import time
from collections import OrderedDict
from datetime import datetime, timezone

import aiohttp
import deepseek_tokenizer
import wasmtime

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

logger = logging.getLogger("deeperseeker.functions")

# B12：所有运行时相对路径锚定到包目录。此前 app.py 在导入时执行全局
# os.chdir(BASE_DIR)，把进程级状态改成了「依赖导入顺序」——内嵌、多 worker、
# 测试与打包都会因此行为漂移。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

wasm_path = os.path.join(BASE_DIR, "wasm", "deepseek_pow_solver.wasm")
_session = None
_db = os.getenv("DB_PATH") or os.path.join(BASE_DIR, "deeperseeker.db")


def data_dir():
    """返回存放 SQLite 数据库的目录。

    与数据库同级的运行时文件（如自动生成的 API key）也放在这里。
    显式设置 DB_PATH 时以它为准；缺省锚定包目录，不依赖进程 CWD。
    """
    return os.path.dirname(os.path.abspath(_db))


class DeepSeekRateLimitError(Exception):
    """Upstream rate limiting reported inside an otherwise successful SSE response."""

    def __init__(self, message="DeepSeek rate limit reached"):
        super().__init__(f"HTTP 429: {message}")


def cookie_file_path():
    """Resolve where the DeepSeek cookie file lives.

    Order: DEEPSEEKER_COOKIE_PATH env > the target of a legacy Docker symlink
    > next to the real DB file (which honors DB_PATH) > the package directory.
    Writing goes to the RESOLVED path so os.replace() can never destroy a
    symlink that bridges the file into the persistent data volume.
    """
    p = os.getenv("DEEPSEEKER_COOKIE_PATH")
    if p:
        return p
    p = os.path.join(BASE_DIR, "aws_cookies_deepseek.json")
    try:
        if os.path.islink(p):
            target = os.path.realpath(p)
            if target:
                return target
    except Exception:
        pass
    d = os.path.dirname(os.path.abspath(_db))
    if d and os.path.abspath(d) != os.path.abspath(os.getcwd()):
        return os.path.join(d, "aws_cookies_deepseek.json")
    return p


try:
    _TZ_OFFSET = str(int(datetime.now().astimezone().utcoffset().total_seconds()))
except Exception:
    _TZ_OFFSET = "19800"


def get_db():
    conn = sqlite3.connect(_db, timeout=30)
    conn.row_factory = sqlite3.Row
    with contextlib.suppress(Exception):
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias TEXT,
            token TEXT,
            status TEXT DEFAULT 'ACTIVE',
            last_checked_at TEXT,
            last_check_error TEXT,
            account_id INTEGER,
            limited_until REAL
        );
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            alias TEXT,
            token_id INTEGER,
            status TEXT DEFAULT 'ACTIVE',
            last_login_at TEXT,
            last_error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            signature TEXT PRIMARY KEY,
            token_id INTEGER,
            deepseek_session_id TEXT,
            parent_message_id INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS session_map (
            old_session TEXT PRIMARY KEY,
            new_session TEXT,
            token_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            token_id INTEGER,
            created_at REAL
        );
        -- Token usage samples. One row per (hour, model, token, api key):
        -- hours, not days, because DeepSeek bills input and output at double
        -- the rate inside its peak windows and those boundaries fall on whole
        -- hours (see usage_stats.py) — a daily row cannot reconstruct the
        -- tier of each call. token_id -1 and '' stand for "unknown" so the
        -- primary key never sees a NULL (SQLite treats NULLs as distinct,
        -- which would let the same bucket be inserted twice).
        CREATE TABLE IF NOT EXISTS usage_hours (
            hour TEXT NOT NULL,
            model TEXT NOT NULL,
            token_id INTEGER NOT NULL DEFAULT -1,
            api_key TEXT NOT NULL DEFAULT '',
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            requests INTEGER NOT NULL DEFAULT 0,
            last_seen REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (hour, model, token_id, api_key)
        );
        CREATE INDEX IF NOT EXISTS idx_usage_hours_hour ON usage_hours(hour);
    """)
    # Keep existing Docker volumes compatible with the token health metadata.
    token_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(tokens)").fetchall()
    }
    if "last_checked_at" not in token_columns:
        conn.execute("ALTER TABLE tokens ADD COLUMN last_checked_at TEXT")
    if "last_check_error" not in token_columns:
        conn.execute("ALTER TABLE tokens ADD COLUMN last_check_error TEXT")
    if "account_id" not in token_columns:
        conn.execute("ALTER TABLE tokens ADD COLUMN account_id INTEGER")
    if "limited_until" not in token_columns:
        conn.execute("ALTER TABLE tokens ADD COLUMN limited_until REAL")
    # 调度器并列时的「最久未用优先」需要记录 token 的上次选中时刻；
    # 注意与 sessions.last_used（B13 的会话清理）分属两张表。
    if "last_used" not in token_columns:
        conn.execute("ALTER TABLE tokens ADD COLUMN last_used REAL")
    # B13: session pruning keys on real recency of USE, so older volumes need
    # the column; the ALTER is idempotent behind the pragma check.
    session_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    if "last_used" not in session_columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN last_used REAL")
    conn.commit()
    conn.close()
    try:
        prune_sessions()
    except Exception:
        logger.exception("Startup session pruning failed (non-fatal)")


_session_lock = asyncio.Lock()


async def get_session():
    global _session
    if _session is None or _session.closed:
        async with _session_lock:
            if _session is None or _session.closed:
                logger.info("Opening shared aiohttp ClientSession")
                _session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=None, connect=15, sock_read=600)
                )
    return _session


async def close_session():
    """关闭共享的 aiohttp ClientSession。

    由应用 lifespan 关闭阶段调用。缺少该调用时，每次 uvicorn 重载或退出都会
    泄漏会话的 connector 套接字，并产生 "Unclosed client session" 警告。
    """
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


# ==============================================================================
# Stage 0.4 — Dual-endpoint failover
#
# Every upstream POST goes through post_with_failover(): if the primary host
# fails at the connection level (ClientError / timeout) or answers HTTP 5xx,
# the identical request is replayed against the fallback endpoint. 4xx
# responses are returned as-is — a bad token or permission error will not
# improve on a different endpoint, so we fail fast instead of retrying.
#
# Endpoints (env, both optional to override):
#   DEEPSEEKER_UPSTREAM_BASE     primary   (default https://chat.deepseek.com)
#   DEEPSEEKER_UPSTREAM_FALLBACK fallback  (default: none; single-endpoint mode)
# ==============================================================================


def _build_upstream_bases():
    primary = (
        (os.getenv("DEEPSEEKER_UPSTREAM_BASE") or "https://chat.deepseek.com")
        .strip()
        .rstrip("/")
    )
    fallback = (os.getenv("DEEPSEEKER_UPSTREAM_FALLBACK") or "").strip().rstrip("/")
    return [primary] + ([fallback] if fallback else [])


UPSTREAM_BASES = _build_upstream_bases()


async def post_with_failover(path, *, headers, session=None, **kwargs):
    """POST to an upstream endpoint with automatic failover across
    UPSTREAM_BASES. This is a plain coroutine that RETURNS an owned response —
    it is not an async context manager:

        resp = await post_with_failover(...)
        async with resp:
            ...

    The caller owns and closes the returned response (async with). Raises the
    last connection error if every endpoint is unreachable, or returns the
    final 4xx/5xx response when the last endpoint answers with one (callers
    keep their existing error paths). `session` is injectable for tests;
    defaults to the shared ClientSession.
    """
    if session is None:
        session = await get_session()
    last_exc = None
    for attempt, base in enumerate(UPSTREAM_BASES):
        is_last = attempt == len(UPSTREAM_BASES) - 1
        try:
            resp = await session.post(base + path, headers=headers, **kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Upstream POST %s failed on %s: %s", path, base, e)
            last_exc = e
            if is_last:
                raise
            continue
        if resp.status >= 500 and not is_last:
            try:
                body = await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                body = ""
            # PR #26 review fix (Medium): hand the failed connection back to
            # the pool. Without this, repeated failovers leak responses and
            # eventually exhaust the connector.
            resp.release()
            logger.warning(
                "Upstream POST %s -> HTTP %d on %s; failing over to fallback endpoint",
                path,
                resp.status,
                base,
            )
            last_exc = Exception(f"HTTP {resp.status}: {body[:200]}")
            continue
        if attempt > 0:
            logger.info(
                "Upstream POST %s succeeded on fallback endpoint %s", path, base
            )
        return resp
    raise (
        last_exc
        if last_exc is not None
        else RuntimeError("post_with_failover: no endpoints configured")
    )


# B11 (Stage 1 audit): the Android client identity used to be hardcoded to one
# version for ALL tokens — when DeepSeek's app moves, a single stale string
# degrades every account simultaneously, and identical fingerprints correlate
# bans. The version is env-configurable now; per-token identity profiles
# (rotating UA/version/locale) remain a P1.
DEEPSEEKER_CLIENT_VERSION = os.getenv("DEEPSEEKER_CLIENT_VERSION", "2.4.5")


def get_headers(auth_token, pow=None):
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/",
        "user-agent": "Dalvik/2.1.0 (Linux; U; Android 14; Pixel 7)",
        "x-client-platform": "android",
        "x-client-version": DEEPSEEKER_CLIENT_VERSION,
        "x-client-locale": "en_US",
        "x-client-bundle-id": "com.deepseek.chat",
        "x-client-timezone-offset": _TZ_OFFSET,
    }
    if auth_token:
        headers["authorization"] = f"Bearer {auth_token}"
    if pow:
        headers["x-ds-pow-response"] = pow
    return headers


def _extract_login_token(payload):
    """Find a bearer token across the login response shapes used by DeepSeek."""
    if isinstance(payload, str):
        candidate = payload.strip()
        if candidate.startswith("{") or candidate.startswith("["):
            try:
                return _extract_login_token(json.loads(candidate))
            except json.JSONDecodeError:
                return None
        return None
    if isinstance(payload, dict):
        # Current web login shape: data.biz_data.user.token.
        data_section = payload.get("data")
        biz_data = (
            data_section.get("biz_data") if isinstance(data_section, dict) else None
        )
        user = biz_data.get("user") if isinstance(biz_data, dict) else None
        if (
            isinstance(user, dict)
            and isinstance(user.get("token"), str)
            and user["token"].strip()
        ):
            return user["token"].strip()
        for key in ("token", "access_token", "auth_token"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            token = _extract_login_token(value)
            if token:
                return token
    elif isinstance(payload, list):
        for value in payload:
            token = _extract_login_token(value)
            if token:
                return token
    return None


def _redact_login_response(value):
    """Redact credentials before writing a DeepSeek login response to logs."""
    sensitive_markers = ("token", "password", "authorization", "cookie", "secret")
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key).lower()
            redacted[key] = (
                "[REDACTED]"
                if any(marker in key_text for marker in sensitive_markers)
                else _redact_login_response(item)
            )
        return redacted
    if isinstance(value, list):
        return [_redact_login_response(item) for item in value]
    return value


# The fingerprint SDK envelope submitted by the web client decodes to a JSON
# object carrying appId/organization/ep/data and runs to several thousand
# characters. The floor below only has to separate that shape from the storage
# keys it is easily confused with (smidV2 is a short timestamp+hex token, and
# deepseek-device-id:chat is a UUID), so it is deliberately far under the real
# length rather than an exact match on a value the SDK may re-version.
_DEVICE_ID_MIN_LENGTH = 200
_DEVICE_ID_B64_RE = re.compile(r"[A-Za-z0-9+/]+={0,2}\Z")


def _decode_device_id_envelope(candidate):
    """Return the JSON envelope carried by a device ID, or None when absent.

    The fingerprint SDK prefixes the base64 payload with a single tag
    character, so the encoded body is not aligned to the start of the string.
    Both the whole value and the value past that tag are attempted; only a
    body that decodes to an object carrying ``appId`` is accepted.
    """
    for offset in (0, 1):
        body = candidate[offset:]
        if not body or len(body) % 4 != 0:
            continue
        try:
            decoded = base64.b64decode(body, validate=True)
        except Exception:
            continue
        try:
            parsed = json.loads(decoded.decode("utf-8"))
        except Exception:
            continue
        if isinstance(parsed, dict) and "appId" in parsed:
            return parsed
    return None


def _looks_like_device_id(value):
    """Report whether a value has the structure the login endpoint expects.

    The risk check correlates the submitted identifier with the fingerprint
    SDK's envelope, so a value of the wrong shape leaves the request rejected
    exactly as a missing one would. Storage keys under similar names are
    rejected here to keep them from being cached and then reused indefinitely.
    """
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if len(candidate) < _DEVICE_ID_MIN_LENGTH:
        return False
    if not _DEVICE_ID_B64_RE.match(candidate):
        return False
    return _decode_device_id_envelope(candidate) is not None


def _generate_web_device_id():
    """Generate a structurally valid placeholder device ID for the endpoint.

    This keeps the request well formed when no real identifier can be obtained,
    but it is not a fingerprint: the risk check rejects it, and callers must
    treat rejection as the expected outcome rather than a transient failure.
    """
    envelope = json.dumps(
        {
            "appId": "default",
            "organization": "",
            "ep": "",
            # Padded so the placeholder also satisfies _looks_like_device_id;
            # the endpoint parses this field before the risk check runs.
            "data": "0" * 400,
        },
        separators=(",", ":"),
    )
    return base64.b64encode(envelope.encode("utf-8")).decode("ascii")


def device_id_file_path():
    """Resolve where the harvested DeepSeek device ID is stored.

    Mirrors cookie_file_path(): an explicit override wins, otherwise the file
    is placed next to the database so it lands inside the persistent Docker
    volume and survives container recreation.
    """
    p = os.getenv("DEEPSEEKER_DEVICE_ID_PATH")
    if p:
        return p
    d = os.path.dirname(os.path.abspath(_db))
    if d and os.path.abspath(d) != os.path.abspath(os.getcwd()):
        return os.path.join(d, "deepseek_device_id")
    return os.path.join(BASE_DIR, "deepseek_device_id")


def _read_persisted_device_id():
    """Return the stored device ID, or None when absent, unreadable or malformed.

    A malformed value is treated as absent rather than returned: an earlier run
    may have cached a storage key with a similar name, and reusing it would keep
    every subsequent login failing the risk check with no path to recovery short
    of deleting the file by hand.
    """
    try:
        with open(device_id_file_path()) as f:
            value = f.read().strip()
    except Exception:
        return None
    if not _looks_like_device_id(value):
        if value:
            logger.warning(
                "Discarding persisted DeepSeek device ID at %s: it does not have "
                "the shape the login endpoint expects",
                device_id_file_path(),
            )
        return None
    return value


def _persist_device_id(value):
    """Store a harvested device ID so later logins reuse the same identity.

    Identity reuse matters: the risk engine correlates logins by device, so a
    new random ID per login looks like an unfamiliar device each time. Values
    failing the structural check are refused here so a bad harvest cannot
    poison every later login.
    """
    if not _looks_like_device_id(value):
        logger.warning("Refusing to persist a malformed DeepSeek device ID")
        return
    target = device_id_file_path()
    try:
        target_dir = os.path.dirname(os.path.abspath(target))
        os.makedirs(target_dir, exist_ok=True)
        tmp_path = target + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(value)
        os.replace(tmp_path, target)
        logger.info("DeepSeek device ID persisted to %s", target)
    except Exception as e:
        logger.warning("Could not persist DeepSeek device ID to %s: %s", target, e)


# Runs in the page context. Only the SDK accessors are consulted: storage keys
# with similar names (smidV2, deepseek-device-id:chat) hold a different kind of
# identifier, and returning one would submit a value the risk check rejects.
_DEVICE_ID_PROBE_JS = """
async () => {
  const sdk = window.SMSdk || window.smSdk || window.SMSDK;
  if (!sdk) return "";
  for (const name of ["getDeviceId", "getDeviceIdSync", "getDeviceIdAsync"]) {
    try {
      const fn = sdk[name];
      if (typeof fn !== "function") continue;
      const value = await fn.call(sdk);
      if (typeof value === "string" && value) return value;
    } catch (e) {}
  }
  return "";
}
"""


def device_id_harvest_timeout():
    """Return the harvest budget, falling back when the setting is unusable.

    Resolved per call rather than at import: the variable is documented in
    .env.example, and a non-numeric value there must not abort module import
    and take the whole service down over an unrelated feature.
    """
    raw = os.getenv("DEEPSEEKER_DEVICE_ID_TIMEOUT", "60")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid DEEPSEEKER_DEVICE_ID_TIMEOUT=%r; using 60 seconds", raw
        )
        return 60.0


# Serializes harvesting so concurrent logins cannot launch several browsers.
_device_id_lock = asyncio.Lock()


async def _fetch_real_device_id():
    """Harvest the device ID the browser fingerprint SDK reports for this host.

    A random string does not satisfy the login risk check: the endpoint expects
    the identifier produced by the fingerprint SDK the web client loads. This
    opens the sign-in page in headless Chromium, waits for the SDK to publish
    its ID, and returns it.
    """
    if async_playwright is None:
        raise CookieGenerationError("playwright is not installed")
    launch_kwargs = {"headless": True}
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        launch_kwargs["args"] = ["--no-sandbox", "--disable-dev-shm-usage"]
    harvested = None
    last_error = None
    timeout = device_id_harvest_timeout()
    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_kwargs)
        try:
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                locale="zh-CN",
                viewport={"width": 1280, "height": 720},
            )
            page = await context.new_page()
            await page.goto(
                "https://chat.deepseek.com/sign_in",
                wait_until="domcontentloaded",
                timeout=45000,
            )
            # The SDK loads asynchronously; a short settle wait avoids polling
            # before its script has even been requested.
            with contextlib.suppress(Exception):
                await page.wait_for_load_state("networkidle", timeout=15000)
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    value = await page.evaluate(_DEVICE_ID_PROBE_JS)
                except Exception as e:
                    # Record rather than swallow: a renamed SDK global and a
                    # failed navigation both surface here, and the timeout
                    # message alone cannot distinguish them.
                    last_error = e
                    value = ""
                if _looks_like_device_id(value):
                    harvested = value.strip()
                    break
                await asyncio.sleep(1)
        finally:
            await browser.close()
    if not harvested:
        detail = f"; last page error: {last_error}" if last_error else ""
        raise CookieGenerationError(
            "the fingerprint SDK did not publish a device ID within "
            f"{timeout:.0f}s{detail}"
        )
    return harvested


async def _resolve_login_device_id(explicit_device_id=None):
    """Resolve the device ID submitted to the login endpoint.

    Precedence: explicit argument > DEEPSEEKER_DEVICE_ID > persisted file >
    freshly harvested ID > random placeholder. The random placeholder keeps the
    request well-formed but is known to be rejected by the risk check, so it is
    used only after every real source fails.
    """
    if explicit_device_id and str(explicit_device_id).strip():
        return str(explicit_device_id).strip()
    env_device_id = os.getenv("DEEPSEEKER_DEVICE_ID", "").strip()
    if env_device_id:
        return env_device_id
    persisted = _read_persisted_device_id()
    if persisted:
        return persisted
    async with _device_id_lock:
        # A concurrent caller may have harvested and cached it while we waited.
        persisted = _read_persisted_device_id()
        if persisted:
            return persisted
        try:
            harvested = await asyncio.wait_for(
                _fetch_real_device_id(), timeout=device_id_harvest_timeout() + 90
            )
        except Exception as e:
            logger.warning("Could not harvest a browser device ID: %s", e)
            harvested = None
        # Persist while still holding the lock: releasing it first would let a
        # concurrent caller read an empty file and launch a second browser,
        # defeating the double-check above.
        if harvested and _looks_like_device_id(harvested):
            _persist_device_id(harvested)
            return harvested
    logger.warning(
        "No device ID available; falling back to a placeholder that the risk "
        "check will most likely reject"
    )
    return _generate_web_device_id()


async def login_deepseek_account(email, password, device_id=None):
    """Log in through DeepSeek's web endpoint and return its bearer token."""
    if not email or not password:
        raise ValueError("邮箱和密码不能为空")
    # Match the browser login request; Android client headers can change the response shape.
    headers = {
        "content-type": "application/json",
        "accept": "application/json",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    }
    normalized_email = str(email).strip().replace("\\@", "@")
    # The risk check validates the fingerprint SDK's device ID, so a real value
    # is harvested and cached rather than generated per request.
    normalized_device_id = await _resolve_login_device_id(device_id)
    payload = {
        "email": normalized_email,
        "mobile": "",
        "password": password,
        "area_code": "",
        "device_id": normalized_device_id,
        "os": "web",
    }
    # Log the outbound contract for diagnosis, while never exposing the password.
    logger.info(
        "DeepSeek login request path=%s headers=%s payload=%s",
        "/api/v0/users/login",
        json.dumps(
            _redact_login_response(headers), ensure_ascii=False, separators=(",", ":")
        ),
        json.dumps(
            _redact_login_response(payload), ensure_ascii=False, separators=(",", ":")
        ),
    )
    response = await post_with_failover(
        "/api/v0/users/login",
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=30),
    )
    async with response:
        raw = await response.text()
        # Log non-JSON responses too; 202 responses are often empty during upstream checks.
        try:
            response_for_log = (
                json.dumps(
                    _redact_login_response(json.loads(raw)),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if raw
                else "<empty>"
            )
        except json.JSONDecodeError:
            response_for_log = raw[:4000] or "<empty>"
        logger.info(
            "DeepSeek login response status=%s body=%s",
            response.status,
            response_for_log,
        )
        if response.status != 200:
            # Never include the submitted password in an error or log message.
            detail = raw[:300] if raw else "empty response body"
            raise Exception(f"HTTP {response.status}: {detail}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise Exception("DeepSeek 登录接口返回了无效 JSON") from e
    token = _extract_login_token(data)
    if not token:
        if isinstance(data, dict):
            response_data = (
                data.get("data") if isinstance(data.get("data"), dict) else {}
            )
            biz_code = response_data.get("biz_code")
            biz_msg = response_data.get("biz_msg")
            logger.warning(
                "DeepSeek login returned no token; biz_code=%s biz_msg=%s response keys=%s data keys=%s",
                biz_code,
                biz_msg,
                sorted(data.keys()),
                sorted(response_data.keys())
                if response_data
                else type(data.get("data")).__name__,
            )
            if biz_code or biz_msg:
                raise Exception(
                    f"DeepSeek 登录失败（{biz_code or 'unknown'}: {biz_msg or 'unknown'}）"
                )
        raise Exception("DeepSeek 登录成功但响应中没有找到 Token")
    return token


# ==============================================================================
# BACKUP WAF COOKIE GENERATION (DEPRECATED IN FAVOR OF ANDROID CLIENT HEADERS)
#
# DeepSeek's backend does not enforce AWS WAF on requests using Android client
# headers. The Playwright/Chromium cookie generation below is retained as a backup
# so that if DeepSeek ever tightens WAF rules in the future, it can easily be
# re-enabled simply by uncommenting cookies=cookie in API calls.
# ==============================================================================

_cookie_lock = asyncio.Lock()

COOKIE_REGEN_ATTEMPTS = int(os.getenv("DEEPSEEKER_COOKIE_ATTEMPTS", "2"))
COOKIE_REGEN_TIMEOUT = float(os.getenv("DEEPSEEKER_COOKIE_TIMEOUT", "120"))
COOKIE_FAIL_COOLDOWN = float(os.getenv("DEEPSEEKER_COOKIE_COOLDOWN", "20"))

_cookie_fail = {"until": 0.0, "error": ""}


class CookieGenerationError(Exception):
    """Raised when the DeepSeek WAF cookie file cannot be produced."""


def _read_cookie_file():
    """Return valid (unexpired) cookies, or None."""
    try:
        with open(cookie_file_path()) as f:
            c = json.load(f)
        if c.get("expiry") is not None and c["expiry"] > time.time():
            return c["cookie"]
    except Exception:
        pass
    return None


def _read_stale_cookie_file():
    """Return cookies even if expired (last-resort fallback)."""
    try:
        with open(cookie_file_path()) as f:
            c = json.load(f)
        return c.get("cookie") or None
    except Exception:
        return None


async def get_cookies():
    cookies = _read_cookie_file()
    if cookies:
        return cookies

    def _cooldown_error():
        return CookieGenerationError(
            _cookie_fail["error"]
            + " (cooling down; will retry automatically — try again shortly)"
        )

    if time.time() < _cookie_fail["until"]:
        stale = _read_stale_cookie_file()
        if stale:
            return stale
        raise _cooldown_error()
    async with _cookie_lock:
        cookies = _read_cookie_file()
        if cookies:
            return cookies
        if time.time() < _cookie_fail["until"]:
            stale = _read_stale_cookie_file()
            if stale:
                return stale
            raise _cooldown_error()
        last_err = None
        for attempt in range(1, COOKIE_REGEN_ATTEMPTS + 1):
            try:
                logger.info(
                    "Generating DeepSeek cookies (attempt %d/%d)...",
                    attempt,
                    COOKIE_REGEN_ATTEMPTS,
                )
                await asyncio.wait_for(
                    _generate_cookies(), timeout=COOKIE_REGEN_TIMEOUT
                )
                cookies = _read_cookie_file()
                if cookies:
                    _cookie_fail["until"] = 0.0
                    _cookie_fail["error"] = ""
                    return cookies
                last_err = CookieGenerationError(
                    "cookie file missing/invalid after generation"
                )
            except Exception as e:
                last_err = e
                logger.warning(
                    "DeepSeek cookie generation attempt %d/%d failed: %s",
                    attempt,
                    COOKIE_REGEN_ATTEMPTS,
                    e,
                )
            if attempt < COOKIE_REGEN_ATTEMPTS:
                await asyncio.sleep(min(5 * attempt, 10))
        _cookie_fail["until"] = time.time() + COOKIE_FAIL_COOLDOWN
        _cookie_fail["error"] = f"Could not generate DeepSeek cookies: {last_err}"
        logger.error("%s", _cookie_fail["error"])
        stale = _read_stale_cookie_file()
        if stale:
            logger.warning(
                "Serving STALE DeepSeek cookies after generation failure (upstream may reject them)"
            )
            return stale
        raise CookieGenerationError(_cookie_fail["error"])


async def _generate_cookies():
    if async_playwright is None:
        raise CookieGenerationError("playwright is not installed")
    launch_kwargs = {"headless": False}
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        launch_kwargs["args"] = ["--no-sandbox"]
    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_kwargs)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(
                "https://chat.deepseek.com/",
                wait_until="domcontentloaded",
                timeout=45000,
            )
            await page.wait_for_selector("body", timeout=30000)
            with contextlib.suppress(Exception):
                await page.wait_for_url("**/sign_in*", timeout=30000)
            cookies = await context.cookies()
        finally:
            await browser.close()
    final_cookies = {}
    expiry = None
    for i in cookies:
        if i.get("name") == "aws-waf-token":
            expiry = i.get("expires")
        final_cookies[i["name"]] = i["value"]
    final_cookies["ds_cookie_preference"] = (
        "%257B%2522level%2522%253A%2522all%2522%257D"
    )
    if not expiry or expiry < 0:
        expiry = time.time() + 1800
    target = cookie_file_path()
    target_dir = os.path.dirname(os.path.abspath(target))
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = os.path.join(target_dir, os.path.basename(target) + ".tmp")
    with open(tmp_path, "w") as f:
        f.write(json.dumps({"cookie": final_cookies, "expiry": expiry}))
    os.replace(tmp_path, target)
    logger.info(
        "DeepSeek cookies saved to %s (expires %s)",
        target,
        datetime.fromtimestamp(expiry) if expiry else "n/a",
    )


def get_auth_token():
    conn = get_db()
    row = conn.execute("SELECT token FROM tokens LIMIT 1").fetchone()
    conn.close()
    if row:
        return row[0]
    return None


def add_token(token, alias=None, account_id=None):
    conn = get_db()
    if not conn.execute("SELECT 1 FROM tokens WHERE id = 1").fetchone():
        next_id = 1
    else:
        row = conn.execute("""
            SELECT min(t1.id + 1)
            FROM tokens t1
            LEFT JOIN tokens t2 ON t1.id + 1 = t2.id
            WHERE t2.id IS NULL
        """).fetchone()
        next_id = row[0] if row and row[0] else 1
    conn.execute(
        "INSERT INTO tokens (id, alias, token, status, account_id) VALUES (?, ?, ?, 'ACTIVE', ?)",
        (next_id, alias, token, account_id),
    )
    conn.commit()
    conn.close()
    return next_id


def get_tokens():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, alias, token, status, last_checked_at, last_check_error, account_id FROM tokens ORDER BY id"
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "alias": r[1],
            "token": r[2],
            "status": r[3],
            "last_checked_at": r[4],
            "last_check_error": r[5],
            "account_id": r[6],
        }
        for r in rows
    ]


def get_token(token_id):
    conn = get_db()
    row = conn.execute(
        "SELECT id, alias, token, status, last_checked_at, last_check_error, account_id FROM tokens WHERE id = ?",
        (token_id,),
    ).fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "alias": row[1],
            "token": row[2],
            "status": row[3],
            "last_checked_at": row[4],
            "last_check_error": row[5],
            "account_id": row[6],
        }
    return None


def delete_token(token_id):
    conn = get_db()
    conn.execute("DELETE FROM tokens WHERE id = ?", (token_id,))
    conn.commit()
    conn.close()


RATE_LIMIT_COOLDOWN_SEC = float(os.getenv("DEEPSEEKER_RATE_LIMIT_COOLDOWN", "60"))

# 单 token 软并发上限：在途数达到上限的 token 仅在无其它可用账号时才被
# 选中。账号数少而并发高时，纯「最少在途优先」仍会把多出来的请求压向
# 同一账号，软上限把这部分请求导向其它尚未打满的账号。
TOKEN_CONCURRENCY_CAP = int(os.getenv("DEEPSEEKER_TOKEN_CONCURRENCY", "8"))

# 各 token 当前在途的上游请求数，供 pick_token() 做负载均衡。
# 仅由事件循环线程读写：token_acquire/token_release 与 pick_token() 之间不存在
# await 边界，dict 的单项读写本身是原子的，因此无需额外加锁。
_inflight = {}


def token_acquire(token_id):
    """登记一个在途上游请求（对应一次 handle_chat 的完整生命周期）。"""
    if token_id is None:
        return
    _inflight[token_id] = _inflight.get(token_id, 0) + 1


def token_release(token_id):
    """注销一个在途上游请求；计数归零时删除键，避免字典随历史无限增长。"""
    if token_id is None:
        return
    remaining = _inflight.get(token_id, 0) - 1
    if remaining > 0:
        _inflight[token_id] = remaining
    else:
        _inflight.pop(token_id, None)


def inflight_snapshot():
    """当前在途计数的快照（排障用，不参与分配决策）。"""
    return dict(_inflight)


class TokenLease:
    """一次上游请求对某个 token 的在途占用凭证。

    与 _OwnedChatLock 同构：release() 幂等，只注销自己登记的那一份计数，
    因此可以在流式响应里把所有权转移给生成器、也可以在重试递归前提前
    释放。rebind() 支持 token 轮换：旧账号被限流而请求改由新账号承载时，
    把占用从旧 token 转移过去，避免旧计数泄漏。
    """

    __slots__ = ("_token_id",)

    def __init__(self, token_id=None):
        self._token_id = None
        if token_id is not None:
            token_acquire(token_id)
            self._token_id = token_id

    def rebind(self, token_id):
        if token_id == self._token_id:
            return
        self.release()
        if token_id is not None:
            token_acquire(token_id)
            self._token_id = token_id

    def release(self):
        if self._token_id is None:
            return
        token_release(self._token_id)
        self._token_id = None

    @property
    def token_id(self):
        return self._token_id


def pick_token(exclude=None):
    """选出当前最空闲的可用 token。

    改为「最少在途优先」而非无状态随机：账号数少而并发新会话多时，
    ORDER BY RANDOM() 的方差会让某个账号同时吸收多个请求、先撞上游的
    按账号限流，而其它账号仍然空闲。并列时依次按「最久未用」与 id 升序
    定序，保证同等空闲的账号之间继续均摊且结果确定。

    limited_until 冷却过滤：刚返回 429 的账号在冷却窗口内不进入候选集；
    冷却一旦到期即重新参与分配，并在本次选中时把状态翻回 ACTIVE，不再
    只依赖间隔长得多（默认 300s）的定时健康检查来恢复。永久性故障状态
    （INVALID / CHECK_ERROR）始终排除，只有限流这一种可自愈的状态享受
    冷却到期即恢复。

    在途数达到 TOKEN_CONCURRENCY_CAP 的账号排序靠后，仅在无其它可用账号
    时才被选中——软上限不会饿死任何账号，但能抑制并发向单一账号堆积。

    `exclude` 供重试轮换使用：列出刚失败过的 token id，只要还有其它可用
    账号就不回落到它们，避免重试再次命中被污染的账号。

    全部账号仍在冷却时返回最先恢复者（有界等待），而非 id 最小者；池中
    无任何 token 时返回 None。
    """
    conn = get_db()
    try:
        now = time.time()
        rows = conn.execute(
            "SELECT id, status, limited_until, last_used FROM tokens"
        ).fetchall()
        if not rows:
            return None

        excluded = set(exclude or ())
        available = []  # (token_id, last_used)：冷却已过、可参与分配
        recovered = []  # 冷却到期、本次翻回 ACTIVE 的 token
        cooling = []  # (token_id, 恢复时刻)：仍在冷却，用于全冷却兜底
        for tid, status, until, last_used in rows:
            if status not in ("ACTIVE", "RATE_LIMITED"):
                continue  # INVALID / CHECK_ERROR 属永久性故障，始终排除
            if status == "RATE_LIMITED" and until and until > now:
                cooling.append((tid, until))
                continue
            if status == "RATE_LIMITED":
                recovered.append(tid)
            available.append((tid, last_used or 0.0))

        pickable = [e for e in available if e[0] not in excluded] or available
        if pickable:
            if recovered:
                conn.executemany(
                    "UPDATE tokens SET status = 'ACTIVE', limited_until = NULL "
                    "WHERE id = ?",
                    [(tid,) for tid in recovered],
                )
                logger.info(
                    "Token pool: cooldown expired, auto-recovered token(s) %s",
                    recovered,
                )
            chosen = min(
                pickable,
                key=lambda e: (
                    1 if _inflight.get(e[0], 0) >= TOKEN_CONCURRENCY_CAP else 0,
                    _inflight.get(e[0], 0),
                    e[1],
                    e[0],
                ),
            )[0]
            # 刷新上次选中时刻，使「最久未用」在从未被限流的账号上同样成立。
            conn.execute("UPDATE tokens SET last_used = ? WHERE id = ?", (now, chosen))
            conn.commit()
            return chosen

        if cooling:
            return min(cooling, key=lambda e: (e[1], e[0]))[0]

        row = conn.execute("SELECT id FROM tokens ORDER BY id LIMIT 1").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def mark_limited(token_id):
    """将 token 标记为限流并进入冷却期。

    limited_until 让 pick_token() 在冷却窗口内绕开该账号；冷却到期后该账号
    重新进入候选集（见 pick_token），无需等待定时健康检查。健康检查探活
    成功仍会把状态正式恢复为 ACTIVE（见 update_token_check）。
    """
    logger.warning("Token #%d marked RATE_LIMITED", token_id)
    conn = get_db()
    conn.execute(
        "UPDATE tokens SET status = ?, limited_until = ? WHERE id = ?",
        ("RATE_LIMITED", time.time() + RATE_LIMIT_COOLDOWN_SEC, token_id),
    )
    conn.commit()
    conn.close()


def mark_active(token_id):
    """恢复账号为 ACTIVE，清除冷却截止时间并刷新上次选中时刻。

    last_used 在此同步刷新，使调度器的「最久未用」并列定序在限流恢复后
    不会把该账号立刻当成最旧候选。
    """
    conn = get_db()
    conn.execute(
        "UPDATE tokens SET status = ?, limited_until = NULL, last_used = ? WHERE id = ?",
        ("ACTIVE", time.time(), token_id),
    )
    conn.commit()
    conn.close()


def get_accounts():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, email, alias, token_id, status, last_login_at, last_error, created_at "
        "FROM accounts ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_account(account_id):
    conn = get_db()
    row = conn.execute(
        "SELECT token_id FROM accounts WHERE id = ?", (account_id,)
    ).fetchone()
    if row and row[0]:
        conn.execute("DELETE FROM tokens WHERE id = ?", (row[0],))
    conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
    conn.commit()
    conn.close()


def _set_account_result(account_id, status, error=None, token_id=None):
    conn = get_db()
    conn.execute(
        "UPDATE accounts SET status = ?, last_login_at = ?, last_error = ?, token_id = COALESCE(?, token_id) WHERE id = ?",
        (status, datetime.now(timezone.utc).isoformat(), error, token_id, account_id),
    )
    conn.commit()
    conn.close()


def save_account_login(email, alias, auth_token):
    """Create or replace an account and its corresponding pool token."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id, token_id FROM accounts WHERE email = ?", (email,)
    ).fetchone()
    if existing:
        account_id, token_id = existing[0], existing[1]
        if token_id:
            conn.execute(
                "UPDATE tokens SET alias = ?, token = ?, status = 'ACTIVE', last_check_error = NULL, account_id = ? WHERE id = ?",
                (alias or email, auth_token, account_id, token_id),
            )
        else:
            token_id = None
    else:
        account_id = None
        token_id = None

    if token_id is None:
        conn.commit()
        conn.close()
        token_id = add_token(auth_token, alias or email, account_id=None)
        conn = get_db()
        if account_id is None:
            cur = conn.execute(
                "INSERT INTO accounts (email, alias, token_id, status, last_login_at, last_error) VALUES (?, ?, ?, 'ACTIVE', ?, NULL)",
                (email, alias, token_id, datetime.now(timezone.utc).isoformat()),
            )
            account_id = cur.lastrowid
        else:
            conn.execute(
                "UPDATE accounts SET alias = ?, token_id = ?, status = 'ACTIVE', last_login_at = ?, last_error = NULL WHERE id = ?",
                (alias, token_id, datetime.now(timezone.utc).isoformat(), account_id),
            )
        conn.execute(
            "UPDATE tokens SET account_id = ? WHERE id = ?", (account_id, token_id)
        )
    else:
        conn.execute(
            "UPDATE accounts SET alias = ?, status = 'ACTIVE', last_login_at = ?, last_error = NULL WHERE id = ?",
            (alias, datetime.now(timezone.utc).isoformat(), account_id),
        )
    conn.commit()
    conn.close()
    return account_id


def update_token_check(token_id, status, error=None):
    """Persist the latest periodic/manual token check result."""
    conn = get_db()
    conn.execute(
        "UPDATE tokens SET status = ?, last_checked_at = ?, last_check_error = ?, "
        "limited_until = CASE WHEN ? = 'ACTIVE' THEN NULL ELSE limited_until END "
        "WHERE id = ?",
        (status, datetime.now(timezone.utc).isoformat(), error, status, token_id),
    )
    conn.execute(
        "UPDATE accounts SET status = ?, last_error = ? WHERE id = "
        "(SELECT account_id FROM tokens WHERE id = ? AND account_id IS NOT NULL)",
        (status, error, token_id),
    )
    conn.commit()
    conn.close()


async def check_token_status(token_id):
    """Probe authentication without creating a chat or sending model content."""
    token = get_token(token_id)
    if not token:
        return {"id": token_id, "status": "NOT_FOUND", "error": "Token 不存在"}

    try:
        # The challenge endpoint authenticates the bearer token and has no model
        # side effect, unlike chat_session/create or chat/completion.
        challenge = await create_challange_pow(
            "/api/v0/chat/completion", token["token"]
        )
        if isinstance(challenge, dict) and challenge.get("challenge"):
            update_token_check(token_id, "ACTIVE")
            return {"id": token_id, "status": "ACTIVE", "error": None}
        error = "上游返回了无效的认证挑战"
        update_token_check(token_id, "CHECK_ERROR", error)
        return {"id": token_id, "status": "CHECK_ERROR", "error": error}
    except Exception as e:
        message = str(e)[:300]
        match = re.match(r"HTTP (\d{3}):", message)
        code = int(match.group(1)) if match else None
        status = (
            "INVALID"
            if code in (401, 403)
            else "RATE_LIMITED"
            if code == 429
            else "CHECK_ERROR"
        )
        update_token_check(token_id, status, message)
        logger.warning(
            "Token #%s health check failed (%s): %s", token_id, status, message
        )
        return {"id": token_id, "status": status, "error": message}


def find_session(sig):
    conn = get_db()
    row = conn.execute(
        "SELECT token_id, deepseek_session_id, parent_message_id FROM sessions WHERE signature = ?",
        (sig,),
    ).fetchone()
    if row:
        # B13: touch on read so pruning evicts by real recency of USE, not by
        # insertion order — long-running chats get older rows every turn, and
        # they are precisely the sessions pruning must protect.
        conn.execute(
            "UPDATE sessions SET last_used = ? WHERE signature = ?", (time.time(), sig)
        )
        conn.commit()
    conn.close()
    if row:
        return {"token_id": row[0], "session_id": row[1], "parent_message_id": row[2]}
    return None


# Cap on stored session signatures. Every request stores 2 rows and nothing
# ever removed them, so after many chats the SQLite file (and its WAL) grew
# unbounded — on volume-limited deployments a full disk freezes ALL requests,
# including brand-new chats. PRUNE_EVERY saves trigger a prune that keeps the
# MAX_SESSIONS MOST-RECENTLY-USED rows (B13: by last_used, not insertion
# order — the old rowid policy evicted long-running chats, the most valuable
# sessions, exactly because they hold the oldest rows).
MAX_SESSIONS = int(os.getenv("DEEPSEEKER_MAX_SESSIONS", "20000"))
PRUNE_EVERY = int(os.getenv("DEEPSEEKER_PRUNE_EVERY", "500"))
_save_counter = {"n": 0}


def prune_sessions():
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        deleted = 0
        if total > MAX_SESSIONS:
            # B13: evict LEAST-RECENTLY-USED. NULL last_used (never re-hit)
            # sorts as 0, i.e. evicted first; rowid breaks ties deterministically.
            deleted = conn.execute(
                "DELETE FROM sessions WHERE rowid IN "
                "(SELECT rowid FROM sessions ORDER BY COALESCE(last_used, 0) ASC, rowid ASC LIMIT ?)",
                (total - MAX_SESSIONS,),
            ).rowcount
        deleted_map = conn.execute(
            "DELETE FROM session_map WHERE created_at < datetime('now', '-7 days')"
        ).rowcount
        conn.commit()
        if deleted or deleted_map:
            logger.info(
                "Pruned %d session signature(s) and %d stale session_map row(s)",
                deleted,
                deleted_map,
            )
    finally:
        conn.close()


def save_session(sig, token_id, session_id, parent_message_id=0):
    conn = get_db()
    conn.execute(
        """INSERT OR REPLACE INTO sessions (signature, token_id, deepseek_session_id, parent_message_id)
           VALUES (?, ?, ?, ?)""",
        (sig, token_id, session_id, parent_message_id),
    )
    conn.commit()
    conn.close()
    _save_counter["n"] += 1
    if _save_counter["n"] >= PRUNE_EVERY:
        _save_counter["n"] = 0
        try:
            prune_sessions()
        except Exception:
            logger.exception("Session pruning failed (non-fatal)")


def delete_session(sig):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE signature = ?", (sig,))
    conn.commit()
    conn.close()


def next_parent(parent_message_id):
    """Compute the parent_message_id to use for the next turn on a chat session.

    Invariant: each successful /chat/completion request appends exactly one user
    message and one assistant message to the DeepSeek chat session, so if the
    request was sent with parent_message_id P, the last message id afterwards is
    P + 1 and the next request must use P + 2.

    DeepSeek's web API exposes no endpoint to list a session's messages, so this
    increment cannot be verified against the server; it is centralized here so
    every save_session() call site (token-rotation branch, non-stream and both
    streaming paths) stays consistent if the invariant ever changes.
    """
    return parent_message_id + 2


def delete_sessions_for_chat(token_id, session_id):
    conn = get_db()
    conn.execute(
        "DELETE FROM sessions WHERE token_id = ? AND deepseek_session_id = ?",
        (token_id, session_id),
    )
    conn.commit()
    conn.close()


# ==============================================================================
# B4（Stage 1 审计）—— 文件归属注册表
#
# /v1/files 上传此前随机挑选 token，而上游的文件按账号隔离，因此后续
# /v1/chat/completions 引用该 file_id 时可能落到另一个账号并收到上游的
# “file not found”——OpenAI 风格的「上传后引用」流程（Claude Code / Cline
# 的文件流程）因此在设计上就是坏的。现在上传会钉在当时的 token 上；聊天
# 处理器首轮优先使用文件属主 token，后续轮次则把异主引用改挂到本会话自己
# 的 token 上。
# ==============================================================================


def record_file(file_id, token_id):
    """把上游 file_id 钉在拥有它的 token（账号）上。

    先到者胜：INSERT OR IGNORE 保证最初的映射不会被后来的上传改写。
    """
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO files (file_id, token_id, created_at) VALUES (?, ?, ?)",
        (file_id, token_id, time.time()),
    )
    conn.commit()
    conn.close()


def get_file_token(file_id):
    """返回拥有该 file_id 的 token id；未登记（历史遗留）时返回 None。"""
    conn = get_db()
    row = conn.execute(
        "SELECT token_id FROM files WHERE file_id = ?", (file_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


# DeepSeek now serves a single model (v4.1flash) as the website default. The
# web API uses the website default when model_type is null, so no explicit
# value is sent. If DeepSeek ever exposes an explicit model_type enum value
# for v4.1flash, this constant is the single place to set it.
DEFAULT_MODEL_TYPE = None

# 单价表已迁至 usage_stats.PRICING：按「缓存命中 / 缓存未命中 / 输出」三档、
# 高峰与低峰两档计价，由 format_response 的成本字段与 /stats 面板共用同一份
# 定义，避免两处价格表各自漂移。


def count_tokens(text, model="deepseek-v4.1-flash"):
    return len(deepseek_tokenizer.ds_token.encode(text))


# --- tokenizer 结果缓存 ------------------------------------------------------
# ds_token.encode() 实测只有约 0.6 MB/s，是全链路最慢的纯 CPU 操作，而同一段
# 文本在一次请求里会被统计多次（上下文估算、usage 上报、成本计算）。按内容摘要
# 缓存可以把它收敛为一次分词。
#
# key 用 sha1 摘要而非原文：单条工具结果可达几十 KB，拿原文当 key 会在长会话里
# 累积出可观的常驻内存；sha1 约 1 GB/s，比分词快三个数量级，开销可忽略。
_TOKEN_CACHE = OrderedDict()
_TOKEN_CACHE_MAX = int(os.getenv("DEEPSEEKER_TOKEN_CACHE_MAX", "4096"))


def count_tokens_cached(text):
    """带内容缓存的 count_tokens，返回值与 count_tokens 完全一致。

    纯同步实现，不含 await 点，因此在事件循环里不会被打断；未命中时的开销与
    直接调用 count_tokens 相同，命中时省下整次分词。
    """
    if not text:
        return 0
    key = hashlib.sha1(text.encode("utf-8", errors="replace")).digest()
    hit = _TOKEN_CACHE.get(key)
    if hit is not None:
        _TOKEN_CACHE.move_to_end(key)
        return hit
    n = count_tokens(text)
    _TOKEN_CACHE[key] = n
    while len(_TOKEN_CACHE) > _TOKEN_CACHE_MAX:
        _TOKEN_CACHE.popitem(last=False)
    return n


# --- Prompt-cache accounting ------------------------------------------------
# The DeepSeek web endpoint returns no token metadata, so the usage this bridge
# reports is computed locally and an upstream cache hit can never be read back.
# Caching still applies upstream, but on a different axis than the one the
# reported figure describes. The upstream chat is a stateful conversation that
# already holds every earlier turn, so the history a client resends is not
# reprocessed; only the new suffix is. The reported usage, however, describes
# the client's own full prompt (count_tok over the whole message list), so the
# cache figure must share that basis to mean anything to the caller.
#
# The two agree because the client resends the whole conversation every turn:
# the previous turn's prompt is exactly the reusable prefix of this turn's, and
# the difference is the genuinely new material. Remembering that size per
# conversation signature reproduces the figure DeepSeek's own API reports as
# `prompt_cache_hit_tokens` — for a session that has not been rebuilt.
# reset_cached_input() drops the entry whenever a rollover rewrites history,
# because then the previous prompt is no longer a prefix of the current one.
_CACHED_INPUT_HISTORY = OrderedDict()
_CACHED_INPUT_HISTORY_MAX = 512


def peek_cached_input(sig):
    """Prompt tokens this conversation can reuse from its previous turn."""
    return _CACHED_INPUT_HISTORY.get(sig, 0)


def record_cached_input(sig, input_tokens):
    """Remember this turn's input size as the next turn's cached prefix."""
    if not sig:
        return
    _CACHED_INPUT_HISTORY.pop(sig, None)
    _CACHED_INPUT_HISTORY[sig] = max(0, input_tokens)
    while len(_CACHED_INPUT_HISTORY) > _CACHED_INPUT_HISTORY_MAX:
        _CACHED_INPUT_HISTORY.popitem(last=False)


def reset_cached_input(sig):
    """Forget the remembered prefix after a rollover rebuilds the history."""
    _CACHED_INPUT_HISTORY.pop(sig, None)


# 工具调用的包装标签名。格式漂移时，DSML 提取分支无法匹配工具名，会把包装
# 标签名或其片段（如 "_call"、"|calls"）当作工具名外传，上层据此抛出
# unknown tool。此集合用于在工具名的唯一出口处拦截并丢弃。
_WRAPPER_TAG_NAMES = frozenset(
    {
        "calls",
        "call",
        "toolcall",
        "toolcalls",
        "functioncall",
        "invoke",
        "parameter",
        "param",
    }
)


def normalize_tool_call(tool_data_or_name, args_if_name=None):
    if isinstance(tool_data_or_name, str):
        name = tool_data_or_name
        args = args_if_name if args_if_name is not None else {}
    elif isinstance(tool_data_or_name, dict):
        tool_data = tool_data_or_name
        if "function" in tool_data and isinstance(tool_data["function"], dict):
            fn = tool_data["function"]
            name = fn.get("name") or tool_data.get("name")
            args = (
                fn.get("arguments")
                or fn.get("parameters")
                or fn.get("input")
                or fn.get("args")
                or fn.get("params")
                or {}
            )
        else:
            name = (
                tool_data.get("name")
                or tool_data.get("tool")
                or tool_data.get("tool_name")
                or tool_data.get("function")
                or tool_data.get("action")
            )
            args = (
                tool_data.get("arguments")
                or tool_data.get("parameters")
                or tool_data.get("input")
                or tool_data.get("args")
                or tool_data.get("params")
                or tool_data.get("tool_input")
                or tool_data.get("action_input")
                or {}
            )
    else:
        return None

    if not name or not isinstance(name, str):
        return None

    # 黑名单拦截：包装标签名不是真实工具调用，直接丢弃，由调用方继续查找
    # invoke 的真实 name= 属性。判定前先剥离分隔符，与 _WRAPPER_TAG_NAMES
    # 的存储形态保持一致，避免分隔符被剔除后漏判。
    probe = re.sub(r"[<>|｜_\-\s]", "", name).lower()
    if probe.startswith("dsml"):
        probe = probe[4:]
    if not probe or probe in _WRAPPER_TAG_NAMES:
        return None

    if isinstance(args, (dict, list)):
        args_str = json.dumps(args)
    elif isinstance(args, str):
        args_str = args
        try:
            json.loads(args_str)
        except Exception:
            args_str = json.dumps(args_str)
    else:
        args_str = json.dumps({})
    call_id = "call_" + "".join(
        random.choices(string.ascii_letters + string.digits, k=8)
    )
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": args_str,
        },
    }


def clean_json_str(s):
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _code_fence_spans(text):
    """Return the (start, end) spans of markdown fenced code blocks.

    Tool-call markup inside a fence is documentation/example text, not an
    actual tool call, so matches falling inside these spans are ignored.
    Unclosed fences extend to end-of-text.
    """
    return [
        (m.start(), m.end()) for m in re.finditer(r"```.*?(?:```|$)", text, re.DOTALL)
    ]


def parse_tools(text):
    tools = []
    clean_text = text
    fence_spans = _code_fence_spans(text)

    def fenced(pos):
        return any(s <= pos < e for s, e in fence_spans)

    param_names = {
        "command",
        "description",
        "file_path",
        "content",
        "path",
        "prompt",
        "query",
        "subject",
        "old_string",
        "new_string",
        "url",
        "input",
    }
    tool_matches = list(
        re.finditer(
            r"<[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_call|invoke|function_call)\s+(?:name|tool)=[\x27\x22]([^\x27\x22]+)[\x27\x22][^>]*>",
            text,
            re.IGNORECASE,
        )
    )
    real_tool_matches = [tm for tm in tool_matches if not fenced(tm.start())]

    if real_tool_matches:
        for i, tm in enumerate(real_tool_matches):
            candidate_name = tm.group(1).strip()
            start_idx = tm.end()
            end_idx = (
                real_tool_matches[i + 1].start()
                if i + 1 < len(real_tool_matches)
                else len(text)
            )
            body = text[start_idx:end_idx]
            args = {}
            p_matches = re.finditer(
                r"<[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:parameter|tool_call|param|invoke)\s+name=[\x27\x22]([^\x27\x22]+)[\x27\x22][^>]*>(.*?)(?:</[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:parameter|tool_call|param|invoke)>|(?=<[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:parameter|tool_call|param|invoke)\s+name=)|$)",
                body,
                flags=re.DOTALL | re.IGNORECASE,
            )
            for pm in p_matches:
                p_name = pm.group(1).strip()
                p_val = pm.group(2).strip()
                p_val = re.sub(
                    r"</?(?:tool_calls?|invoke|function_call|parameter|param)\b[^>]*>",
                    "",
                    p_val,
                    flags=re.IGNORECASE,
                ).strip()
                try:
                    args[p_name] = json.loads(p_val)
                except Exception:
                    args[p_name] = p_val
            tag_param_matches = re.finditer(
                r"<([A-Za-z0-9_\-]+)>(.*?)(?:</\1>|$)",
                body,
                flags=re.DOTALL | re.IGNORECASE,
            )
            for pm in tag_param_matches:
                t_name = pm.group(1).strip().lower()
                if t_name in param_names:
                    t_val = pm.group(2).strip()
                    t_val = re.sub(
                        r"</?(?:tool_calls?|invoke|function_call|parameter|param)\b[^>]*>",
                        "",
                        t_val,
                        flags=re.IGNORECASE,
                    ).strip()
                    try:
                        args[t_name] = json.loads(t_val)
                    except Exception:
                        args[t_name] = t_val
            if candidate_name:
                norm = normalize_tool_call(candidate_name, args)
                if norm:
                    tools.append(norm)

    if tools:
        clean_text = re.sub(
            r"<[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_calls?|calls)[^>]*>.*?(?:</[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_calls?|calls)>|$)",
            "",
            clean_text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        clean_text = re.sub(
            r"<[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:invoke|function_call)[^>]*>.*?(?:</[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:invoke|function_call)>|$)",
            "",
            clean_text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    if not tools and "DSML" in text:
        dsml_block_pattern = re.compile(
            r"<[｜\|]{0,2}DSML[｜\|]{0,2}\s*([A-Za-z0-9_]+)>(.*?)(?:</[｜\|]{0,2}DSML[｜\|]{0,2}\s*\1>|$)",
            re.DOTALL | re.IGNORECASE,
        )
        param_pattern_b = re.compile(
            r"<[｜\|]{0,2}DSML[｜\|]{0,2}\s*B([A-Za-z0-9_]+)[^>]*>(.*?)(?:</[｜\|]{0,2}DSML[｜\|]{0,2}\s*B.*?>|$)",
            re.DOTALL | re.IGNORECASE,
        )
        for m in dsml_block_pattern.finditer(text):
            if fenced(m.start()):
                continue
            tool_name = m.group(1).strip()
            body = m.group(2)
            args = {}
            for pm in param_pattern_b.finditer(body):
                p_name = pm.group(1).lower().strip()
                p_val = pm.group(2).strip()
                try:
                    args[p_name] = json.loads(p_val)
                except Exception:
                    args[p_name] = p_val
            norm = normalize_tool_call(tool_name, args)
            if norm:
                tools.append(norm)
        if not tools:
            tool_match = re.search(
                r"[｜\|]{0,2}DSML[｜\|]{0,2}\s*(Bash|Read|Write|Edit|Agent|TaskList|TaskCreate|WebSearch|[A-Za-z0-9_]+)",
                text,
                re.IGNORECASE,
            )
            if tool_match and not fenced(tool_match.start()):
                candidate = tool_match.group(1).strip()
                tool_name = (
                    "Bash"
                    if candidate.lower().startswith("b")
                    and candidate.lower() not in ["bdescription", "bparam"]
                    else candidate
                )
                args = {}
                cmd_match = re.search(
                    r"[｜\|]{2}B[\x22\x27]?command[\x22\x27]?[^>]*>(.*?)(?:</[｜\|]{2}B|$)",
                    text,
                    re.DOTALL | re.IGNORECASE,
                )
                desc_match = re.search(
                    r"[｜\|]{2}B[\x22\x27]?description[\x22\x27]?[^>]*>(.*?)(?:</[｜\|]{2}B|$)",
                    text,
                    re.DOTALL | re.IGNORECASE,
                )
                if cmd_match:
                    clean_cmd = re.sub(
                        r"</?[｜\|]{0,2}DSML[｜\|]{0,2}\s*[^>]*>",
                        "",
                        cmd_match.group(1),
                    ).strip("\x22\x27() ")
                    args["command"] = clean_cmd
                if desc_match:
                    clean_desc = re.sub(
                        r"</?[｜\|]{0,2}DSML[｜\|]{0,2}\s*[^>]*>",
                        "",
                        desc_match.group(1),
                    ).strip("\x22\x27() ")
                    args["description"] = clean_desc
                norm = normalize_tool_call(tool_name, args)
                if norm:
                    tools.append(norm)
        if tools:
            clean_text = re.sub(
                r"<[｜\|]{0,2}DSML[｜\|]{0,2}\s*[^>]*>.*?(?:</[｜\|]{0,2}DSML[｜\|]{0,2}\s*[^>]*>|$)",
                "",
                clean_text,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            clean_text = re.sub(
                r"</?[｜\|]{0,2}DSML[｜\|]{0,2}\s*[^>]*>",
                "",
                clean_text,
                flags=re.IGNORECASE,
            ).strip()

    if not tools:
        fn_call_pattern = re.compile(
            r"<function_call>\s*<name>([^<]+)</name>\s*<arguments>(.*?)</arguments>\s*</function_call>",
            re.DOTALL | re.IGNORECASE,
        )
        for m in fn_call_pattern.finditer(text):
            if fenced(m.start()):
                continue
            name = m.group(1).strip()
            args_raw = m.group(2).strip()
            try:
                args = json.loads(args_raw)
            except Exception:
                args = args_raw
            norm = normalize_tool_call(name, args)
            if norm:
                tools.append(norm)
        if tools:
            clean_text = re.sub(
                r"<function_call>.*?</function_call>",
                "",
                clean_text,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()

    if not tools:
        tag_regex = re.compile(
            r"<[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_call|function_call)(?:\s+(?:name|tool|function)=[\x27\x22]([^\x27\x22]+)[\x27\x22])?\s*>",
            re.IGNORECASE,
        )
        decoder = json.JSONDecoder()
        matches = [m for m in tag_regex.finditer(text) if not fenced(m.start())]
        if matches:
            for m in matches:
                tag_name = m.group(1)
                after_tag = text[m.end() :]
                brace_pos = after_tag.find("{")
                if brace_pos != -1:
                    json_substr = after_tag[brace_pos:]
                    data = None
                    with contextlib.suppress(Exception):
                        data, _ = decoder.raw_decode(json_substr)
                    if not data:
                        cleaned_json = re.sub(
                            r"</?[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_call|function_call|tool_calls|invoke)[^>]*>.*",
                            "",
                            json_substr,
                            flags=re.DOTALL,
                        ).strip()
                        open_b = cleaned_json.count("{")
                        close_b = cleaned_json.count("}")
                        if open_b > close_b:
                            cleaned_json += "}" * (open_b - close_b)
                        with contextlib.suppress(Exception):
                            data = json.loads(cleaned_json)
                    if isinstance(data, dict):
                        if tag_name:
                            name = tag_name
                            if "arguments" in data and isinstance(
                                data["arguments"], dict
                            ):
                                args = data["arguments"]
                            elif "parameters" in data and isinstance(
                                data["parameters"], dict
                            ):
                                args = data["parameters"]
                            elif "input" in data and isinstance(data["input"], dict):
                                args = data["input"]
                            else:
                                args = {
                                    k: v
                                    for k, v in data.items()
                                    if k not in ["name", "tool", "function"]
                                }
                        else:
                            name = (
                                data.get("name")
                                or data.get("tool")
                                or data.get("tool_name")
                                or data.get("function")
                                or data.get("action")
                            )
                            args = (
                                data.get("arguments")
                                or data.get("parameters")
                                or data.get("input")
                                or data.get("args")
                                or data.get("params")
                                or data.get("tool_input")
                                or data.get("action_input")
                            )
                            if args is None:
                                args = {}
                        if name:
                            norm = normalize_tool_call(name, args)
                            if norm:
                                tools.append(norm)
            clean_text = re.sub(
                r"<[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_call|function_call)[^>]*>.*?(?:</[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_call|function_call)>|$)",
                "",
                text,
                flags=re.DOTALL,
            ).strip()

    if not tools:
        codeblock_pattern = r"```(?:tool_call|function_call)\s*(.*?)\s*```"
        cb_matches = list(re.finditer(codeblock_pattern, clean_text, flags=re.DOTALL))
        for m in cb_matches:
            cleaned = clean_json_str(m.group(1))
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    name = (
                        data.get("name")
                        or data.get("tool")
                        or data.get("function")
                        or data.get("action")
                    )
                    args = (
                        data.get("arguments")
                        or data.get("parameters")
                        or data.get("input")
                        or data.get("args")
                        or {}
                    )
                    norm = normalize_tool_call(name, args)
                    if norm:
                        tools.append(norm)
            except Exception:
                pass
        if tools:
            clean_text = re.sub(
                codeblock_pattern, "", clean_text, flags=re.DOTALL
            ).strip()

    if not tools:
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        json_matches = list(re.finditer(json_pattern, clean_text, flags=re.DOTALL))
        for m in json_matches:
            cleaned = clean_json_str(m.group(1))
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict) and (
                    "name" in data or "tool" in data or "function" in data
                ):
                    name = (
                        data.get("name")
                        or data.get("tool")
                        or data.get("function")
                        or data.get("action")
                    )
                    args = (
                        data.get("arguments")
                        or data.get("parameters")
                        or data.get("input")
                        or data.get("args")
                        or {}
                    )
                    norm = normalize_tool_call(name, args)
                    if norm:
                        tools.append(norm)
            except Exception:
                pass
        if tools:
            clean_text = re.sub(json_pattern, "", clean_text, flags=re.DOTALL).strip()
    # Keep companion text: the tool-call blocks themselves were already removed
    # from clean_text above; only leftover bare tags are stripped here.
    clean_text = re.sub(
        r"</?[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_calls?|calls|invoke|function_call|parameter)[^>]*>",
        "",
        clean_text,
        flags=re.IGNORECASE,
    ).strip()
    return tools, clean_text


# Family-wide closer pattern for the tool-call wrapper family, including
# the |/｜-decorated variants that parse_tools accepts. Used by
# StreamToolParser so a mismatched closer closes the open block instead of
# hanging until flush(). [FIX 3]
_TOOL_END_TAG_RE = re.compile(
    r"</[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_calls?|calls|invoke|function_call)\s*>",
    re.IGNORECASE,
)

# [FIX 4] Spaced-DSML dialect: deepseek-harness emits decorated tags with a
# space between the ｜｜DSML｜｜ marker and the tag name, e.g.
# "<｜｜DSML｜｜ invoke name=...>". Entry detection cannot rely on plain
# substring start tags; this regex accepts bars, the DSML marker and the
# space in any combination.
_STREAM_ENTRY_RE = re.compile(
    r"<[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(tool_calls?|calls|function_call|invoke)\b[^>]*>",
    re.IGNORECASE,
)

# Orphan closers trailing a block already closed by the per-tag or family
# fallback (e.g. "</｜｜DSML｜｜ calls>" after the inner invoke was flushed).
# NOTE: parameter/param MUST be listed here. The model can emit a bare
# closing parameter tag with no matching opener (e.g. when an invoke block was
# already consumed), and anything this regex misses is dumped to the client as
# prose. parse_tools() already strips parameter via its own regex (line ~1132),
# which is why the persisted history is clean and a page refresh looks correct
# while the live stream is polluted.
# Do NOT add parameter to _TOOL_END_TAG_RE: that pattern closes an OPEN tool
# block, and a parameter closer sits inside the block, so matching it would cut
# the block short before </invoke> arrives.
_ORPHAN_CLOSER_RE = re.compile(
    r"</[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_calls?|calls|function_call|invoke|param(?:eter)?)\s*[｜\|]{0,2}>",
    re.IGNORECASE,
)

# Tag names _STREAM_ENTRY_RE can open on.
_STREAM_ENTRY_TAGS = ("tool_calls", "tool_call", "function_call", "invoke", "calls")

# 孤立闭合标签的候选标签名，比入口标签多出 param/parameter。模型可能发出
# 没有配对开标签的 parameter 闭合标签，分片途中这种片段必须被 hold 住，
# 否则会被当成正文漏给客户端（见 _ORPHAN_CLOSER_RE 的说明）。
_STREAM_CLOSER_TAGS = _STREAM_ENTRY_TAGS + ("param", "parameter")

# 推理(reasoning)通道的标记入口。比工具入口多出 param/parameter：模型在思考
# 中直接书写参数标签时不会有外层 invoke 包裹，漏掉这两个名字会让参数标签
# 原样透传（app.py 的 reasoning 分支只做净化，不产出工具调用）。
_REASONING_MARKUP_RE = re.compile(
    r"<[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*"
    r"(tool_calls?|calls|function_call|invoke|param(?:eter)?)\b[^>]*>",
    re.IGNORECASE,
)


# 未闭合块的缓冲上限。闭标签始终不到时按此上限丢弃并复位，避免异常流让
# 缓冲无界增长；正常工具块远小于该值。
_MAX_REASONING_BLOCK = 1 << 20


def _markup_closer_re(tag_name):
    """构造与 tag_name 配对的闭标签正则，容忍竖线与 DSML 装饰。"""
    return re.compile(
        r"</[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*"
        + re.escape(tag_name)
        + r"\s*[｜\|]{0,2}>",
        re.IGNORECASE,
    )


class ReasoningMarkupFilter:
    """从推理(reasoning)分片中移除 DSML 工具标记。

    parse_tools() 只作用于正文，因此推理通道会把模型在思考中写出的工具标记
    原样透传给客户端，表现为思考内容里出现 ``｜｜DSML｜｜`` 一类的协议残留。
    本过滤器复用正文同一套标签文法做净化，并处理两类分片边界：

    ① 标签本身被切断——可能是标签前缀的尾部暂存到下一片再判定，判定依据
       与 StreamToolParser 相同（_is_plausible_stream_entry_prefix 等）；
    ② 开标签已到、闭标签未到——此时整段内容都属于待丢弃的块，必须一并丢弃。
       若只删标签而保留内容，参数值会作为思考文本漏出，等于把泄漏换了个形态。

    闭标签始终不到时缓冲按 _MAX_REASONING_BLOCK 上限丢弃并复位，避免异常流
    造成无界增长；flush() 负责在流结束时释放残余。
    """

    def __init__(self):
        self._buffer = ""
        self._in_block = False
        self._closer_re = None

    def feed(self, chunk):
        self._buffer += chunk
        out = []
        while True:
            if self._in_block:
                match = (
                    self._closer_re.search(self._buffer) if self._closer_re else None
                )
                if match is None:
                    # 只接受工具族闭标签：parameter/param 位于块内部，若在此
                    # 收束，块内参数值会落到块外被当作思考文本输出。
                    match = _TOOL_END_TAG_RE.search(self._buffer)
                if match is None:
                    if len(self._buffer) > _MAX_REASONING_BLOCK:
                        # 闭标签始终未到：丢弃整块并复位，避免缓冲无限增长。
                        self._buffer = ""
                        self._in_block = False
                        self._closer_re = None
                    return "".join(out)
                self._buffer = self._buffer[match.end() :]
                self._in_block = False
                self._closer_re = None
                continue
            match = _REASONING_MARKUP_RE.search(self._buffer)
            if match:
                out.append(_ORPHAN_CLOSER_RE.sub("", self._buffer[: match.start()]))
                self._buffer = self._buffer[match.end() :]
                self._in_block = True
                self._closer_re = _markup_closer_re(match.group(1))
                continue
            last_lt = self._buffer.rfind("<")
            if last_lt != -1:
                tail = self._buffer[last_lt:]
                if ">" not in tail and (
                    _is_plausible_stream_entry_prefix(tail, _STREAM_CLOSER_TAGS)
                    or _is_plausible_stream_closer_prefix(tail)
                ):
                    out.append(_ORPHAN_CLOSER_RE.sub("", self._buffer[:last_lt]))
                    self._buffer = tail
                    return "".join(out)
            out.append(_ORPHAN_CLOSER_RE.sub("", self._buffer))
            self._buffer = ""
            return "".join(out)

    def flush(self):
        """释放流结束时的残余缓冲；未闭合的块内容一并丢弃。"""
        buffered = self._buffer
        was_in_block = self._in_block
        self._buffer = ""
        self._in_block = False
        self._closer_re = None
        if was_in_block:
            # 块未闭合即流结束：缓冲内容属于待丢弃的工具标记，整体丢弃。
            return ""
        return _ORPHAN_CLOSER_RE.sub("", buffered)


def _skip_bars(text, pos):
    for _ in range(2):
        if pos < len(text) and text[pos] in "|｜":
            pos += 1
    return pos


def _is_plausible_stream_entry_prefix(segment: str, tags=None) -> bool:
    """segment 是否为某标签名的前缀（容忍 DSML 装饰、竖线、空格）。

    tags 省略时按工具入口标签判断；判断闭合标签前缀时由
    _is_plausible_stream_closer_prefix 传入 _STREAM_CLOSER_TAGS。
    """
    if not segment.startswith("<") or ">" in segment:
        return False
    body = segment[1:]
    pos = _skip_bars(body, 0)
    length = len(body)
    if pos < length and body[pos] in "Dd":
        matched = 0
        while matched < 4 and pos < length and body[pos].upper() == "DSML"[matched]:
            pos += 1
            matched += 1
        if matched < 4:
            return pos == length
        pos = _skip_bars(body, pos)
    while pos < length and body[pos].isspace():
        pos += 1
    if pos == length:
        return True
    remainder = body[pos:]
    remainder_lower = remainder.lower()
    for tag in tags or _STREAM_ENTRY_TAGS:
        if tag.startswith(remainder_lower):
            return True
        if remainder_lower.startswith(tag):
            suffix = remainder[len(tag) :]
            if not suffix or not (suffix[0].isalnum() or suffix[0] == "_"):
                return True
    return False


def _is_plausible_stream_closer_prefix(segment: str) -> bool:
    if not segment.startswith("</"):
        return False
    return _is_plausible_stream_entry_prefix("<" + segment[2:], _STREAM_CLOSER_TAGS)


class StreamToolParser:
    def __init__(self):
        self.buffer = ""
        self.in_tool = False
        self.has_tool = False
        self.json_done = False
        self._end_re = None

    def feed(self, chunk):
        self.buffer += chunk
        results = []
        while True:
            if self.in_tool:
                # [FIX 3] Family-wide closer fallback: accept ANY wrapper closer
                # the tool-call family can emit (including |/｜-decorated
                # variants parse_tools tolerates) instead of hanging an open
                # block until flush() when the model closes with a wrong tag.
                # [FIX 4] Prefer the closer for the tag that OPENED the block
                # so nested wrappers (<｜｜DSML｜｜ calls> wrapping invokes)
                # close as one unit; fall back to the family-wide closer for
                # mismatched or decorated closers [FIX 3].
                end_match = self._end_re.search(self.buffer) if self._end_re else None
                if end_match is None:
                    end_match = _TOOL_END_TAG_RE.search(self.buffer)
                if end_match:
                    if not self.json_done:
                        tool_xml = self.buffer[: end_match.end()]
                        parsed, _ = parse_tools(tool_xml)
                        for item in parsed:
                            results.append({"tool": item})
                    self.buffer = self.buffer[end_match.end() :]
                    self.in_tool = False
                    self.json_done = False
                    self._end_re = None
                    continue
                brace_idx = self.buffer.find("{")
                if brace_idx != -1 and not self.json_done:
                    decoder = json.JSONDecoder()
                    try:
                        data, consumed = decoder.raw_decode(self.buffer[brace_idx:])
                        norm = normalize_tool_call(data)
                        if norm:
                            results.append({"tool": norm})
                            self.json_done = True
                            self.buffer = self.buffer[brace_idx + consumed :]
                            continue
                    except Exception:
                        pass
                break
            else:
                # [FIX 4] Regex entry detection replaces plain substring finds
                # so decorated/spaced DSML openers enter tool mode.
                m = _STREAM_ENTRY_RE.search(self.buffer)
                if m:
                    start = m.start()
                    if start > 0:
                        results.append(
                            {"text": _ORPHAN_CLOSER_RE.sub("", self.buffer[:start])}
                        )
                    self.buffer = self.buffer[start:]
                    tag_name = m.group(1).lower()
                    self._end_re = re.compile(
                        r"</[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*"
                        + re.escape(tag_name)
                        + r"[｜\|]{0,2}\s*>",
                        re.IGNORECASE,
                    )
                    self.in_tool = True
                    self.has_tool = True
                    continue
                # [FIX 2] Hold an unclosed '<' tail only while it remains a
                # plausible partial match of _STREAM_ENTRY_RE or of a wrapper
                # closer ("</..."), so a split closer is never dumped as prose.
                last_lt = self.buffer.rfind("<")
                tail = self.buffer[last_lt:] if last_lt != -1 else ""
                hold = ">" not in tail and (
                    _is_plausible_stream_entry_prefix(tail)
                    or _is_plausible_stream_closer_prefix(tail)
                )
                if last_lt != -1 and hold:
                    if last_lt > 0:
                        results.append(
                            {"text": _ORPHAN_CLOSER_RE.sub("", self.buffer[:last_lt])}
                        )
                    self.buffer = tail
                    break
                if self.buffer:
                    results.append({"text": _ORPHAN_CLOSER_RE.sub("", self.buffer)})
                self.buffer = ""
                break
        return results

    def flush(self):
        out = []
        if self.buffer and not self.in_tool:
            out.append({"text": self.buffer})
        elif self.in_tool:
            # [FIX 1] Recover the tool before stripping: if the stream was cut
            # off before the closing tag arrived but the payload itself is
            # parseable, emit the tool call instead of dumping raw parameter
            # values into the chat text. Strip only when nothing parses.
            parsed, _ = parse_tools(self.buffer)
            if parsed:
                for item in parsed:
                    out.append({"tool": item})
            elif not self.json_done:
                stripped = re.sub(
                    r"</?[｜\|]{0,2}(?:DSML[｜\|]{0,2})?\s*(?:tool_calls?|calls|invoke|function_call|parameter)[^>]*>",
                    "",
                    self.buffer,
                    flags=re.IGNORECASE,
                ).strip()
                if stripped:
                    out.append({"text": stripped})
            # With json_done set, whatever is left after the consumed JSON
            # payload is wrapper noise (partial closers / whitespace) and is
            # dropped instead of leaking into the chat text.
        self.buffer = ""
        self.in_tool = False
        self.json_done = False
        self._end_re = None
        return out


def summarize_messages(messages, max_tokens=500):
    recent = messages[-10:] if len(messages) > 10 else messages
    parts = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        if content:
            parts.append(f"{role}: {content[:200]}")
    summary = "\n".join(parts)
    tokens = count_tokens(summary)
    while tokens > max_tokens and len(parts) > 1:
        parts = parts[1:]
        summary = "\n".join(parts)
        tokens = count_tokens(summary)
    return summary


_pow_setup = None


def _get_pow_setup():
    global _pow_setup
    if _pow_setup is None:
        engine = wasmtime.Engine()
        with open(wasm_path, "rb") as f:
            module = wasmtime.Module(engine, f.read())
        _pow_setup = (engine, module, wasmtime.Linker(engine))
    return _pow_setup


def _find_pow_answer_blocking(challange_data):
    engine, module, linker = _get_pow_setup()
    store = wasmtime.Store(engine)
    instance = linker.instantiate(store, module)
    memory = instance.exports(store)["memory"]
    alloc_func = instance.exports(store)["alloc"]
    solve_func = instance.exports(store)["solve_pow"]
    ch_ptr, ch_len = write_string_pow(
        challange_data["challenge"], alloc_func, memory, store
    )
    salt_ptr, salt_len = write_string_pow(
        challange_data["salt"], alloc_func, memory, store
    )
    result = solve_func(
        store,
        ch_ptr,
        ch_len,
        salt_ptr,
        salt_len,
        challange_data["expire_at"],
        challange_data["difficulty"],
    )
    if result < 0:
        result = result + 0x10000000000000000
    return result if result != 0xFFFFFFFFFFFFFFFF else None


async def create_challange_pow(target_path, auth_token):
    headers = get_headers(auth_token)
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    response = await post_with_failover(
        "/api/v0/chat/create_pow_challenge",
        headers=headers,
        json={"target_path": target_path},
        # cookies=cookie,  # Backup WAF fallback
        timeout=aiohttp.ClientTimeout(total=20),
    )
    async with response:
        data = await response.json()
    return data["data"]["biz_data"]["challenge"]


def write_string_pow(text, alloc_func, memory, store):
    data = text.encode("utf-8")
    ptr = alloc_func(store, len(data))
    mem = memory.data_ptr(store)
    for i in range(len(data)):
        mem[ptr + i] = data[i]
    return ptr, len(data)


async def find_pow_answer(challange_data):
    return await asyncio.to_thread(_find_pow_answer_blocking, challange_data)


# ==============================================================================
# PoW 预取池
#
# solve_create_pow() 是每个上游请求前的必经关卡，由两段串行工作组成：一次到
# create_pow_challenge 的 HTTP 往返，加一次 wasm 解算。两者都排在第一个字节
# 发给上游之前，所以全额计入用户感知的首字延迟。
#
# 但 challenge 不依赖本轮的消息内容——它的约束只有 expire_at，以及签发给哪个
# target_path / token。这意味着整段等待可以提前做掉：上一轮收尾时预取，下一轮
# 到达时直接取用。
#
# 池按 (target_path, auth_token) 分桶。token 必须进 key：challenge 按 token 签发
# 并被 token 校验，跨 token 复用会被上游拒绝；token 池还会在 mark_limited /
# mark_active 之间轮换，复用等于随机 403。target_path 也必须进 key：
# chat/completion 与 file/upload_file 是两套互相独立、不能顶替的挑战。
#
# 取用路径必须能降级：池里命中且未过期才用，否则原地走同步求解。任何预取失败
# 都被吞掉——池只是加速器，坏掉时行为退化成今天的样子，而不是报错。
# ==============================================================================

_POW_POOL = {}
_POW_POOL_MAX = int(os.getenv("DEEPSEEKER_POW_POOL_MAX", "64"))
# 不掌握上游真实有效期时的保守兜底；拿到 expire_at 就用更早的那个。
_POW_PREFETCH_TTL = float(os.getenv("DEEPSEEKER_POW_PREFETCH_TTL", "60"))
_POW_PREFETCH_ENABLED = os.getenv(
    "DEEPSEEKER_POW_PREFETCH", "1"
).strip().lower() not in ("0", "false", "no")
_pow_prefetch_tasks = set()


def _pow_expire_epoch(pow_data):
    """把 challenge 的 expire_at 归一化成秒级 Unix 时间戳；识别不了返回 None。

    expire_at 的单位没有文档保证，所以按量级判断：超过 1e11 视为毫秒。落在
    合理区间之外的值（纳秒、损坏值）一律返回 None，交由调用方退回固定 TTL，
    而不是拿一个错误的过期时间去做判断。
    """
    try:
        value = float(pow_data.get("expire_at"))
    except (TypeError, ValueError, AttributeError):
        return None
    if value <= 0:
        return None
    if value > 1e11:
        value /= 1000.0
    if value < 1e9 or value > 4e9:
        return None
    return value


def _build_pow_header(pow_data, answer, target_path):
    """把 challenge 与解算结果打包成 x-ds-pow-response 需要的 base64 串。"""
    json_data = {
        "algorithm": "DeepSeekHashV1",
        "challenge": pow_data["challenge"],
        "salt": pow_data["salt"],
        "answer": answer,
        "signature": pow_data["signature"],
        "target_path": target_path,
    }
    return base64.b64encode(json.dumps(json_data).encode()).decode()


def _pow_pool_take(target_path, auth_token):
    """取走一个未过期的预取结果；没有可用条目时返回 None。

    条目取出即弹出：预取的 challenge 只服务一个请求。让并发请求都拿到同一个
    challenge，只会把校验风险推给上游，不如让它们各自同步求解。
    """
    entry = _POW_POOL.pop((target_path, auth_token), None)
    if entry is None:
        return None
    pow_b64, usable_until = entry
    if usable_until is not None and time.time() >= usable_until:
        logger.debug("丢弃已过期的预取 PoW：%s", target_path)
        return None
    return pow_b64


def _pow_pool_store(target_path, auth_token, pow_b64, pow_data):
    """把解好的 PoW 放进池里，附带一个保守的可用期。"""
    if len(_POW_POOL) >= _POW_POOL_MAX:
        # 池满说明有大量 token 在轮换；淘汰最旧的一条而不是无限增长。
        _POW_POOL.pop(next(iter(_POW_POOL)), None)
    # 固定 TTL 是不掌握真实有效期时的兜底；拿到 expire_at 就取更早的那个，并留
    # 5 秒余量，避免正好卡在边界上被上游以「已过期」拒绝。
    usable_until = time.time() + _POW_PREFETCH_TTL
    expire_epoch = _pow_expire_epoch(pow_data)
    if expire_epoch is not None:
        usable_until = min(usable_until, expire_epoch - 5.0)
    _POW_POOL[(target_path, auth_token)] = (pow_b64, usable_until)


async def _prefetch_pow_once(target_path, auth_token):
    """后台预取一个 PoW；任何失败都只记日志，绝不外抛。"""
    try:
        pow_data = await create_challange_pow(target_path, auth_token)
        answer = await find_pow_answer(pow_data)
        if answer is None:
            logger.debug("预取 PoW 未解出结果：%s", target_path)
            return
        _pow_pool_store(
            target_path,
            auth_token,
            _build_pow_header(pow_data, answer, target_path),
            pow_data,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("预取 PoW 失败：%s", target_path, exc_info=True)


def _schedule_pow_prefetch(target_path, auth_token):
    """安排一次后台预取；同一个 (target_path, token) 不重复排队。"""
    if not _POW_PREFETCH_ENABLED:
        return
    key = (target_path, auth_token)
    if key in _POW_POOL:
        return
    for task in _pow_prefetch_tasks:
        if not task.done() and getattr(task, "_pow_key", None) == key:
            return
    try:
        task = asyncio.create_task(_prefetch_pow_once(target_path, auth_token))
    except RuntimeError:
        # 没有运行中的事件循环（同步测试环境），静默跳过。
        return
    # 持有强引用：create_task 的结果若无人引用，可能在完成前被 GC 回收。
    task._pow_key = key
    _pow_prefetch_tasks.add(task)
    task.add_done_callback(_pow_prefetch_tasks.discard)


async def solve_create_pow(target_path, auth_token):
    cached = _pow_pool_take(target_path, auth_token)
    if cached is not None:
        _schedule_pow_prefetch(target_path, auth_token)
        return cached

    pow_data = await create_challange_pow(target_path, auth_token)
    answer = await find_pow_answer(pow_data)
    if answer is None:
        raise Exception("PoW solve failed")
    header = _build_pow_header(pow_data, answer, target_path)
    # 同步求解说明池是空的，顺手为下一轮补上。
    _schedule_pow_prefetch(target_path, auth_token)
    return header


async def create_new_chat(auth_token):
    headers = get_headers(auth_token)
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    response = await post_with_failover(
        "/api/v0/chat_session/create",
        headers=headers,
        # cookies=cookie,  # Backup WAF fallback
        timeout=aiohttp.ClientTimeout(total=20),
    )
    async with response:
        data = await response.json()
    return data["data"]["biz_data"]["chat_session"]["id"]


def _deepseek_event_fragments(data):
    """Extract response fragments from both direct and batched DeepSeek SSE events."""
    found = []

    def add_fragment(fragment):
        if not isinstance(fragment, dict):
            return
        content = fragment.get("content")
        if content is None:
            return
        fragment_type = str(fragment.get("type") or "RESPONSE").upper()
        if fragment_type in {"RESPONSE", "THINK"}:
            found.append((fragment_type, str(content)))

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        if node.get("p") == "response/fragments" and isinstance(node.get("v"), list):
            for fragment in node["v"]:
                add_fragment(fragment)
            return

        # Some upstream deployments expose the same generation through an
        # OpenAI-compatible delta envelope instead of the native fragment
        # protocol. Keep this fallback narrow so status strings are not emitted
        # as assistant text.
        choices = node.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    delta = choice.get("message")
                if not isinstance(delta, dict):
                    continue
                if delta.get("reasoning_content") is not None:
                    found.append(("THINK", str(delta["reasoning_content"])))
                if delta.get("content") is not None:
                    found.append(("RESPONSE", str(delta["content"])))
            return

        response = node.get("response")
        if isinstance(response, dict) and isinstance(response.get("fragments"), list):
            for fragment in response["fragments"]:
                add_fragment(fragment)
            return

        if "content" in node and str(node.get("type") or "").upper() in {
            "RESPONSE",
            "THINK",
        }:
            add_fragment(node)
            return

        # Large replies are sometimes wrapped in BATCH operations. Walk only
        # structured values so status strings are never mistaken for output.
        for key in ("v", "data", "value"):
            value = node.get(key)
            if isinstance(value, (dict, list)):
                visit(value)

    visit(data)
    return found


def _deepseek_event_finished(data):
    """Return True when a direct or batched event marks generation finished."""
    if isinstance(data, list):
        return any(_deepseek_event_finished(item) for item in data)
    if not isinstance(data, dict):
        return False
    if (
        data.get("p") in {"response/status", "quasi_status"}
        and data.get("v") == "FINISHED"
    ):
        return True
    return any(
        _deepseek_event_finished(data.get(key))
        for key in ("v", "data", "value")
        if isinstance(data.get(key), (dict, list))
    )


def _deepseek_event_search_results(data):
    """Collect structured web-search results carried by a DeepSeek SSE event.

    The upstream publishes one event per search round at the JSON path
    ``response/fragments/<id>/results``; its ``v`` array holds objects with
    ``url``/``title``/``snippet``/``cite_index``/``published_at``. Batched
    envelopes nest the identical payload, so the walker recurses through the
    container keys used by the other event parsers in this module. Items lack a
    URL are dropped here: the Anthropic result schema requires one, and a
    downstream consumer cannot cite an address-less entry.
    """
    found = []

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        path = node.get("p")
        value = node.get("v")
        if (
            isinstance(path, str)
            and path.endswith("/results")
            and isinstance(value, list)
        ):
            for item in value:
                if isinstance(item, dict) and item.get("url"):
                    found.append(item)
            return

        for key in ("v", "data", "value"):
            child = node.get(key)
            if isinstance(child, (dict, list)):
                visit(child)

    visit(data)
    return found


def _deepseek_event_rate_limited(data):
    """Detect rate-limit errors encoded in a successful SSE response."""
    rate_markers = (
        "rate_limit",
        "rate limit",
        "rate-limited",
        "too frequent",
        "too many requests",
    )

    if isinstance(data, list):
        return any(_deepseek_event_rate_limited(item) for item in data)
    if not isinstance(data, dict):
        return False

    finish_reason = data.get("finish_reason")
    if isinstance(finish_reason, str):
        normalized_reason = finish_reason.lower().replace("-", "_")
        if any(
            marker in normalized_reason for marker in ("rate_limit", "too_frequent")
        ):
            return True

    event_type = str(data.get("type") or "").lower()
    nested_error = data.get("error")
    if event_type in {"error", "exception", "rate_limit", "rate_limited"} or isinstance(
        nested_error, dict
    ):
        error_text = " ".join(
            str(data.get(key) or "")
            for key in ("content", "message", "error", "code", "reason")
        ).lower()
        if isinstance(nested_error, dict):
            error_text += (
                " "
                + " ".join(
                    str(nested_error.get(key) or "")
                    for key in ("message", "code", "reason", "type")
                ).lower()
            )
        if event_type in {"rate_limit", "rate_limited"} or any(
            marker in error_text for marker in rate_markers
        ):
            return True

    return any(
        _deepseek_event_rate_limited(data.get(key))
        for key in ("v", "data", "value", "error")
        if isinstance(data.get(key), (dict, list))
    )


class _DeepSeekSSEParser:
    """Accept native SSE framing while tolerating direct JSON responses."""

    def __init__(self):
        self._data_lines = []

    @staticmethod
    def _is_json(value):
        try:
            json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return False
        return True

    def _flush_data(self):
        if not self._data_lines:
            return []
        payload = "\n".join(self._data_lines).strip()
        self._data_lines = []
        return [payload] if payload else []

    def feed(self, raw_line):
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = str(raw_line)
        line = line.lstrip("\ufeff").rstrip("\r\n")
        stripped = line.strip()

        if not stripped:
            return self._flush_data()
        if stripped.startswith(":"):
            return []

        if stripped.startswith("data:"):
            self._data_lines.append(stripped[5:].lstrip())
            # Normal DeepSeek events are one JSON object per data line. Emit
            # them immediately; multiline JSON remains buffered until blank.
            payload = "\n".join(self._data_lines).strip()
            if self._is_json(payload):
                self._data_lines = []
                return [payload]
            return []

        # A few proxies strip the SSE field name from a JSON-only response.
        # Accept only complete JSON here; HTML and diagnostic text are ignored.
        if not self._data_lines and stripped[:1] in "[{" and self._is_json(stripped):
            return [stripped]
        return []

    def flush(self):
        return self._flush_data()


async def _iter_deepseek_sse_payloads(content):
    parser = _DeepSeekSSEParser()
    async for raw_line in content:
        for payload in parser.feed(raw_line):
            yield payload
    for payload in parser.flush():
        yield payload


_EMPTY_SSE_DEFAULT = (
    "Empty response from DeepSeek (no parseable SSE content): "
    "stream ended with no assistant fragments"
)
_EMPTY_SSE_CONTEXT = (
    "Empty response from DeepSeek (prompt may exceed the session context limit)"
)
_CONTEXT_LIMIT_HINT_RE = re.compile(
    r"context|token.?limit|maximum.?context|too.?long|exceed",
    re.IGNORECASE,
)
_SSE_RECENT_LINES = 8


class _SseStreamFinished(Exception):
    pass


def _redact_sse_line(line: str) -> str:
    if len(line) > 500:
        return line[:500] + "..."
    return line


def _remember_sse_line(recent_lines, line: str):
    recent_lines.append(_redact_sse_line(line))
    if len(recent_lines) > _SSE_RECENT_LINES:
        del recent_lines[0]


def _sse_context_limit_hint(recent_lines, parsed_events):
    for obj in parsed_events:
        try:
            blob = json.dumps(obj, ensure_ascii=False)
        except Exception:
            blob = str(obj)
        if _CONTEXT_LIMIT_HINT_RE.search(blob):
            return True
    return any(_CONTEXT_LIMIT_HINT_RE.search(line) for line in recent_lines)


def _scan_recent_sse_errors(parsed_events):
    """Return (http_status, message) if a recent parsed event is an upstream error."""
    for obj in reversed(parsed_events):
        if not isinstance(obj, dict) or obj.get("type") != "error":
            continue
        content = obj.get("content") or obj.get("message") or "Upstream error"
        finish = str(obj.get("finish_reason") or "")
        if finish == "rate_limit_reached" or "rate_limit" in finish.lower():
            return 429, content
        if finish in ("permission_denied", "forbidden"):
            return 403, content
        return 502, content
    return None


def _raise_empty_sse_response(recent_lines, parsed_events, protocol_detail=""):
    """Raise for a stream that ended without assistant output (c181d06).

    `protocol_detail` 由调用方附带流层统计（payload 数、是否见 [DONE]、
    Content-Type）。缺少该统计时，协议层故障与真正的上下文超限无法区分，
    排查只能靠猜测，因此该参数在 send_message 的正常路径上必传。
    """
    err = _scan_recent_sse_errors(parsed_events)
    if err:
        code, msg = err
        logger.warning(
            "DeepSeek SSE error event (not empty stream); last events: %s",
            recent_lines[-_SSE_RECENT_LINES:],
        )
        if code == 429:
            # 限流有专用异常类型，上层据其类型驱动 token 轮换；此处保留上游
            # 原始措辞于日志，异常消息仍用固定文案，避免上游文案漂移影响判定。
            raise DeepSeekRateLimitError()
        raise Exception(f"HTTP {code}: {msg}")
    if _sse_context_limit_hint(recent_lines, parsed_events):
        msg = _EMPTY_SSE_CONTEXT
    else:
        msg = _EMPTY_SSE_DEFAULT
    logger.warning(
        "DeepSeek SSE ended without assistant output; last events: %s",
        recent_lines[-_SSE_RECENT_LINES:],
    )
    raise Exception(msg + protocol_detail)


def _raise_sse_error_event(data, recent_lines):
    """Raise a status-carrying exception for an upstream SSE error event (c181d06).

    状态码经 'HTTP <code>: ' 前缀编码，app._upstream_http_code 据此驱动 token
    轮换与限流标记；上游改动措辞时该前缀由本函数统一产出，不会漂移。
    """
    content = data.get("content") or data.get("message") or "Upstream error"
    finish = str(data.get("finish_reason") or "")
    if finish == "rate_limit_reached" or "rate_limit" in finish.lower():
        logger.warning(
            "DeepSeek SSE error event (not empty stream): %s; last events: %s",
            content,
            recent_lines[-_SSE_RECENT_LINES:],
        )
        # 限流用专用异常，保持与 _deepseek_event_rate_limited 路径同一类型，
        # 上层按类型而非消息文本判定轮换。
        raise DeepSeekRateLimitError()
    if finish in ("permission_denied", "forbidden"):
        raise Exception(f"HTTP 403: {content}")
    raise Exception(f"HTTP 502: {content}")


async def send_message(
    chat_id,
    auth_token,
    message,
    parent_message_id,
    thinking=False,
    search=False,
    file_ids_=None,
    search_sink=None,
):
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    if parent_message_id == 0:
        parent_message_id = None
    file_ids = file_ids_ or []

    headers = get_headers(
        auth_token, await solve_create_pow("/api/v0/chat/completion", auth_token)
    )
    json_data = {
        "chat_session_id": chat_id,
        "parent_message_id": parent_message_id,
        "model_type": DEFAULT_MODEL_TYPE,
        "prompt": message,
        "ref_file_ids": file_ids,
        "thinking_enabled": thinking,
        "search_enabled": search,
        "preempt": False,
        "action": None,
    }

    think_open = False
    got_output = False
    recent_events = []
    # c181d06：错误归类需要两条独立轨迹——parsed_events 供 _scan_recent_sse_errors
    # 读取结构化 error 事件，recent_lines 供 _sse_context_limit_hint 做文本正则匹配。
    # 二者不可互相替代：把 dict 交给正则会在 search() 处抛 TypeError。
    recent_lines = []
    payload_count = 0
    saw_done = False
    resp = await post_with_failover(
        "/api/v0/chat/completion",
        headers=headers,
        json=json_data,
        # cookies=cookie,  # Backup WAF fallback
    )
    async with resp:
        if resp.status != 200:
            error_text = await resp.text()
            logger.warning(
                "DeepSeek completion HTTP %d for chat %s: %s",
                resp.status,
                chat_id,
                error_text[:300],
            )
            raise Exception(f"HTTP {resp.status}: {error_text}")

        async for payload in _iter_deepseek_sse_payloads(resp.content):
            payload_count += 1
            if payload == "[DONE]":
                saw_done = True
                if think_open:
                    yield "\n</think>\n\n"
                    think_open = False
                if got_output:
                    return
                continue
            _remember_sse_line(recent_lines, payload)
            try:
                data = json.loads(payload)
            except Exception as e:
                logger.debug(
                    "Ignoring malformed DeepSeek SSE payload for chat %s: %s",
                    chat_id,
                    e,
                )
                continue
            recent_events.append(data)
            recent_events = recent_events[-5:]

            # c181d06：上游 error 事件携带真实状态码（429/403/502），必须按码
            # 上报；此前一律被归类为「上下文超限」，会误导排查方向。
            if isinstance(data, dict) and data.get("type") == "error":
                _raise_sse_error_event(data, recent_lines)

            # Search results arrive on their own event path and carry no
            # assistant prose. Collecting them here keeps the generator's
            # yielded text limited to RESPONSE/THINK fragments, so callers that
            # need the sources receive them through search_sink instead of
            # parsing them back out of the visible answer.
            if search_sink is not None:
                search_sink.extend(_deepseek_event_search_results(data))

            if _deepseek_event_rate_limited(data):
                logger.warning(
                    "DeepSeek rate limit reported for chat %s; rotating token",
                    chat_id,
                )
                raise DeepSeekRateLimitError()

            for fragment_type, content in _deepseek_event_fragments(data):
                if fragment_type == "THINK":
                    if not think_open:
                        yield "<think>\n"
                        think_open = True
                elif think_open:
                    yield "\n</think>\n\n"
                    think_open = False
                if content:
                    got_output = True
                    yield content

            v = data.get("v")
            if not _deepseek_event_finished(data) and isinstance(v, str) and v:
                got_output = True
                yield v
            if _deepseek_event_finished(data):
                if think_open:
                    yield "\n</think>\n\n"
                if not got_output:
                    content_type = getattr(resp, "headers", {}).get(
                        "Content-Type", "unknown"
                    )
                    logger.warning(
                        "DeepSeek finished without recognized output for chat %s; payloads=%d done=%s content_type=%s recent events=%s",
                        chat_id,
                        payload_count,
                        saw_done,
                        content_type,
                        json.dumps(recent_events, ensure_ascii=False)[:2000],
                    )
                    _raise_empty_sse_response(
                        recent_lines,
                        recent_events,
                        f" (payloads={payload_count} done={saw_done} content_type={content_type})",
                    )
                return
        if think_open:
            yield "\n</think>\n\n"
        if not got_output:
            content_type = getattr(resp, "headers", {}).get("Content-Type", "unknown")
            logger.warning(
                "DeepSeek stream ended without recognized output for chat %s; payloads=%d done=%s content_type=%s recent events=%s",
                chat_id,
                payload_count,
                saw_done,
                content_type,
                json.dumps(recent_events, ensure_ascii=False)[:2000],
            )
            _raise_empty_sse_response(
                recent_lines,
                recent_events,
                f" (payloads={payload_count} done={saw_done} content_type={content_type})",
            )


async def upload_file(file_bytes, file_name, file_content_type, auth_token):
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    session = await get_session()
    file_size = len(file_bytes)
    pow_response = await solve_create_pow("/api/v0/file/upload_file", auth_token)
    boundary = b"----WebKitFormBoundaryTB0pXOQR2RL219Hu"
    safe_name = re.sub(r"[^ -~]", "_", file_name).replace('"', "_") or "file.bin"
    body_parts = [
        b"--" + boundary + b"\r\n",
        f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'.encode(),
        f"Content-Type: {file_content_type}\r\n\r\n".encode(),
        file_bytes,
        b"\r\n--" + boundary + b"--\r\n",
    ]
    reconstructed_body = b"".join(body_parts)
    headers = get_headers(auth_token, pow_response)
    headers.update(
        {
            "content-type": f"multipart/form-data; boundary={boundary.decode('utf-8')}",
            "x-file-size": str(file_size),
        }
    )
    response = await post_with_failover(
        "/api/v0/file/upload_file",
        data=reconstructed_body,
        headers=headers,
        # cookies=cookie,  # Backup WAF fallback
        timeout=aiohttp.ClientTimeout(total=120),
    )
    async with response:
        resp_json = await response.json()
    biz_data = (resp_json.get("data") or {}).get("biz_data") or {}
    file_id = biz_data.get("id")
    if not file_id:
        # Server returned no file id (WAF block / rate limit / malformed response).
        raise Exception(
            f"File upload failed: {json.dumps(resp_json, ensure_ascii=False)[:500]}"
        )
    yield ("uploaded", file_id)
    js_data = biz_data
    status = js_data["status"]
    headers = get_headers(auth_token)
    deadline = time.time() + 300
    while status in ["PENDING", "PARSING"] and time.time() < deadline:
        yield ("uploaded", file_id)
        await asyncio.sleep(0.3)
        async with session.get(
            "https://chat.deepseek.com/api/v0/file/fetch_files?file_ids=" + file_id,
            headers=headers,
            # cookies=cookie,  # Backup WAF fallback
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp_json = await resp.json()
        biz_data = (resp_json.get("data") or {}).get("biz_data") or {}
        files = biz_data.get("files") or []
        if not files:
            # File still being registered server-side; keep waiting.
            continue
        js_data = files[0]
        status = js_data["status"]
    if status == "SUCCESS":
        tp_data = datetime.fromtimestamp(js_data["updated_at"], timezone.utc)
        yield (
            "success",
            {
                "file_id": file_id,
                "openai_timestamp": int(js_data["updated_at"]),
                "size": js_data["file_size"],
                "anthropic_timestamp": tp_data.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
    else:
        yield ("error", file_id)


async def get_file_content(auth_token, file_id):
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    session = await get_session()
    headers = get_headers(auth_token)
    async with session.get(
        "https://chat.deepseek.com/api/v0/file/fetch_files?file_ids=" + file_id,
        headers=headers,
        # cookies=cookie,  # Backup WAF fallback
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        resp_json = await resp.json()
    js_data = resp_json["data"]["biz_data"]["files"][0]
    yield mimetypes.guess_type(js_data["file_name"])[0]
    deadline = time.time() + 60
    while (
        js_data.get("status") in ("PENDING", "PARSING")
        and time.time() < deadline
        and not js_data.get("signed_path")
    ):
        await asyncio.sleep(0.5)
        async with session.get(
            "https://chat.deepseek.com/api/v0/file/fetch_files?file_ids=" + file_id,
            headers=headers,
            # cookies=cookie,  # Backup WAF fallback
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            js_data = (await resp.json())["data"]["biz_data"]["files"][0]
    if not js_data.get("signed_path"):
        return
    file_path = "https://files.deepseeksvc.com/api" + js_data["signed_path"] + "&ty=r"
    async with session.get(file_path) as data:
        async for chunk in data.content.iter_chunked(8192):
            if chunk:
                yield chunk
