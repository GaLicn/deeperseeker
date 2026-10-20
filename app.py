import asyncio
import json
import logging
import mimetypes
import os
import random
import re
import secrets
import time
import unicodedata
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from urllib.parse import quote, urlparse

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.security import HTTPBasic
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from uvicorn.logging import AccessFormatter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# B12：导入期的全局 os.chdir(BASE_DIR) 已移除——所有相对路径改为显式锚定
# BASE_DIR。进程级 CWD 的副作用对内嵌、多 worker、测试与打包都是有害的。

load_dotenv(os.path.join(BASE_DIR, ".env"))

security = HTTPBasic()


from functions import (  # noqa: E402
    CookieGenerationError,
    ReasoningMarkupFilter,
    StreamToolParser,
    TokenLease,
    add_token,
    check_token_status,
    close_session,
    cookie_file_path,
    count_tokens_cached,
    create_new_chat,
    data_dir,
    delete_account,
    delete_sessions_for_chat,
    delete_token,
    find_session,
    get_accounts,
    get_auth_token,
    get_file_content,
    get_file_token,
    get_token,
    get_tokens,
    init_db,
    login_deepseek_account,
    mark_active,
    mark_limited,
    next_parent,
    parse_tools,
    peek_cached_input,
    pick_token,
    record_cached_input,
    record_file,
    reset_cached_input,
    save_account_login,
    save_session,
    send_message,
    upload_file,
)


def _resolve_api_key():
    """B9 (Stage 1 audit): never ship an open relay.

    The API key used to fall back to the publicly documented 'dseeker'
    silently, and the dashboard to admin/admin — an exposed host plus these
    defaults manufactured an open relay (the failure class that killed
    ds2api). Now: an unset key is GENERATED (dsk- + 24 urlsafe chars),
    printed once on boot and persisted next to the database so restarts keep
    the same key. An explicit DEEPSEEKER_API_KEY is honored unchanged.

    Returns (api_key, was_generated)."""
    key = os.getenv("DEEPSEEKER_API_KEY", "").strip()
    if key:
        return key, False
    key_file = os.path.join(data_dir(), "api_key.txt")
    try:
        with open(key_file) as f:
            saved = f.read().strip()
        if saved:
            return saved, True
    except OSError:
        pass
    generated = "dsk-" + secrets.token_urlsafe(24)
    try:
        os.makedirs(os.path.dirname(key_file) or ".", exist_ok=True)
        with open(key_file, "w") as f:
            f.write(generated + "\n")
        os.chmod(key_file, 0o600)
    except OSError:
        pass  # best-effort persistence; the key is printed below regardless
    return generated, True


API_KEY, _api_key_generated = _resolve_api_key()
ADMIN_USER = os.getenv("DEEPSEEKER_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("DEEPSEEKER_ADMIN_PASSWORD", "admin")

from middleware import RealIPMiddleware, RecovererMiddleware, RequestIDMiddleware
from plugin_helper import (
    MAX_SUMMARY_TOKENS,
    build_prompt,
    build_summary_request_prompt,
    context_window_tokens,
    estimate_conversation_tokens,
    extract_and_upload_files,
    generate_signature,
    generate_signature_sync,
    max_output_tokens,
    needs_rollover,
    strip_summary_tags,
)
from usage_stats import (
    RANGE_LABELS,
    aggregate,
    beijing_now,
    build_heatmap,
    cost_of_hour,
    earliest_day,
    hit_rate_series,
    hour_key,
    query_hours,
    record_usage,
    resolve_range,
)

logger = logging.getLogger("uvicorn.error")

# Surface deeperseeker.* logs (cookie generation, prunes, rate-limit events,
# upstream failures) alongside uvicorn's own output, unless already configured.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _log_security_banner():
    """B9 (Stage 1 audit): surface insecure defaults loudly on boot instead of
    quietly shipping an open relay."""
    host = (os.getenv("HOST") or "").strip().lower()
    loopback = host in ("", "127.0.0.1", "localhost", "::1")
    if _api_key_generated:
        logger.warning(
            "SECURITY: DEEPSEEKER_API_KEY was not set — a strong API key was generated for this "
            "instance and saved to api_key.txt next to the database. It is shown ONCE here:\n"
            "  API key: %s",
            API_KEY,
        )
    elif API_KEY.strip().lower() == "dseeker":
        logger.warning(
            "SECURITY: DEEPSEEKER_API_KEY is the publicly documented default 'dseeker'. "
            "Set a strong key before exposing this service beyond loopback."
        )
    if ADMIN_USER == "admin" and ADMIN_PASSWORD == "admin":
        if loopback:
            logger.warning(
                "SECURITY: the dashboard uses the default admin/admin credentials — "
                "set DEEPSEEKER_ADMIN_USER / DEEPSEEKER_ADMIN_PASSWORD."
            )
        else:
            logger.error(
                "SECURITY: the dashboard uses admin/admin while binding a NON-LOOPBACK host (%s). "
                "Anyone who can reach this service owns its token pool — set DEEPSEEKER_ADMIN_USER "
                "and DEEPSEEKER_ADMIN_PASSWORD before exposing it.",
                host or "0.0.0.0",
            )


_log_security_banner()

# The token used for this request, logged as "key: <alias>".
# Must stay a dict: BaseHTTPMiddleware runs the endpoint in a child task, and
# only in place mutation of the same object reaches the access log context.
_key_holder = ContextVar("deeperseeker_key", default=None)
_ALIAS_MAX_LEN = 64
# Strip characters that forge log lines, drive the cursor, or render invisibly.
_ALIAS_STRIP_CATS = frozenset({"Cc", "Cf", "Zl", "Zp"})
_ALIAS_INVISIBLE = frozenset(
    [
        0x034F,
        0x115F,
        0x1160,
        0x17B4,
        0x17B5,
        0x180B,
        0x180C,
        0x180D,
        0x2800,
        0x3164,
        0xFFA0,
    ]
    + list(range(0xFE00, 0xFE10))
    + list(range(0xE0100, 0xE01F0))
)
# Noncharacters are illegal in interchange; some log processors reject them.
_ALIAS_NONCHARACTERS = frozenset(
    list(range(0xFDD0, 0xFDF0))
    + [plane << 16 | low for plane in range(0x11) for low in (0xFFFE, 0xFFFF)]
)


def _sanitize_alias(name):
    if not name:
        return None
    cleaned = "".join(
        ch
        for ch in str(name)
        if ord(ch) not in _ALIAS_INVISIBLE
        and ord(ch) not in _ALIAS_NONCHARACTERS
        and unicodedata.category(ch) not in _ALIAS_STRIP_CATS
    ).strip()
    return cleaned[:_ALIAS_MAX_LEN] or None


def _set_key_name(name):
    holder = _key_holder.get()
    if holder is not None:
        holder["name"] = _sanitize_alias(name)


class KeyAccessFormatter(AccessFormatter):
    def formatMessage(self, record):
        line = super().formatMessage(record)
        # Sanitize again at emit time so no raw value reaches the log line.
        name = _sanitize_alias((_key_holder.get() or {}).get("name"))
        return f"{line} key: {name}" if name else line


def _install_key_access_formatter():
    for handler in logging.getLogger("uvicorn.access").handlers:
        formatter = handler.formatter
        if not isinstance(formatter, AccessFormatter) or isinstance(
            formatter, KeyAccessFormatter
        ):
            continue
        handler.setFormatter(
            KeyAccessFormatter(
                fmt=formatter._fmt,
                datefmt=formatter.datefmt,
                use_colors=getattr(formatter, "use_colors", None),
            )
        )


def count_tok(text):
    # 走内容缓存：同一轮请求里 messages 不变，_messages_text() 的结果会被统计
    # 多次（上下文估算、usage 上报、成本计算），缓存让整次分词只发生一次。
    return count_tokens_cached(text)


async def _record_response_usage(result, model, token_id, scope, req_model=None):
    """从 handle_chat 的返回值里取 usage 并落库。

    返回值不一定是 chat.completion 字典——上游失败时 handle_chat 会直接回
    JSONResponse 之类，此时无用量可记，静默跳过即可。

    @param req_model 客户端请求的原始模型名。落库要用它而不是归一后的
        model，否则统计页的「按模型 / 按服务商」拆分只剩一个值。
    """
    if not isinstance(result, dict):
        return
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return
    await _record_usage(
        model,
        token_id,
        scope,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        req_model=req_model,
    )


async def _record_usage(
    model, token_id, scope, in_tokens, out_tokens, cached_tokens=0, req_model=None
):
    """把一次成功调用的用量累加进小时桶。

    统计是旁路：任何异常都只记日志、绝不冒泡，否则一次落库失败会连带
    把已经生成好的响应打成 500。走 _db 是为了不在事件循环里做同步
    SQLite 写（与其它 store 调用一致）。

    落库的 model 优先取客户端原始模型名（``provider/model`` 或裸名），
    归一后的 ``model`` 只在客户端没给名字时兜底——usage_stats 的
    provider_of / model_name_of 依赖原始名做归因。
    """
    try:
        await _db(
            record_usage,
            model=req_model or model,
            token_id=token_id,
            api_key=scope or "",
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_read_tokens=cached_tokens,
        )
    except Exception:
        logger.exception("usage record failed")


TOKEN_CHECK_INTERVAL = int(os.getenv("DEEPSEEKER_TOKEN_CHECK_INTERVAL", "300"))
_token_check_lock = asyncio.Lock()
_token_check_task = None
_token_check_state = {"running": False, "last_run_at": None, "last_results": []}


async def _run_token_checks():
    """Check every token sequentially and expose a compact dashboard summary."""
    if _token_check_lock.locked():
        return list(_token_check_state["last_results"])
    async with _token_check_lock:
        _token_check_state["running"] = True
        results = []
        try:
            for token in get_tokens():
                results.append(await check_token_status(token["id"]))
            _token_check_state["last_results"] = results
            _token_check_state["last_run_at"] = time.time()
            return results
        except Exception:
            logger.exception("Token health-check batch failed")
            raise
        finally:
            _token_check_state["running"] = False


async def _token_check_loop():
    """Run one delayed check, then repeat every configured five-minute interval."""
    while True:
        await asyncio.sleep(TOKEN_CHECK_INTERVAL)
        try:
            await _run_token_checks()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scheduled token health-check failed")


async def _db(fn, *args, **kwargs):
    """Run a blocking SQLite store helper off the event loop (B5, Stage 1 audit).

    Every sessions/tokens read + write used to run inline: under concurrent
    streams each sqlite3.connect() round-trip and WAL commit stalled the whole
    loop exactly when the proxy was busiest — head-of-line blocking on every
    request (B5). to_thread keeps the store helpers sync (they are shared by
    CLI paths and tests) while the loop never blocks on disk I/O. The
    aiosqlite migration belongs to the store split (Stage 6)."""
    return await asyncio.to_thread(fn, *args, **kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _token_check_task
    # B5 (Stage 1 audit): 数据库初始化是阻塞的 sqlite3 往返，启动时同样不得
    # 占用事件循环，统一经 _db() 转入线程池执行。
    await _db(init_db)
    _install_key_access_formatter()
    _token_check_task = asyncio.create_task(
        _token_check_loop(), name="token-health-check"
    )
    yield
    if _token_check_task:
        _token_check_task.cancel()
        try:
            await _token_check_task
        except asyncio.CancelledError:
            logger.info("Token health-check task stopped")
        except Exception:
            logger.exception("Token health-check task stopped with an error")
        _token_check_task = None
    # Stage 1 次要项：释放共享 aiohttp ClientSession，避免关闭阶段泄漏其
    # connector 套接字。
    await close_session()


app = FastAPI(title="DeeperSeeker", lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount(
    "/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"
)


def _static_version() -> str:
    """用 style.css 的 mtime 作为缓存版本号。

    StaticFiles 不发送 Cache-Control，浏览器会按 Last-Modified 做启发式缓存；
    样式表 URL 固定不变时，即便镜像已重建，旧页面仍会沿用缓存的旧样式。
    在 URL 上挂版本号可强制浏览器重新拉取，无需用户手动强刷。
    """
    try:
        return str(int(os.path.getmtime(os.path.join(BASE_DIR, "static", "style.css"))))
    except OSError:
        return "0"


templates.env.globals["static_version"] = _static_version()


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    _key_holder.set({})
    cl = request.headers.get("content-length", "")
    if cl.isdigit() and int(cl) > 32 * 1024 * 1024:
        return JSONResponse({"error": "Request body too large"}, status_code=413)
    return await call_next(request)


# Stage 0.1 + 0.2 — core rails: fail-closed real-client-IP resolution behind
# proxies (TRUSTED_PROXIES), per-request correlation ids with access logging,
# and a last-resort exception barrier that turns handler crashes into logged
# JSON 500s. Starlette runs the LAST-registered middleware FIRST (outermost),
# so registration order RealIP -> RequestID -> Recoverer yields the execution
# order Recoverer -> RequestID -> RealIP -> limit_body_size -> routes: the
# recoverer sees every error below it, and every response (including its own
# 500s) carries the request id.
app.add_middleware(RealIPMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RecovererMiddleware)


SESSIONS = {}
SESSION_TTL = 7 * 24 * 3600
# B12：SESSIONS（以及下方的 _login_fails）是进程内后台管理状态——重启即重置，
# 且不跨 worker 共享。本服务按设计为单 worker（uvicorn workers=1）；共享存储
# 属于 Stage 6 的 store 拆分。过期条目改在登录时顺带清理，而非驻留至重启。
# Short-lived handoff summaries bridge concurrent requests that observe the
# same pre-rollover signature while the first request is migrating the chat.
ROLLOVER_HANDOFFS = OrderedDict()
ROLLOVER_HANDOFFS_MAX = 1024
# One asyncio.Lock per conversation signature, used to serialize first-time
# session creation. Signatures are unique per message prefix, so without a cap
# this dict grows FOREVER — after many chats it becomes a serious memory leak.
# _take_lock() enforces the cap as a true LRU (the old evict-half loop broke
# out when its whole chunk was locked and then inserted the new key anyway,
# so the cap never held under load — PR #26 review, High). Under pressure it
# registers over cap rather than aliasing chats onto a shared lock — see
# _take_lock for why that trade is required for correctness.
_sig_locks = OrderedDict()
SIG_LOCKS_MAX = int(os.getenv("DEEPSEEKER_MAX_SIG_LOCKS", "4096"))
_login_fails = {"count": 0, "locked_until": 0}

# Stage 0.3 — Per-chat locks (session-collision fix).
# One asyncio.Lock per UPSTREAM chat session id. _sig_locks above only
# serializes first-time session CREATION for one signature; it does nothing
# for two concurrent requests that already share (or race to use) the same
# upstream DeepSeek chat: both would send with the same parent_message_id,
# fork the upstream conversation, and the last save_session() would corrupt
# the stored parent counter. The chat lock serializes the send -> save
# critical section per upstream session, so same-chat requests queue instead
# of colliding (different chats remain fully parallel). Keyed by upstream
# session id rather than signature, because each completed turn derives a new
# signature while the upstream chat stays the same. The registry is capped by
# _take_lock() (true LRU; see its docstring for the PR #26 review fix that
# replaced the old evict-half loop, which could not actually cap).
_chat_locks = OrderedDict()
CHAT_LOCKS_MAX = int(os.getenv("DEEPSEEKER_MAX_CHAT_LOCKS", "4096"))


def _take_lock(label, registry, key, max_entries):
    """Return the lock for `key` from an LRU-capped registry, creating it on
    first use.

    Review fix (PR #26, High): the previous eviction loop shared by
    _sig_locks/_chat_locks scanned the oldest MAX//2+1 entries, broke out when
    ALL of them were locked, and then setdefault'ed the new key anyway — so
    under sustained load with many live chats the dict grew without bound; the
    "memory-leak guard" was the leak. Semantics now:
      - a hit moves the key to the most-recently-used end;
      - over cap, the least-recently-used UNLOCKED entry is evicted (a locked
        entry is never evicted — that would fork a chat's critical section;
        worst case stays one benign re-creation race, as before);
      - if EVERY entry is held, the new key is STILL registered, growing the
        registry over cap. Addendum: an earlier draft returned a shared
        fallback lock WITHOUT registering the key — that traded the memory
        bound for a correctness one. If pressure dropped while that request
        was still in flight, the next request for the same chat found room,
        created a fresh per-key lock, and two live holders sat in one chat's
        critical section — reopening the exact parent_message_id race Stage
        0.3 exists to close, precisely under the load the branch was designed
        for. Registering over cap keeps same chat -> same lock object
        unconditionally. The cost stays bounded: over-cap entries appear only
        when every existing lock is held (in-flight pressure), so depth
        tracks request concurrency during pressure windows — never total chat
        history — and drained entries are inert until recycled by LRU churn.
        Each over-cap registration logs a warning with the live depth: the
        ops signal for sustained pressure (raise the cap if it fires
        continuously).
    """
    lock = registry.get(key)
    if lock is not None:
        registry.move_to_end(key)
        return lock
    if len(registry) >= max_entries:
        for old_key, held in registry.items():
            if not held.locked():
                del registry[old_key]
                break
        # nothing unlocked? fall through and register over cap — refusing to
        # register (or aliasing to a shared lock) would break same-chat
        # identity, which is the invariant this registry exists to guarantee
    lock = asyncio.Lock()
    registry[key] = lock
    if len(registry) > max_entries:
        logger.warning(
            "%s lock registry over cap: %d entries (cap %d) — every existing lock is "
            "held; depth tracks in-flight requests, not chat history",
            label,
            len(registry),
            max_entries,
        )
    return lock


def _chat_lock(session_id):
    return _take_lock("chat", _chat_locks, str(session_id), CHAT_LOCKS_MAX)


class _OwnedChatLock:
    """Ownership token for one acquisition of a per-chat lock.

    Review fix (PR #26, Medium): release sites used `if lock.locked():
    lock.release()`, but locked() reports whether ANYONE holds the lock —
    after an early release and another request's acquisition, a late release
    dropped the OTHER request's lock and put two requests inside the critical
    section the lock exists to prevent. A token tracks only its own
    acquisition: release() is idempotent and can never release a stranger's
    hold, which also makes ownership transfer to a stream generator and
    release-before-retry safe by construction. `owned` says whether THIS
    token still holds the lock (the question locked() could not answer)."""

    __slots__ = ("lock", "_owned")

    def __init__(self, lock):
        self.lock = lock
        self._owned = False

    async def acquire(self):
        await self.lock.acquire()
        self._owned = True

    def release(self):
        if not self._owned:
            return
        self._owned = False
        # defensive: releasing an already-released lock must never kill a request
        with suppress(RuntimeError):
            self.lock.release()

    @property
    def owned(self):
        return self._owned


async def _own_chat_lock(session_id):
    """Acquire this chat's lock and return its ownership token."""
    token = _OwnedChatLock(_chat_lock(session_id))
    await token.acquire()
    return token


def _release_chat_lock_stream(gen, owner):
    """Wrap a streaming generator so the per-chat lock stays held until the
    stream completes (or the client aborts), then is released exactly once.

    The lock is acquired in handle_chat before the upstream POST; for streaming
    responses the final save_session() happens inside the stream generator, so
    ownership of the lock must transfer from handle_chat to the generator —
    releasing any earlier would reopen the parent_message_id race the lock
    exists to prevent. `owner` is a _OwnedChatLock token: its release() drops
    ONLY this holder's acquisition, never a stranger's (PR #26 review, Medium)."""

    async def _wrapped():
        try:
            async for chunk in gen:
                yield chunk
        finally:
            owner.release()

    return _wrapped()


def get_current_admin(request: Request):
    sid = request.cookies.get("session_id")
    if not sid or sid not in SESSIONS or time.time() - SESSIONS[sid] > SESSION_TTL:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    SESSIONS[sid] = time.time()
    origin = request.headers.get("origin", "")
    if origin:
        parsed = urlparse(origin).netloc
        if parsed and parsed != request.headers.get("host", ""):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return "admin"


def get_api_key(request: Request):
    auth = request.headers.get("authorization", "")
    api_key_header = request.headers.get("x-api-key", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    elif auth:
        return auth
    return api_key_header


def check_key(request: Request):
    key = get_api_key(request)
    return secrets.compare_digest(key.encode("utf-8"), API_KEY.encode("utf-8"))


def _upstream_http_code(exc):
    m = re.match(r"HTTP (\d{3}):", str(exc))
    return int(m.group(1)) if m else None


def _api_error_response(e, is_anthropic=False):
    code = _upstream_http_code(e) or 502
    if isinstance(e, CookieGenerationError):
        code = 503  # WAF cookies cannot be produced right now — upstream unreachable, not a client error
    if code < 400 or code > 599:
        code = 502
    if is_anthropic:
        payload = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)[:500]},
        }
    else:
        err_type = "rate_limit_error" if code == 429 else "api_error"
        payload = {"error": {"message": str(e)[:500], "type": err_type, "code": code}}
    return JSONResponse(payload, status_code=code)


def _referenced_file_ids(messages):
    """扫描会话引用的 file_id（经 /v1/files 上传，或 Anthropic 的 file source）。

    纯扫描、无 I/O，供「优先使用文件属主 token」与「识别异主引用」两处使用（B4）。
    """
    ids = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for part in c:
            if not isinstance(part, dict):
                continue
            if (
                part.get("type") == "file"
                and isinstance(part.get("file"), dict)
                and part["file"].get("file_id")
            ):
                ids.append(part["file"]["file_id"])
            elif (
                part.get("type") in ("document", "image")
                and isinstance(part.get("source"), dict)
                and part["source"].get("type") == "file"
                and part["source"].get("file_id")
            ):
                ids.append(part["source"]["file_id"])
    return ids


async def _copy_file_to_token(file_id, fetch_token, target_token):
    """以 fetch_token 取回文件字节，再上传到 target_token 所属账号（B4 改挂）。

    成功返回新的 file_id，失败返回 None。
    """
    mime = None
    chunks = []
    size = 0
    try:
        gen = get_file_content(fetch_token, file_id)
        mime = await gen.__anext__()  # 首个 yield 是 mime type
        async for chunk in gen:
            size += len(chunk)
            if size > 25 * 1024 * 1024:
                logger.warning(
                    "File ownership: re-home of %s aborted (over 25 MB)", file_id
                )
                return None
            chunks.append(chunk)
    except StopAsyncIteration:
        return None
    except Exception:
        logger.exception("File ownership: fetch of %s failed during re-home", file_id)
        return None
    ext = (mimetypes.guess_extension(mime) if mime else None) or ".bin"
    filename = f"rehomed_{file_id}{ext}"
    async for upload_status, data in upload_file(
        b"".join(chunks), filename, mime or "application/octet-stream", target_token
    ):
        if upload_status == "success":
            return data["file_id"]
    return None


async def _rehome_foreign_files(file_ids, token_id, tok):
    """返回 token_id 所属账号可用的 file_id 列表（B4）。

    上传会钉在当时的 token 上，而上游文件按账号隔离，因此异主引用在聊天时会
    被上游 404。这里把异主 id 复制到本会话的 token 上（用属主取回、用本会话
    token 上传）。未知（历史遗留、未登记）的 id 原样透传——对它们没有比旧行为
    更好的处理方式。
    """
    out = []
    for fid in file_ids:
        owner = await _db(get_file_token, fid)
        if owner is None or owner == token_id:
            out.append(fid)
            continue
        owner_tok = await _db(get_token, owner)
        fetch_token = owner_tok["token"] if owner_tok else tok["token"]
        new_id = await _copy_file_to_token(fid, fetch_token, tok["token"])
        if new_id:
            await _db(record_file, new_id, token_id)
            logger.info(
                "File ownership: re-uploaded file %s (token #%s) onto token #%s as %s",
                fid,
                owner,
                token_id,
                new_id,
            )
            out.append(new_id)
        else:
            # 尽力而为：保留原引用，而不是直接丢掉。
            out.append(fid)
    return out


def _replay_stream(gen, first):
    async def _wrapped():
        if first is not None:
            yield first
        async for chunk in gen:
            yield chunk

    return _wrapped()


# B10（Stage 1 审计）：通用上游故障（空 SSE、瞬时 5xx、被污染的会话）的有界重试
# 预算——最多 MAX_UPSTREAM_ATTEMPTS 次尝试，每次优先换一个刚失败过的 token，
# 尝试之间加入抖动退避。预算耗尽时按 502 类返回上游错误（已脱敏、已截断）。
MAX_UPSTREAM_ATTEMPTS = max(1, int(os.getenv("DEEPSEEKER_MAX_UPSTREAM_ATTEMPTS", "3")))


async def _preflight_stream(gen):
    """Consume the first chunk eagerly so upstream errors surface before streaming starts."""
    try:
        first = await gen.__anext__()
    except StopAsyncIteration:
        first = None
    return _replay_stream(gen, first)


async def handle_chat(
    messages,
    model,
    thinking=False,
    search=False,
    stream=False,
    tools=None,
    is_anthropic=False,
    req_model=None,
    scope="",
    search_sink=None,
    _attempt=0,
    _exclude_token=None,
    is_responses=False,
    response_opts=None,
):
    auth_token = await _db(get_auth_token)
    if not auth_token:
        return JSONResponse(
            {"error": "No auth token. Add via dashboard."}, status_code=401
        )

    # 廉价预筛：精确的 rollover 判定由 build_prompt 内的 needs_rollover() 用
    # 真实 token 数完成，这里算出来的值只喂给下面那行日志。因此在字符数离阈值
    # 还很远时跳过整段分词——30 轮工具调用循环里每轮能省下约 90ms。
    # 预筛偏保守只会少打一行日志，不影响任何决策（字符数少于阈值时，token 数
    # 不可能反超：BPE 词元至少覆盖一个字符）。
    rollover_limit = context_window_tokens()
    rollover_watch = int(rollover_limit * 0.8)
    if len(_messages_text(messages)) >= rollover_watch:
        estimated_context = estimate_conversation_tokens(messages)
        if estimated_context >= rollover_watch:
            logger.info(
                "Context check: estimated=%d limit=%d messages=%d (rollover=%s)",
                estimated_context,
                rollover_limit,
                len(messages),
                estimated_context > rollover_limit,
            )

    # Responses 协议要求 previous_response_id 续接上一轮，且首轮不得按消息内容
    # 派生签名：客户端可能只发增量输入，内容签名无法命中已存会话，会导致每轮
    # 都新开一条上游 chat。因此首轮直接以响应 id 作签名，续接轮再回退查找。
    has_prior_turn = any(
        m.get("role") in ("assistant", "tool") for m in messages
    ) or bool(
        isinstance(response_opts, dict) and response_opts.get("previous_response_id")
    )
    if (
        is_responses
        and not has_prior_turn
        and isinstance(response_opts, dict)
        and response_opts.get("_resp_id")
    ):
        sig = str(response_opts["_resp_id"])
    else:
        sig = await generate_signature(messages, model, scope)
    sess = await _db(find_session, sig)
    if (
        not sess
        and is_responses
        and isinstance(response_opts, dict)
        and response_opts.get("previous_response_id")
    ):
        sess = await _db(find_session, str(response_opts["previous_response_id"]))
    # A concurrent request may already have migrated this exact signature.
    # Reuse its handoff summary when sending the first turn on the new chat.
    rollover_summary = ROLLOVER_HANDOFFS.get(sig)
    # B4：本会话引用的 file_id（由新建会话的分支填充），供后续轮次判断是否需要
    # 把异主引用改挂到本会话的 token 上。
    ref_ids = []

    if sess:
        token_id = sess["token_id"]
        session_id = sess["session_id"]
        parent_message_id = sess["parent_message_id"]
        tok = await _db(get_token, token_id)
        # Proactively migrate a live chat before the upstream memory ceiling is
        # hit. Previously rollover only ran on new-chat/token-rotation paths,
        # so long-lived sessions could fail first and lose their parent chain.
        if (
            tok
            and tok.get("status") != "RATE_LIMITED"
            and parent_message_id != 0
            and needs_rollover(messages)
        ):
            old_session_id = session_id
            old_lock = await _own_chat_lock(old_session_id)
            try:
                # Another request may have completed this signature while we
                # waited for the old-chat lock; reuse its mapping instead of
                # creating a second fresh upstream conversation.
                current = await _db(find_session, sig)
                if current and current["session_id"] != old_session_id:
                    session_id = current["session_id"]
                    parent_message_id = current["parent_message_id"]
                    rollover_summary = ROLLOVER_HANDOFFS.get(sig)
                else:
                    scratch_chat = await create_new_chat(tok["token"])
                    summary_gen = send_message(
                        scratch_chat,
                        tok["token"],
                        build_summary_request_prompt(messages),
                        0,
                        False,
                        False,
                        [],
                    )
                    rollover_summary = strip_summary_tags(
                        await collect_response(summary_gen)
                    )[: MAX_SUMMARY_TOKENS * 4]
                    new_session_id = await create_new_chat(tok["token"])
                    # Replace old mappings only after summary and fresh chat
                    # creation both succeed.
                    await _db(delete_sessions_for_chat, token_id, old_session_id)
                    session_id = new_session_id
                    parent_message_id = 0
                    await _db(
                        save_session, sig, token_id, session_id, parent_message_id
                    )
                    ROLLOVER_HANDOFFS[sig] = rollover_summary
                    ROLLOVER_HANDOFFS.move_to_end(sig)
                    while len(ROLLOVER_HANDOFFS) > ROLLOVER_HANDOFFS_MAX:
                        ROLLOVER_HANDOFFS.popitem(last=False)
                    logger.info(
                        "Context rollover: migrated live chat %s -> %s (summary ~%d tokens)",
                        old_session_id,
                        new_session_id,
                        count_tok(rollover_summary),
                    )
            except Exception:
                # A failed preflight must not destroy the working upstream chat;
                # continue on the existing session and let normal retry logic
                # handle an actual upstream failure.
                rollover_summary = None
                session_id = old_session_id
                logger.exception(
                    "Live context rollover failed; keeping chat %s", old_session_id
                )
            finally:
                old_lock.release()
        if not tok or tok["status"] == "RATE_LIMITED":
            new_token_id = await _db(pick_token)
            if new_token_id and (not tok or new_token_id != token_id):
                new_tok = await _db(get_token, new_token_id)
                if new_tok:
                    _set_key_name(new_tok.get("alias"))
                    # 轮换后的请求由新账号承载，在途占用须登记到新 token 上；
                    # 旧 token 的计数由触发轮换的那一帧自行释放。
                    rot_lease = TokenLease(new_token_id)
                    # Stage 0.3: rotation re-creates the upstream chat; hold the
                    # new chat's lock across its send -> save section so a
                    # concurrent same-signature request cannot race the swap.
                    rot_owner = None
                    try:
                        await _db(delete_sessions_for_chat, token_id, session_id)
                        new_session_id = await create_new_chat(new_tok["token"])
                        rot_owner = await _own_chat_lock(new_session_id)
                        if needs_rollover(messages):
                            scratch_chat = await create_new_chat(new_tok["token"])
                            summary_gen = send_message(
                                scratch_chat,
                                new_tok["token"],
                                build_summary_request_prompt(messages),
                                0,
                                False,
                                False,
                                [],
                            )
                            rollover_summary = strip_summary_tags(
                                await collect_response(summary_gen)
                            )[: MAX_SUMMARY_TOKENS * 4]
                        prompt = await build_prompt(
                            messages,
                            tools or [],
                            model,
                            is_first_message=True,
                            rollover_summary=rollover_summary,
                        )

                        file_ids = await extract_and_upload_files(
                            messages, new_tok["token"]
                        )
                        # B4：轮换后的账号承载这些上传，须登记归属，否则后续轮次
                        # 引用时无从判断属主。
                        for fid in file_ids:
                            await _db(record_file, fid, new_token_id)
                        gen = send_message(
                            new_session_id,
                            new_tok["token"],
                            prompt,
                            0,
                            thinking,
                            search,
                            file_ids,
                        )
                        gen = await _preflight_stream(gen)
                    except Exception as e:
                        if rot_owner is not None:
                            rot_owner.release()
                        # 上游请求未能建立，新 token 的在途占用须立即注销，
                        # 否则该账号会永久显得繁忙，扭曲后续的负载均衡。
                        rot_lease.release()
                        logger.exception(
                            "Token-rotation recovery failed (chat %s): %s",
                            session_id,
                            e,
                        )
                        if _attempt + 1 >= MAX_UPSTREAM_ATTEMPTS:
                            return _api_error_response(e, is_anthropic)
                        return await handle_chat(
                            messages,
                            model,
                            thinking,
                            search,
                            stream,
                            tools,
                            is_anthropic,
                            req_model,
                            scope,
                            search_sink=search_sink,
                            _attempt=_attempt + 1,
                            is_responses=is_responses,
                            response_opts=response_opts,
                        )
                    if stream:
                        gen = _release_chat_lock_stream(gen, rot_owner)
                        # 与主路径一致：流式响应把在途占用的所有权交给生成器，
                        # 由它在 finally 内释放（客户端中断同样触发）。
                        if is_anthropic:
                            return StreamingResponse(
                                stream_anthropic_response(
                                    gen,
                                    model,
                                    messages,
                                    new_token_id,
                                    new_session_id,
                                    sig,
                                    tools,
                                    req_model,
                                    0,
                                    scope,
                                    lease=rot_lease,
                                ),
                                media_type="text/event-stream",
                                background=BackgroundTask(rot_lease.release),
                            )
                        if is_responses:
                            return StreamingResponse(
                                stream_responses_response(
                                    gen,
                                    model,
                                    messages,
                                    new_token_id,
                                    new_session_id,
                                    sig,
                                    tools,
                                    req_model,
                                    0,
                                    scope,
                                    response_opts,
                                    lease=rot_lease,
                                ),
                                media_type="text/event-stream",
                                background=BackgroundTask(rot_lease.release),
                            )
                        return StreamingResponse(
                            stream_response(
                                gen,
                                model,
                                messages,
                                new_token_id,
                                new_session_id,
                                sig,
                                tools,
                                0,
                                scope,
                                lease=rot_lease,
                            ),
                            media_type="text/event-stream",
                            background=BackgroundTask(rot_lease.release),
                        )
                    else:
                        try:
                            resp_text = await collect_response(gen)
                        except Exception as e:
                            logger.exception(
                                "Upstream failed during token-rotation request: %s", e
                            )
                            if rot_owner is not None:
                                rot_owner.release()
                            rot_lease.release()
                            if _attempt + 1 >= MAX_UPSTREAM_ATTEMPTS:
                                return _api_error_response(e, is_anthropic)
                            return await handle_chat(
                                messages,
                                model,
                                thinking,
                                search,
                                stream,
                                tools,
                                is_anthropic,
                                req_model,
                                scope,
                                search_sink=search_sink,
                                _attempt=_attempt + 1,
                                is_responses=is_responses,
                                response_opts=response_opts,
                            )
                        await _db(mark_active, new_token_id)

                        parsed_tools, clean_text = parse_tools(resp_text)
                        clean_text = re.sub(
                            r"<think>.*?</think>", "", clean_text, flags=re.DOTALL
                        ).strip()
                        clean_text = re.sub(
                            r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>",
                            "",
                            clean_text,
                            flags=re.IGNORECASE,
                        ).strip()
                        next_messages = messages.copy()
                        ast_msg = {"role": "assistant"}
                        if parsed_tools:
                            ast_msg["tool_calls"] = parsed_tools
                        else:
                            ast_msg["content"] = clean_text
                        next_messages.append(ast_msg)
                        next_sig = await generate_signature(next_messages, model, scope)

                        await _db(
                            save_session,
                            sig,
                            new_token_id,
                            new_session_id,
                            next_parent(0),
                        )
                        await _db(
                            save_session,
                            next_sig,
                            new_token_id,
                            new_session_id,
                            next_parent(0),
                        )
                        if (
                            is_responses
                            and isinstance(response_opts, dict)
                            and response_opts.get("_resp_id")
                        ):
                            r_id = response_opts["_resp_id"]
                            await _db(
                                save_session,
                                r_id,
                                new_token_id,
                                new_session_id,
                                next_parent(0),
                            )
                            _store_response_history(r_id, next_messages)
                        if rot_owner is not None:
                            rot_owner.release()
                        rot_lease.release()
                        in_tokens = count_tok(_messages_text(messages))
                        _remember_prompt_tokens(next_sig, in_tokens)
                        result = format_response(
                            resp_text,
                            model,
                            messages,
                            tools,
                            cached_tokens=_cached_prompt_tokens(sig, in_tokens),
                            prompt_tokens=in_tokens,
                        )
                        # 轮换后的非流式分支同样要落库：这次调用跑在 new_token_id
                        # 上，若漏记，号池里换过账号的请求就会整段消失在统计里。
                        await _record_response_usage(
                            result, model, new_token_id, scope, req_model
                        )
                        return result
            return JSONResponse(
                {
                    "error": {
                        "message": "No active tokens available (all rate limited). Try again later.",
                        "type": "rate_limit_error",
                    }
                },
                status_code=429,
            )
    else:
        create_lock = _take_lock("sig", _sig_locks, sig, SIG_LOCKS_MAX)
        async with create_lock:
            sess = await _db(find_session, sig)
            if not sess:
                # B10：重试预算把刚失败的 token id 传下来，pick_token() 在还有
                # 其它可用账号时会避开它。
                token_id = await _db(
                    pick_token,
                    exclude=({_exclude_token} if _exclude_token is not None else None),
                )
                if not token_id:
                    return JSONResponse(
                        {"error": "No tokens available"}, status_code=503
                    )
                # B4：上游文件按账号隔离。若本会话首轮引用的文件只有唯一已知
                # 属主，就让聊天跑在该 token 上——调度器若选中别的账号，上游会
                # 直接回 “file not found”。
                ref_ids = _referenced_file_ids(messages)
                if ref_ids:
                    owners = {await _db(get_file_token, fid) for fid in ref_ids}
                    owners.discard(None)
                    if len(owners) == 1:
                        owner_id = owners.pop()
                        if owner_id != token_id:
                            owner_tok = await _db(get_token, owner_id)
                            if owner_tok and owner_tok["status"] == "ACTIVE":
                                logger.info(
                                    "File ownership: chat references file(s) pinned "
                                    "to token #%s; using it",
                                    owner_id,
                                )
                                token_id = owner_id
                tok = await _db(get_token, token_id)
                if not tok:
                    return JSONResponse({"error": "Token not found"}, status_code=503)

                # Accumulated-context rollover (issue #22): when the conversation is
                # nearing the observed remembered-context limit, first obtain a
                # model-generated handoff summary via a scratch chat (the request
                # itself is near the context limit, so it must not be sent into any
                # chat that has to absorb it), then start the real chat seeded with
                # that summary via build_prompt(rollover_summary=...). A single large
                # first exchange is untouched (the first-message path accepts ~1M
                # tokens); this only fires for accumulated session context.
                if needs_rollover(messages):
                    scratch_chat = await create_new_chat(tok["token"])
                    summary_gen = send_message(
                        scratch_chat,
                        tok["token"],
                        build_summary_request_prompt(messages),
                        0,
                        False,
                        False,
                        [],
                    )
                    summary = strip_summary_tags(await collect_response(summary_gen))
                    rollover_summary = summary[
                        : MAX_SUMMARY_TOKENS * 4
                    ]  # ~4 chars/token cap
                    logger.info(
                        "Context rollover: handoff summary of ~%d tokens prepared in scratch chat %s",
                        count_tok(rollover_summary),
                        scratch_chat,
                    )

                session_id = await create_new_chat(tok["token"])
                await _db(save_session, sig, token_id, session_id, 0)
                if rollover_summary:
                    ROLLOVER_HANDOFFS[sig] = rollover_summary
                    ROLLOVER_HANDOFFS.move_to_end(sig)
                    while len(ROLLOVER_HANDOFFS) > ROLLOVER_HANDOFFS_MAX:
                        ROLLOVER_HANDOFFS.popitem(last=False)
                parent_message_id = 0
            else:
                token_id = sess["token_id"]
                session_id = sess["session_id"]
                parent_message_id = sess["parent_message_id"]

    tok = await _db(get_token, token_id)
    if not tok:
        return JSONResponse({"error": "Token expired"}, status_code=503)
    _set_key_name(tok.get("alias"))

    # B1（Stage 1 审计）：累积上下文的滚动此前在获取 per-chat 锁之前执行，仅
    # 首次创建受 sig 锁保护——同签名的两个并发请求可能都判定需要 rollover、都
    # 删除会话行、并各自创建新的上游 chat（后写入的 save_session() 获胜，落败
    # 的那个 chat 泄漏，parent id 也随之分叉）。现在整个 rollover 判定都在当前
    # chat 的锁内完成，且在持有锁后重新读取已存会话状态：本帧等待期间，同签名
    # 的并发请求可能已经完成滚动（或推进了 parent_message_id）。
    #
    # 本 fork 已在「会话已存在」分支内用 old chat 锁串行化滚动（见 handle_chat
    # 入口处的会话迁移段），因此这里只保留 B1 的二次校验职责：取得 session_id
    # 的锁后重读映射，仅在映射未变且 parent_message_id 非零时才执行滚动。
    lock_owner = await _own_chat_lock(session_id)
    # 登记该 token 的在途占用，供 pick_token() 的负载均衡读取。非流式在
    # finally 释放；流式连同锁一起把所有权交给生成器。
    lease = TokenLease(token_id)
    lock_transferred = False
    lease_transferred = False
    try:
        fresh = await _db(find_session, sig)
        if fresh:
            if fresh["session_id"] != session_id:
                # The chat moved under us (a concurrent request already rolled
                # it over). Follow it and hold the NEW chat's lock instead.
                lock_owner.release()
                token_id = fresh["token_id"]
                session_id = fresh["session_id"]
                parent_message_id = fresh["parent_message_id"]
                tok = await _db(get_token, token_id)
                if not tok:
                    return JSONResponse({"error": "Token expired"}, status_code=503)
                _set_key_name(tok.get("alias"))
                lock_owner = await _own_chat_lock(session_id)
                # B1 residual (Stage 1 review, finding 3): the parent adopted
                # above was read while we still held the OLD chat's lock; the
                # awaits since then (get_token, acquiring the NEW chat's lock)
                # gave a same-signature request a window to complete a turn on
                # this chat — sending with that stale parent would fork it,
                # the exact bug class B1 closes. Re-read under the fresh lock,
                # exactly like the rollover branch below.
                fresh = await _db(find_session, sig)
                if fresh and fresh["session_id"] == session_id:
                    parent_message_id = fresh["parent_message_id"]
            else:
                # Same chat: adopt the stored parent so a request that was
                # queued behind a completed turn never re-sends a stale
                # parent_message_id (which would fork the upstream exchange).
                parent_message_id = fresh["parent_message_id"]

        if parent_message_id != 0 and needs_rollover(messages):
            logger.info(
                "Context rollover: accumulated context over limit; purging session mappings for chat %s",
                session_id,
            )
            await _db(delete_sessions_for_chat, token_id, session_id)
            scratch_chat = await create_new_chat(tok["token"])
            summary_gen = send_message(
                scratch_chat,
                tok["token"],
                build_summary_request_prompt(messages),
                0,
                False,
                False,
                [],
            )
            rollover_summary = strip_summary_tags(await collect_response(summary_gen))[
                : MAX_SUMMARY_TOKENS * 4
            ]
            session_id = await create_new_chat(tok["token"])
            await _db(save_session, sig, token_id, session_id, 0)
            parent_message_id = 0
            # The rollover itself was serialized under the OLD chat's lock;
            # the send -> save section below must hold the FRESH chat's lock.
            # Queued same-signature requests re-derive the new mapping from
            # the DB via the re-read above, so nobody double-rolls-over.
            lock_owner.release()
            lock_owner = await _own_chat_lock(session_id)
            # A request that read the fresh mapping in the window between our
            # save and our acquisition may have already appended to this
            # chat; adopt the stored parent so we never fork it.
            fresh = await _db(find_session, sig)
            if fresh and fresh["session_id"] == session_id:
                parent_message_id = fresh["parent_message_id"]

        is_first = parent_message_id == 0
        # Stage 0.3: the lock is held across the whole send -> save critical
        # section; for streams, ownership transfers to the response generator
        # via _release_chat_lock_stream (the final save_session happens there).
        file_ids = await extract_and_upload_files(
            messages, tok["token"], last_user_only=not is_first
        )
        # B4：钉在别的账号上的引用在本会话会被上游 404，须复制到本会话的
        # token 上。本次新上传的（以及改挂产生的副本）登记归属；已有属主的
        # 引用保持其原属主不变（先到者胜）。
        if file_ids:
            ref_set = set(ref_ids)
            file_ids = await _rehome_foreign_files(file_ids, token_id, tok)
            for fid in file_ids:
                if fid not in ref_set:
                    await _db(record_file, fid, token_id)
        # Rollover rebuilds the prompt from a summary instead of the accumulated
        # history, so the previous turn's prompt is no longer a prefix of this
        # one and any remembered cache size would overstate the real hit.
        if rollover_summary:
            reset_cached_input(sig)
        prompt = await build_prompt(
            messages, tools or [], model, is_first, rollover_summary=rollover_summary
        )

        gen = send_message(
            session_id,
            tok["token"],
            prompt,
            parent_message_id,
            thinking,
            search,
            file_ids,
            search_sink=search_sink,
        )
        gen = await _preflight_stream(gen)
        if stream:
            gen = _release_chat_lock_stream(gen, lock_owner)
            lock_transferred = True
            # 流式响应把 token 在途占用的所有权一并交给生成器，由它在 finally
            # 内释放（客户端中断同样触发）；此处不得再自行释放。
            lease_transferred = True
            if is_anthropic:
                return StreamingResponse(
                    stream_anthropic_response(
                        gen,
                        model,
                        messages,
                        token_id,
                        session_id,
                        sig,
                        tools,
                        req_model,
                        parent_message_id,
                        scope,
                        lease=lease,
                    ),
                    media_type="text/event-stream",
                    background=BackgroundTask(lease.release),
                )
            if is_responses:
                return StreamingResponse(
                    stream_responses_response(
                        gen,
                        model,
                        messages,
                        token_id,
                        session_id,
                        sig,
                        tools,
                        req_model,
                        parent_message_id,
                        scope,
                        response_opts,
                        lease=lease,
                    ),
                    media_type="text/event-stream",
                    background=BackgroundTask(lease.release),
                )
            return StreamingResponse(
                stream_response(
                    gen,
                    model,
                    messages,
                    token_id,
                    session_id,
                    sig,
                    tools,
                    req_model,
                    parent_message_id,
                    scope,
                    lease=lease,
                ),
                media_type="text/event-stream",
                background=BackgroundTask(lease.release),
            )
        else:
            resp_text = await collect_response(gen)
            await _db(mark_active, token_id)

            parsed_tools, clean_text = parse_tools(resp_text)
            clean_text = re.sub(
                r"<think>.*?</think>", "", clean_text, flags=re.DOTALL
            ).strip()
            clean_text = re.sub(
                r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>",
                "",
                clean_text,
                flags=re.IGNORECASE,
            ).strip()
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = await generate_signature(next_messages, model, scope)

            await _db(
                save_session, sig, token_id, session_id, next_parent(parent_message_id)
            )
            await _db(
                save_session,
                next_sig,
                token_id,
                session_id,
                next_parent(parent_message_id),
            )
            if (
                is_responses
                and isinstance(response_opts, dict)
                and response_opts.get("_resp_id")
            ):
                r_id = response_opts["_resp_id"]
                await _db(
                    save_session,
                    r_id,
                    token_id,
                    session_id,
                    next_parent(parent_message_id),
                )
                _store_response_history(r_id, next_messages)
            in_tokens = count_tok(_messages_text(messages))
            _remember_prompt_tokens(next_sig, in_tokens)
            result = format_response(
                resp_text,
                model,
                messages,
                tools,
                cached_tokens=_cached_prompt_tokens(sig, in_tokens),
                prompt_tokens=in_tokens,
            )
            await _record_response_usage(result, model, token_id, scope, req_model)
            return result
    except asyncio.CancelledError:
        # B2（Stage 1 审计）：客户端在请求中途断开时，本次交换从未在上游完成，
        # 已保存的 parent_message_id 会使下一轮从错误的分支继续，从而分叉会话。
        # 此处清除该会话的映射行，再让取消继续向上传播；下方 finally 负责释放
        # chat 锁与 token 在途占用。
        await _db(delete_sessions_for_chat, token_id, session_id)
        raise
    except Exception as e:
        code = _upstream_http_code(e)
        # c181d06：按真实状态码区分处置。被上游拒绝（401/403/429）时标记限流，
        # 并以 warning 记录原因——此前一律走 logger.exception，把可预期的限流
        # 与真正的故障混为一谈。会话映射在两种路径下都必须清除，否则已失效的
        # parent_message_id 会让下一轮从错误分支继续，造成会话分叉。
        if code in (401, 403, 429):
            await _db(mark_limited, token_id)
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.warning(
                "Chat request rejected by upstream (session %s, parent %s): %s",
                session_id,
                parent_message_id,
                e,
            )
            if _attempt + 1 >= MAX_UPSTREAM_ATTEMPTS:
                return _api_error_response(e, is_anthropic)
            # 标记限流后递归重试：下一帧读取到 RATE_LIMITED 状态，由入口处的
            # 轮换段挑选新账号重建会话。锁与在途占用须先交还，否则重试会等待
            # 本帧持有的锁而挂起（与下方通用重试路径同理）。
            lock_owner.release()
            lease.release()
            return await handle_chat(
                messages,
                model,
                thinking,
                search,
                stream,
                tools,
                is_anthropic,
                req_model,
                scope,
                search_sink=search_sink,
                _attempt=_attempt + 1,
                _exclude_token=token_id,
                is_responses=is_responses,
                response_opts=response_opts,
            )

        logger.exception(
            "Chat request failed (session %s, parent %s): %s",
            session_id,
            parent_message_id,
            e,
        )
        await _db(delete_sessions_for_chat, token_id, session_id)
        if _attempt + 1 >= MAX_UPSTREAM_ATTEMPTS:
            # B10：预算耗尽——直接抛出（已脱敏、已截断的）上游错误，不再无限重试。
            return _api_error_response(e, is_anthropic)
        # PR #26 review fix (Blocker 2 — retry self-deadlock): the recursive
        # call can resolve to the SAME chat (the retry re-derives the session
        # from whatever the persistence layer still returns). Recursing while
        # this frame still owns the lock made the retry wait on a lock its own
        # caller held — the request hung until the client gave up, on exactly
        # the errors the retry exists for. Surrender the lock first: release()
        # is scoped to this holder and idempotent, so the finally below becomes
        # a no-op, the retry re-acquires cleanly, and queued same-chat requests
        # are no longer starved for the entire retry either.
        lock_owner.release()
        # 在途占用同样须在递归前交还：重试会重新选取 token，本帧登记的计数
        # 若留着，旧账号会持续显得繁忙，干扰后续的负载均衡决策。
        lease.release()
        # B10：先做带抖动的退避，再换一个 token 重试——旧的单次重试会再次进入
        # pick_token() 的抽取，可能又落到同一个被污染的 token/会话上，正是 #33
        # 症状持续的原因。
        await asyncio.sleep(random.uniform(0.25, 0.75) * (1.5**_attempt))
        return await handle_chat(
            messages,
            model,
            thinking,
            search,
            stream,
            tools,
            is_anthropic,
            req_model,
            scope,
            search_sink=search_sink,
            _attempt=_attempt + 1,
            _exclude_token=token_id,
            is_responses=is_responses,
            response_opts=response_opts,
        )
    finally:
        if not lock_transferred:
            lock_owner.release()
        if not lease_transferred:
            lease.release()


async def collect_response(gen):
    text = ""
    async for chunk in gen:
        text += chunk
    return text


def _messages_text(messages):
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            parts.append(
                " ".join(
                    p.get("text", "")
                    for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            )
        else:
            parts.append(str(c))
    return "\n".join(parts)


def _cached_prompt_tokens(sig, prompt_tokens):
    """Prompt tokens this turn can read from the upstream prompt cache.

    The DeepSeek web endpoint reports no token metadata, so the bridge cannot
    read a real cache-hit figure back. Caching still applies upstream: the
    client resends the whole conversation every turn, so the previous turn's
    full prompt is a prefix of this turn's prompt. That remembered size is the
    closest honest reproduction of DeepSeek's own `prompt_cache_hit_tokens`.
    """
    cached = peek_cached_input(sig)
    return max(0, min(cached, prompt_tokens))


def _remember_prompt_tokens(next_sig, prompt_tokens):
    """Record this turn's prompt size as the next turn's cacheable prefix."""
    if next_sig:
        record_cached_input(next_sig, prompt_tokens)


async def _hold_think_tags(gen):
    carry = ""
    async for chunk in gen:
        chunk = carry + chunk
        carry = ""
        hold = 0
        for tag in ("<think>", "</think>"):
            for i in range(1, len(tag)):
                if chunk.endswith(tag[:i]):
                    hold = max(hold, i)
        if hold:
            carry = chunk[-hold:]
            chunk = chunk[:-hold]
        if chunk:
            yield chunk
    if carry:
        yield carry


async def stream_response(
    gen,
    model,
    messages,
    token_id,
    session_id,
    sig,
    tools,
    req_model=None,
    parent_message_id=0,
    scope="",
    lease=None,
):
    parser = StreamToolParser()
    reasoning_filter = ReasoningMarkupFilter()
    full_text = ""
    is_thinking = False
    aborted = False
    failed = False
    try:
        async for chunk in _hold_think_tags(gen):
            if not chunk:
                continue
            full_text += chunk

            if "<think>" in chunk:
                is_thinking = True
                chunk = chunk.replace("<think>", "").lstrip("\n")

            end_thinking = False
            if "</think>" in chunk:
                is_thinking = False
                end_thinking = True
                parts = chunk.split("</think>")
                think_part = parts[0]
                chunk = parts[1].lstrip("\n") if len(parts) > 1 else ""
                if think_part:
                    cleaned = reasoning_filter.feed(think_part)
                    if cleaned:
                        yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': cleaned}}]})}\n\n"

            if is_thinking and chunk:
                cleaned = reasoning_filter.feed(chunk)
                if cleaned:
                    yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': cleaned}}]})}\n\n"
                continue

            if end_thinking and not chunk:
                continue

            for r in parser.feed(chunk):
                if "text" in r:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': r['text']}}]})}\n\n"
        await _db(mark_active, token_id)
    except (asyncio.CancelledError, GeneratorExit):
        aborted = True
        failed = True
        # B2 (Stage 1 audit): the exchange never completed upstream. The stored
        # session rows still point at the pre-abort parent_message_id, so the
        # next turn would reuse them and fork/duplicate the conversation.
        # Purge them — the next request opens a fresh chat, which is already
        # the supported first-message path. Deletion must not be skipped even
        # while the task is being torn down, hence it lives here and not in
        # the finally block.
        try:
            await _db(delete_sessions_for_chat, token_id, session_id)
        except Exception:
            logger.exception(
                "stream_response: failed to purge session rows after client abort"
            )
        raise
    except Exception as e:
        failed = True
        code = _upstream_http_code(e)
        # c181d06：限流属可预期状态，以 warning 记录并按码标记；其它异常才是
        # 故障，走 exception。两条路径都清除会话映射，避免失效的 parent 造成分叉。
        if code in (401, 403, 429):
            await _db(mark_limited, token_id)
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.warning("stream_response upstream HTTP %s: %s", code, e)
        else:
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.exception("stream_response failed")
        with suppress(Exception):
            yield f"data: {json.dumps({'error': {'message': str(e)[:300]}})}\n\n"
    finally:
        parsed_tools, clean_text = parse_tools(full_text)
        clean_text = re.sub(
            r"<think>.*?</think>", "", clean_text, flags=re.DOTALL
        ).strip()
        clean_text = re.sub(
            r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>",
            "",
            clean_text,
            flags=re.IGNORECASE,
        ).strip()

        if not failed:
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = generate_signature_sync(next_messages, model, scope)
            await _db(
                save_session, sig, token_id, session_id, next_parent(parent_message_id)
            )
            await _db(
                save_session,
                next_sig,
                token_id,
                session_id,
                next_parent(parent_message_id),
            )

        if not aborted and not failed:
            try:
                tail_reasoning = reasoning_filter.flush()
                if tail_reasoning:
                    yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': tail_reasoning}}]})}\n\n"
                if not parsed_tools:
                    for r in parser.flush():
                        if "text" in r:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': r['text']}}]})}\n\n"

                if parsed_tools:
                    for i, tc in enumerate(parsed_tools):
                        delta_tc = {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [delta_tc]}}]})}\n\n"
                    yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"
                else:
                    yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                # 流式 usage：OpenAI 规范要求 include_usage 时在 [DONE] 之前补一条只含 usage 的 chunk。
                # 原先直接发 [DONE]，客户端拿不到 completion_tokens，DSH 的 tok/s 面板因此不显示。
                in_tokens = count_tok(_messages_text(messages))
                # B7: usage on the cleaned completion, not raw full_text —
                # think-tag reasoning and tool markup are not billed output.
                usage_out = _completion_usage_text(clean_text, parsed_tools)
                out_tokens = count_tok(usage_out) if usage_out else 0
                cached_tokens = _cached_prompt_tokens(sig, in_tokens)
                if not failed:
                    _remember_prompt_tokens(next_sig, in_tokens)
                await _record_usage(
                    model,
                    token_id,
                    scope,
                    in_tokens,
                    out_tokens,
                    cached_tokens,
                    req_model=req_model,
                )
                yield f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': in_tokens, 'completion_tokens': out_tokens, 'total_tokens': in_tokens + out_tokens, 'prompt_tokens_details': {'cached_tokens': cached_tokens}}})}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                pass
        # 释放该 token 的在途占用。必须置于 finally 内：客户端中断
        # （CancelledError / GeneratorExit）时同样要注销，否则计数只增不减，
        # 会让该账号在此后永久显得最忙，破坏负载均衡。
        if lease is not None:
            lease.release()


async def stream_anthropic_response(
    gen,
    model,
    messages,
    token_id,
    session_id,
    sig,
    tools,
    req_model=None,
    parent_message_id=0,
    scope="",
    lease=None,
):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    in_tokens = count_tok(_messages_text(messages))
    cached_tokens = _cached_prompt_tokens(sig, in_tokens)
    model_name = req_model if req_model else model
    start_evt = f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': in_tokens, 'output_tokens': 1, 'cache_read_input_tokens': cached_tokens}}})}\n\n"
    yield start_evt

    parser = StreamToolParser()
    reasoning_filter = ReasoningMarkupFilter()
    full_text = ""
    text_block_started = False
    block_index = 0
    aborted = False
    failed = False

    try:
        is_thinking = False
        async for chunk in _hold_think_tags(gen):
            if not chunk:
                continue
            full_text += chunk

            if "<think>" in chunk:
                is_thinking = True
                chunk = chunk.replace("<think>", "").lstrip("\n")
                start_block = f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking'}})}\n\n"
                yield start_block

            end_thinking = False
            if "</think>" in chunk:
                is_thinking = False
                end_thinking = True
                parts = chunk.split("</think>")
                think_part = parts[0]
                chunk = parts[1].lstrip("\n") if len(parts) > 1 else ""
                if think_part:
                    cleaned = reasoning_filter.feed(think_part)
                    if cleaned:
                        delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': cleaned}})}\n\n"
                        yield delta_evt

            if is_thinking and chunk:
                cleaned = reasoning_filter.feed(chunk)
                if cleaned:
                    delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': cleaned}})}\n\n"
                    yield delta_evt
                continue

            if end_thinking:
                stop_evt = f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                yield stop_evt
                block_index += 1
                if not chunk:
                    continue

            for r in parser.feed(chunk):
                if "text" in r:
                    if not text_block_started:
                        start_block = f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield start_block
                        text_block_started = True
                    delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': r['text']}})}\n\n"
                    yield delta_evt
        await _db(mark_active, token_id)
    except (asyncio.CancelledError, GeneratorExit):
        aborted = True
        failed = True
        # B2 (Stage 1 audit): same purge as stream_response — a client abort
        # mid-stream leaves session rows pointing at a parent the upstream
        # chat never answered, and the next turn would fork the exchange.
        try:
            await _db(delete_sessions_for_chat, token_id, session_id)
        except Exception:
            logger.exception(
                "stream_anthropic_response: failed to purge session rows after client abort"
            )
        raise
    except Exception as e:
        failed = True
        code = _upstream_http_code(e)
        # c181d06：与 stream_response 同理，限流按 warning 记录并标记，其它异常
        # 走 exception；两条路径均清除会话映射。
        if code in (401, 403, 429):
            await _db(mark_limited, token_id)
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.warning("stream_anthropic_response upstream HTTP %s: %s", code, e)
        else:
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.exception("stream_anthropic_response failed")
        with suppress(Exception):
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)[:300]}})}\n\n"
    finally:
        parsed_tools, clean_text = parse_tools(full_text)
        clean_text = re.sub(
            r"<think>.*?</think>", "", clean_text, flags=re.DOTALL
        ).strip()
        clean_text = re.sub(
            r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>",
            "",
            clean_text,
            flags=re.IGNORECASE,
        ).strip()
        # B7: usage on the cleaned completion, not raw full_text. Computed
        # AFTER the strips above (mirrors stream_response): this path used to
        # count tokens before them, so /v1/messages streams billed the whole
        # <think> reasoning share as output_tokens (Stage 1 review, finding 1).
        usage_out = _completion_usage_text(clean_text, parsed_tools)
        out_tokens = count_tok(usage_out) if usage_out else 0

        if not failed:
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = generate_signature_sync(next_messages, model, scope)
            await _db(
                save_session, sig, token_id, session_id, next_parent(parent_message_id)
            )
            await _db(
                save_session,
                next_sig,
                token_id,
                session_id,
                next_parent(parent_message_id),
            )
            _remember_prompt_tokens(next_sig, in_tokens)

        def _tb(text):
            return (
                f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index_local[0], 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index_local[0], 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"
            )

        block_index_local = [block_index]
        tail_events = ""
        tail_reasoning = reasoning_filter.flush()
        if tail_reasoning:
            tail_events += f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index_local[0], 'delta': {'type': 'thinking_delta', 'thinking': tail_reasoning}})}\n\n"
        if is_thinking:
            tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"
            block_index_local[0] += 1

        flushed_text = ""
        if not parsed_tools:
            for r in parser.flush():
                if "text" in r:
                    flushed_text += r["text"]

        if not text_block_started and not parsed_tools and (clean_text or flushed_text):
            tail_events += _tb(clean_text or flushed_text)
        elif text_block_started and not parsed_tools:
            tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"

        if parsed_tools:
            for tc in parsed_tools:
                tool_input = (
                    json.loads(tc["function"]["arguments"])
                    if isinstance(tc["function"]["arguments"], str)
                    else tc["function"]["arguments"]
                )
                json_str = json.dumps(tool_input)
                tail_events += f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index_local[0], 'content_block': {'type': 'tool_use', 'id': tc['id'], 'name': tc['function']['name'], 'input': {}}})}\n\n"
                tail_events += f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index_local[0], 'delta': {'type': 'input_json_delta', 'partial_json': json_str}})}\n\n"
                tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"
                block_index_local[0] += 1
            tail_events += f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
        else:
            tail_events += f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
        tail_events += (
            f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        )

        if not aborted and not failed:
            await _record_usage(
                model,
                token_id,
                scope,
                in_tokens,
                out_tokens,
                cached_tokens,
                req_model=req_model,
            )
            try:
                for evt in tail_events.split("\n\n"):
                    if evt.strip():
                        yield evt + "\n\n"
            except asyncio.CancelledError:
                pass
        # 同 stream_response：在 finally 内注销在途占用，客户端中断也生效。
        if lease is not None:
            lease.release()


def _completion_usage_text(clean_text, parsed_tools):
    """Text whose token count is billed as completion tokens (B7, Stage 1 audit).

    The cleaned reply — <think> reasoning and tool-call markup stripped — plus
    the serialized tool-call arguments the client actually receives. Counting
    the raw upstream text over-billed by the thinking+markup share, inflating
    completion_tokens and every gateway cost derived from them."""
    parts = [clean_text] if clean_text else []
    for tc in parsed_tools or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        if fn.get("name"):
            parts.append(str(fn["name"]))
        if fn.get("arguments"):
            parts.append(str(fn["arguments"]))
    return "\n".join(parts)


def format_response(
    text, model, messages, tools=None, cached_tokens=0, prompt_tokens=None
):
    parsed_tools, clean_text = parse_tools(text)

    reasoning = None
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*", text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
    clean_text = re.sub(
        r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>",
        "",
        clean_text,
        flags=re.IGNORECASE,
    ).strip()

    in_tokens = (
        prompt_tokens
        if prompt_tokens is not None
        else count_tok(_messages_text(messages))
    )
    # B7: bill the CLEANED completion (see _completion_usage_text), not the
    # raw upstream text whose thinking + markup share the client never asked
    # to pay for.
    usage_text = _completion_usage_text(clean_text, parsed_tools)
    out_tokens = count_tok(usage_text) if usage_text else 0
    # 计价与 /stats 面板同源：输入拆缓存命中/未命中，且按当前时刻的峰谷档取价。
    # 这里是非流式的最终响应，只能按「生成完成的这一刻」定档；面板的逐小时
    # 聚合仍以采样落库时的小时键为准，两者对同一小时不会给出不同档位。
    cache_miss = max(0, in_tokens - cached_tokens)
    charge = cost_of_hour(
        model,
        {"cache_hit": cached_tokens, "cache_miss": cache_miss, "output": out_tokens},
        hour_key(beijing_now()),
    )
    cost = charge["total"]

    msg_dict = {
        "role": "assistant",
        "content": clean_text if not parsed_tools else None,
        "tool_calls": parsed_tools if parsed_tools else None,
    }
    if reasoning:
        msg_dict["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": msg_dict,
                "finish_reason": "tool_calls" if parsed_tools else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "cost": round(cost, 6),
        },
    }


def _anthropic_search_result_block(search_results):
    """Build the Anthropic ``web_search_tool_result`` block from upstream results.

    Anthropic clients (and the DSH web-search provider in particular) require the
    sources to be machine-readable: the provider walks this block for citeable
    items and rejects the whole response when it is absent. Upstream DeepSeek
    events already carry ``url``/``title``/``snippet``/``published_at``, so the
    block is assembled verbatim rather than reconstructed from the prose, whose
    ``[citation:N]`` markers do not survive tokenization reliably.

    Items without a URL are dropped: the schema requires one and an address-less
    entry cannot be cited by a client.

    @param search_results - raw result dicts collected from the upstream stream.
    @returns the block, or ``None`` when no citeable item remains.
    """
    items = []
    for r in search_results or []:
        if not isinstance(r, dict) or not r.get("url"):
            continue
        item = {"type": "web_search_result", "url": r["url"]}
        if r.get("title"):
            item["title"] = r["title"]
        published = r.get("published_at")
        if isinstance(published, (int, float)) and published > 0:
            item["page_age"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(published)
            )
        items.append(item)
    if not items:
        return None
    return {"type": "web_search_tool_result", "content": items}


def format_anthropic_response(result, model, search_results=None):
    choice = result["choices"][0]
    msg = choice["message"]
    ant_content = []

    search_block = _anthropic_search_result_block(search_results)
    if search_block:
        ant_content.append(search_block)

    if msg.get("reasoning_content"):
        ant_content.append({"type": "thinking", "thinking": msg["reasoning_content"]})

    if msg.get("content"):
        text_block = {"type": "text", "text": msg["content"]}
        # The DSH provider reads each source's excerpt from a text block's
        # citations, keyed by URL, not from the result block itself. Attaching
        # them here is what makes the snippet survive into the normalized result.
        citations = []
        for r in search_results or []:
            if not isinstance(r, dict) or not r.get("url") or not r.get("snippet"):
                continue
            citations.append(
                {
                    "type": "web_search_result_location",
                    "url": r["url"],
                    "title": r.get("title"),
                    "cited_text": r["snippet"],
                }
            )
        if citations:
            text_block["citations"] = citations
        ant_content.append(text_block)

    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            args = tc["function"]["arguments"]
            tool_input = json.loads(args) if isinstance(args, str) else args
            ant_content.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": tool_input,
                }
            )
    usage = result.get("usage", {})
    msg_id = result["id"]
    if not msg_id.startswith("msg_"):
        msg_id = f"msg_{msg_id.replace('chatcmpl-', '')}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": ant_content,
        "model": model,
        "stop_reason": "tool_use" if msg.get("tool_calls") else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": usage.get("prompt_tokens_details", {}).get(
                "cached_tokens", 0
            ),
        },
    }


@app.post("/v1/files")
@app.post("/v1/files/upload")
async def files_upload(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    tok_id = await _db(pick_token)
    if not tok_id:
        return JSONResponse({"error": "No tokens available"}, status_code=503)
    tok = await _db(get_token, tok_id)
    if not tok:
        return JSONResponse({"error": "Token not found"}, status_code=503)
    _set_key_name(tok.get("alias"))
    # 上传同样占用该 token 的一个在途名额：pick_token() 的负载均衡读取
    # 这份计数，漏登记会让大文件上传期间的账号看起来仍然空闲。
    lease = TokenLease(tok_id)
    try:
        form = await request.form()
        file_obj = form.get("file")
        if not file_obj:
            return JSONResponse({"error": "No file provided"}, status_code=400)
        file_bytes = await file_obj.read(25 * 1024 * 1024 + 1)
        if len(file_bytes) > 25 * 1024 * 1024:
            return JSONResponse({"error": "File too large"}, status_code=413)
        filename = getattr(file_obj, "filename", "file.bin")
        content_type = getattr(file_obj, "content_type", "application/octet-stream")
        file_info = None
        async for upload_status, data in upload_file(
            file_bytes, filename, content_type, tok["token"]
        ):
            if upload_status == "success":
                file_info = data
                break
        if not file_info:
            return JSONResponse({"error": "Upload failed"}, status_code=500)
    finally:
        lease.release()
    # B4：把上传钉在完成它的 token 上，后续聊天才能优先使用（或改挂到）该账号。
    await _db(record_file, file_info["file_id"], tok_id)

    if request.url.path.startswith("/v1/files/upload"):
        return {
            "id": file_info["file_id"],
            "type": "file",
            "filename": filename,
            "size": file_info["size"],
            "created_at": file_info["anthropic_timestamp"],
        }
    return {
        "id": file_info["file_id"],
        "object": "file",
        "bytes": file_info["size"],
        "created_at": file_info["openai_timestamp"],
        "filename": filename,
        "purpose": "answers",
    }


@app.get("/v1/files/{file_id}/content")
@app.get("/v1/files/{file_id}")
async def files_content(file_id: str, request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    tok_id = await _db(pick_token)
    if not tok_id:
        return JSONResponse({"error": "No tokens available"}, status_code=503)
    tok = await _db(get_token, tok_id)
    if not tok:
        return JSONResponse({"error": "Token not found"}, status_code=503)
    _set_key_name(tok.get("alias"))
    # 仅在取回响应头的握手阶段占用在途名额：mime 到手后上游响应已建立，
    # 后续分片下载直接从该连接读取，不再计入调度器的负载视图。
    lease = TokenLease(tok_id)
    try:
        gen = get_file_content(tok["token"], file_id)
        try:
            mime = await gen.__anext__()
        except StopAsyncIteration:
            return JSONResponse({"error": "File not found"}, status_code=404)
        except Exception:
            return JSONResponse({"error": "File fetch failed"}, status_code=502)
    finally:
        lease.release()

    async def stream_chunks():
        async for chunk in gen:
            yield chunk

    return StreamingResponse(
        stream_chunks(), media_type=mime or "application/octet-stream"
    )


def is_thinking_enabled(body, request=None):
    effort = body.get("effort")
    if effort is not None:
        e_str = str(effort).strip().lower()
        if e_str in [
            "medium",
            "high",
            "max",
            "ultra",
            "extreme",
            "enabled",
            "adaptive",
            "on",
        ]:
            return True
        if e_str in ["low", "none", "off", "disable", "disabled", "false"]:
            return False

    out_cfg = body.get("output_config")
    if isinstance(out_cfg, dict):
        out_effort = out_cfg.get("effort") or out_cfg.get("reasoning_effort")
        if out_effort is not None:
            e_str = str(out_effort).strip().lower()
            if e_str in [
                "medium",
                "high",
                "max",
                "ultra",
                "extreme",
                "enabled",
                "adaptive",
                "on",
            ]:
                return True
            if e_str in ["low", "none", "off", "disable", "disabled", "false"]:
                return False

    thinking_val = body.get("thinking")
    if isinstance(thinking_val, dict):
        t_type = str(thinking_val.get("type", "")).strip().lower()
        if t_type in ["enabled", "adaptive", "true"]:
            return True
        if t_type == "disabled":
            return False
        budget = thinking_val.get("budget_tokens", 0)
        if isinstance(budget, (int, float)) and budget > 0:
            return True
        t_effort = (
            thinking_val.get("effort")
            or thinking_val.get("reasoning_effort")
            or thinking_val.get("level")
        )
        if t_effort is not None:
            e_str = str(t_effort).strip().lower()
            if e_str in [
                "medium",
                "high",
                "max",
                "ultra",
                "extreme",
                "enabled",
                "adaptive",
                "on",
            ]:
                return True
            if e_str in ["low", "none", "off", "disable", "disabled", "false"]:
                return False
    elif isinstance(thinking_val, str):
        t_str = thinking_val.strip().lower()
        if t_str in [
            "medium",
            "high",
            "max",
            "ultra",
            "extreme",
            "true",
            "enabled",
            "adaptive",
            "on",
        ]:
            return True
        if t_str in ["low", "none", "off", "disable", "disabled", "false"]:
            return False
    elif isinstance(thinking_val, bool):
        return thinking_val

    reasoning_effort = body.get("reasoning_effort")
    if reasoning_effort is not None:
        effort_str = str(reasoning_effort).strip().lower()
        if effort_str in ["medium", "high", "max", "ultra", "extreme"]:
            return True
        if effort_str in ["low", "none", "off", "disable", "disabled"]:
            return False

    if request:
        req_effort = (
            request.headers.get("anthropic-thinking")
            or request.headers.get("x-anthropic-thinking")
            or request.headers.get("effort")
            or request.headers.get("x-effort")
        )
        if req_effort:
            e_str = str(req_effort).strip().lower()
            if e_str in [
                "medium",
                "high",
                "max",
                "ultra",
                "extreme",
                "enabled",
                "adaptive",
                "on",
            ]:
                return True
    return False


# DeepSeek now serves a single model (v4.1flash) as the website default.
# Every request — whatever model name the client sends, including legacy
# aliases (instant, expert, vision, anthropic/claude-*) — is served by it.
SINGLE_MODEL = "v4.1flash"


def resolve_model(model_raw):
    return SINGLE_MODEL


async def _json_body(request: Request):
    """以 JSON 解析请求体；空体或畸形载荷属于客户端错误。

    Stage 1 次要项：补全端点此前直接调用 request.json()，因此空请求体或非法
    JSON 会上报为 500——把客户端错误报告成服务端故障，并触发服务端告警。
    """
    try:
        raw = await request.body()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Unable to read request body"
        ) from None
    if not raw or not raw.strip():
        raise HTTPException(
            status_code=400, detail="Empty request body; expected a JSON object"
        )
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON in request body"
        ) from None
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object"
        )
    return body


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await _json_body(request)
    messages = body.get("messages", [])
    model = resolve_model(body.get("model"))
    thinking = is_thinking_enabled(body, request)
    search = body.get("search", False)
    stream = body.get("stream", False)
    tools = body.get("tools", None)
    return await handle_chat(
        messages,
        model,
        thinking,
        search,
        stream,
        tools,
        req_model=body.get("model"),
        scope=get_api_key(request),
    )


_response_history = OrderedDict()
RESPONSE_HISTORY_MAX = 2048
GENERIC_RESPONSE_ERROR = "An internal error occurred while generating the response."


def _store_response_history(resp_id, msgs):
    if not resp_id:
        return
    copied = [dict(m) for m in msgs]
    _response_history[resp_id] = copied
    _response_history.move_to_end(resp_id)
    while len(_response_history) > RESPONSE_HISTORY_MAX:
        _response_history.popitem(last=False)


def _normalize_responses_tools(tools):
    if not isinstance(tools, list):
        return []
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            strict_val = fn.get("strict", t.get("strict"))
            out.append(
                {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description"),
                    "parameters": fn.get(
                        "parameters",
                        fn.get("input_schema", {"type": "object", "properties": {}}),
                    ),
                    "strict": strict_val if isinstance(strict_val, bool) else True,
                }
            )
        else:
            out.append(t)
    return out


def _responses_usage(in_tokens, out_tokens):
    return {
        "input_tokens": in_tokens,
        "input_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        },
        "output_tokens": out_tokens,
        "output_tokens_details": {
            "reasoning_tokens": 0,
        },
        "total_tokens": in_tokens + out_tokens,
    }


def _responses_effort(opts):
    reasoning_cfg = opts.get("reasoning")
    effort = reasoning_cfg.get("effort") if isinstance(reasoning_cfg, dict) else None
    if effort is None and isinstance(reasoning_cfg, str):
        effort = reasoning_cfg
    if effort is None:
        for key in ("reasoning_effort", "effort"):
            candidate = opts.get(key)
            if isinstance(candidate, str) and candidate.strip():
                effort = candidate
                break
    if isinstance(effort, str) and effort.strip().lower() in (
        "minimal",
        "low",
        "medium",
        "high",
    ):
        return effort.strip().lower()
    return None


def _build_response_object(
    resp_id,
    created_at,
    model,
    output,
    status="completed",
    usage=None,
    completed_at=None,
    error=None,
    opts=None,
):
    opts = opts if isinstance(opts, dict) else {}
    reasoning_cfg = opts.get("reasoning")
    effort = _responses_effort(opts)
    if isinstance(reasoning_cfg, dict):
        reasoning_obj = {
            "effort": effort,
            "summary": reasoning_cfg.get("summary"),
        }
    else:
        reasoning_obj = {
            "effort": effort,
            "summary": None,
        }
    text_cfg = opts.get("text")
    if not isinstance(text_cfg, dict) or not isinstance(text_cfg.get("format"), dict):
        text_cfg = {"format": {"type": "text"}}
    metadata = opts.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    parallel_tc = opts.get("parallel_tool_calls")
    if not isinstance(parallel_tc, bool):
        parallel_tc = True
    store_val = opts.get("store")
    if not isinstance(store_val, bool):
        store_val = True
    temp_val = opts.get("temperature")
    temp_val = float(temp_val) if isinstance(temp_val, (int, float)) else 1.0
    top_p_val = opts.get("top_p")
    top_p_val = float(top_p_val) if isinstance(top_p_val, (int, float)) else 1.0
    top_logprobs = opts.get("top_logprobs")
    top_logprobs = top_logprobs if isinstance(top_logprobs, int) else 0
    truncation = opts.get("truncation")
    if truncation not in ("auto", "disabled"):
        truncation = "disabled"
    background = opts.get("background")
    if not isinstance(background, bool):
        background = False
    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "completed_at": completed_at,
        "background": background,
        "error": error,
        "incomplete_details": None,
        "instructions": opts.get("instructions"),
        "max_output_tokens": opts.get("max_output_tokens"),
        "max_tool_calls": opts.get("max_tool_calls"),
        "model": model,
        "output": output,
        "parallel_tool_calls": parallel_tc,
        "previous_response_id": opts.get("previous_response_id"),
        "reasoning": reasoning_obj,
        "service_tier": opts.get("service_tier") or "default",
        "store": store_val,
        "temperature": temp_val,
        "text": text_cfg,
        "tool_choice": (
            opts.get("tool_choice") if opts.get("tool_choice") is not None else "auto"
        ),
        "tools": _normalize_responses_tools(opts.get("tools")),
        "top_logprobs": top_logprobs,
        "top_p": top_p_val,
        "truncation": truncation,
        "usage": usage,
        "user": opts.get("user"),
        "metadata": metadata,
    }


def format_responses_response(result, model=None, opts=None):
    choice = result["choices"][0]
    msg = choice["message"]
    output = []

    if msg.get("reasoning_content"):
        output.append(
            {
                "id": f"rs_{uuid.uuid4().hex}",
                "type": "reasoning",
                "status": "completed",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": msg["reasoning_content"],
                    }
                ],
            }
        )

    tool_calls = msg.get("tool_calls") or []
    content_text = msg.get("content")
    if content_text is not None or not tool_calls:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content_text or "",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )

    for tc in tool_calls:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name", ""),
                "arguments": args_str,
            }
        )

    raw_usage = result.get("usage", {})
    in_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
    out_tokens = raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0))
    preset_id = opts.get("_resp_id") if isinstance(opts, dict) else None
    raw_id = str(result.get("id", ""))
    if preset_id:
        resp_id = preset_id
    elif raw_id.startswith("resp_"):
        resp_id = raw_id
    elif raw_id.startswith("chatcmpl-"):
        resp_id = f"resp_{raw_id[len('chatcmpl-') :].replace('-', '')}"
    else:
        resp_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(result.get("created", result.get("created_at", time.time())))
    completed_at = int(time.time())
    model_name = model or result.get("model", SINGLE_MODEL)
    return _build_response_object(
        resp_id,
        created_at,
        model_name,
        output,
        status="completed",
        usage=_responses_usage(in_tokens, out_tokens),
        completed_at=completed_at,
        error=None,
        opts=opts,
    )


async def stream_responses_response(
    gen,
    model,
    messages,
    token_id,
    session_id,
    sig,
    tools,
    req_model=None,
    parent_message_id=0,
    scope="",
    opts=None,
    lease=None,
):
    preset_id = opts.get("_resp_id") if isinstance(opts, dict) else None
    resp_id = preset_id or f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    model_name = req_model if req_model else model
    seq = 0
    output_index = 0
    completed_output = []
    rs_id = None
    rs_index = None
    rs_started = False
    rs_closed = False
    rs_text = ""
    msg_id = None
    msg_index = None
    msg_started = False
    msg_text = ""

    def _sse(payload):
        nonlocal seq
        payload["sequence_number"] = seq
        seq += 1
        return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"

    def _open_rs():
        nonlocal rs_id, rs_index, output_index, rs_started
        rs_id = f"rs_{uuid.uuid4().hex}"
        rs_index = output_index
        output_index += 1
        rs_started = True
        return [
            _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": rs_index,
                    "item": {
                        "id": rs_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": [],
                    },
                }
            ),
            _sse(
                {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": rs_id,
                    "output_index": rs_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                }
            ),
        ]

    def _close_rs():
        nonlocal rs_closed
        rs_closed = True
        rs_part = {"type": "summary_text", "text": rs_text}
        rs_item = {
            "id": rs_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [rs_part],
        }
        completed_output.append(rs_item)
        return [
            _sse(
                {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": rs_id,
                    "output_index": rs_index,
                    "summary_index": 0,
                    "text": rs_text,
                }
            ),
            _sse(
                {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": rs_id,
                    "output_index": rs_index,
                    "summary_index": 0,
                    "part": rs_part,
                }
            ),
            _sse(
                {
                    "type": "response.output_item.done",
                    "output_index": rs_index,
                    "item": rs_item,
                }
            ),
        ]

    def _open_msg():
        nonlocal msg_id, msg_index, output_index, msg_started
        msg_id = f"msg_{uuid.uuid4().hex}"
        msg_index = output_index
        output_index += 1
        msg_started = True
        return [
            _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": msg_index,
                    "item": {
                        "id": msg_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                }
            ),
            _sse(
                {
                    "type": "response.content_part.added",
                    "item_id": msg_id,
                    "output_index": msg_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                        "logprobs": [],
                    },
                }
            ),
        ]

    def _delta_msg(text):
        nonlocal msg_text
        frames = _open_msg() if not msg_started else []
        msg_text += text
        frames.append(
            _sse(
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": msg_index,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                }
            )
        )
        return frames

    initial_resp = _build_response_object(
        resp_id,
        created_at,
        model_name,
        [],
        status="in_progress",
        usage=None,
        completed_at=None,
        error=None,
        opts=opts,
    )
    yield _sse({"type": "response.created", "response": initial_resp})
    yield _sse({"type": "response.in_progress", "response": initial_resp})

    parser = StreamToolParser()
    full_text = ""
    aborted = False
    failed = False

    try:
        is_thinking = False
        async for chunk in _hold_think_tags(gen):
            if not chunk:
                continue
            full_text += chunk

            if "<think>" in chunk:
                is_thinking = True
                chunk = chunk.replace("<think>", "").lstrip("\n")
                if not rs_started:
                    for f in _open_rs():
                        yield f

            end_thinking = False
            if "</think>" in chunk:
                is_thinking = False
                end_thinking = True
                parts = chunk.split("</think>")
                think_part = parts[0]
                chunk = parts[1].lstrip("\n") if len(parts) > 1 else ""
                if think_part:
                    if not rs_started:
                        for f in _open_rs():
                            yield f
                    rs_text += think_part
                    yield _sse(
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": rs_id,
                            "output_index": rs_index,
                            "summary_index": 0,
                            "delta": think_part,
                        }
                    )

            if is_thinking and chunk:
                rs_text += chunk
                yield _sse(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": rs_id,
                        "output_index": rs_index,
                        "summary_index": 0,
                        "delta": chunk,
                    }
                )
                continue

            if end_thinking:
                if rs_started and not rs_closed:
                    for f in _close_rs():
                        yield f
                if not chunk:
                    continue

            for r in parser.feed(chunk):
                if "text" in r and r["text"]:
                    if not msg_started and not r["text"].strip():
                        continue
                    for f in _delta_msg(r["text"]):
                        yield f
        await _db(mark_active, token_id)
    except (asyncio.CancelledError, GeneratorExit):
        aborted = True
        failed = True
        try:
            await _db(delete_sessions_for_chat, token_id, session_id)
        except Exception:
            logger.exception(
                "stream_responses_response: failed to purge session rows after client abort"
            )
        raise
    except Exception as e:
        failed = True
        code = _upstream_http_code(e)
        if code in (401, 403, 429):
            await _db(mark_limited, token_id)
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.warning("stream_responses_response upstream HTTP %s: %s", code, e)
        else:
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.exception("stream_responses_response failed")
        with suppress(Exception):
            failed_resp = _build_response_object(
                resp_id,
                created_at,
                model_name,
                completed_output,
                status="failed",
                usage=None,
                completed_at=None,
                error={"code": "server_error", "message": GENERIC_RESPONSE_ERROR},
                opts=opts,
            )
            yield _sse({"type": "response.failed", "response": failed_resp})
    finally:
        parsed_tools, clean_text = parse_tools(full_text)
        clean_text = re.sub(
            r"<think>.*?</think>", "", clean_text, flags=re.DOTALL
        ).strip()
        clean_text = re.sub(
            r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>",
            "",
            clean_text,
            flags=re.IGNORECASE,
        ).strip()
        flushed_chunks = []
        if not parsed_tools:
            for r in parser.flush():
                if "text" in r and r["text"]:
                    if not msg_started and not flushed_chunks and not r["text"].strip():
                        continue
                    flushed_chunks.append(r["text"])
        combined_stream_text = msg_text + "".join(flushed_chunks)
        final_text = combined_stream_text if combined_stream_text else clean_text
        in_tokens = count_tok(_messages_text(messages))
        usage_out = _completion_usage_text(clean_text, parsed_tools)
        out_tokens = count_tok(usage_out) if usage_out else 0

        if not failed:
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = final_text.strip()
            next_messages.append(ast_msg)
            next_sig = generate_signature_sync(next_messages, model, scope)
            await _db(
                save_session, sig, token_id, session_id, next_parent(parent_message_id)
            )
            await _db(
                save_session,
                next_sig,
                token_id,
                session_id,
                next_parent(parent_message_id),
            )
            await _db(
                save_session,
                resp_id,
                token_id,
                session_id,
                next_parent(parent_message_id),
            )
            _store_response_history(resp_id, next_messages)

        if not failed:
            # 统计与计价的落点：与 chat/anthropic 两条流式路径同源，按清理后的
            # 文本计费，思考内容与工具标记不计入输出。
            await _record_usage(
                model,
                token_id,
                scope,
                in_tokens,
                out_tokens,
                _cached_prompt_tokens(sig, in_tokens),
                req_model=req_model,
            )

        if not aborted and not failed:
            try:
                if rs_started and not rs_closed:
                    for f in _close_rs():
                        yield f

                if not parsed_tools:
                    for fc in flushed_chunks:
                        for f in _delta_msg(fc):
                            yield f
                    if not msg_started:
                        for f in _delta_msg(clean_text) if clean_text else _open_msg():
                            yield f

                if msg_started:
                    final_part = {
                        "type": "output_text",
                        "text": final_text,
                        "annotations": [],
                        "logprobs": [],
                    }
                    final_msg_item = {
                        "id": msg_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [final_part],
                    }
                    completed_output.append(final_msg_item)
                    yield _sse(
                        {
                            "type": "response.output_text.done",
                            "item_id": msg_id,
                            "output_index": msg_index,
                            "content_index": 0,
                            "text": final_text,
                            "logprobs": [],
                        }
                    )
                    yield _sse(
                        {
                            "type": "response.content_part.done",
                            "item_id": msg_id,
                            "output_index": msg_index,
                            "content_index": 0,
                            "part": final_part,
                        }
                    )
                    yield _sse(
                        {
                            "type": "response.output_item.done",
                            "output_index": msg_index,
                            "item": final_msg_item,
                        }
                    )

                if parsed_tools:
                    for tc in parsed_tools:
                        fn = tc.get("function", {})
                        raw_args = fn.get("arguments", "{}")
                        args_str = (
                            raw_args
                            if isinstance(raw_args, str)
                            else json.dumps(raw_args)
                        )
                        fc_id = f"fc_{uuid.uuid4().hex}"
                        call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                        fn_name = fn.get("name", "")
                        fc_index = output_index
                        output_index += 1
                        yield _sse(
                            {
                                "type": "response.output_item.added",
                                "output_index": fc_index,
                                "item": {
                                    "id": fc_id,
                                    "type": "function_call",
                                    "status": "in_progress",
                                    "call_id": call_id,
                                    "name": fn_name,
                                    "arguments": "",
                                },
                            }
                        )
                        yield _sse(
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": fc_id,
                                "output_index": fc_index,
                                "delta": args_str,
                            }
                        )
                        yield _sse(
                            {
                                "type": "response.function_call_arguments.done",
                                "item_id": fc_id,
                                "output_index": fc_index,
                                "arguments": args_str,
                            }
                        )
                        fc_item = {
                            "id": fc_id,
                            "type": "function_call",
                            "status": "completed",
                            "call_id": call_id,
                            "name": fn_name,
                            "arguments": args_str,
                        }
                        completed_output.append(fc_item)
                        yield _sse(
                            {
                                "type": "response.output_item.done",
                                "output_index": fc_index,
                                "item": fc_item,
                            }
                        )

                completed_resp = _build_response_object(
                    resp_id,
                    created_at,
                    model_name,
                    completed_output,
                    status="completed",
                    usage=_responses_usage(in_tokens, out_tokens),
                    completed_at=int(time.time()),
                    error=None,
                    opts=opts,
                )
                yield _sse({"type": "response.completed", "response": completed_resp})
            except asyncio.CancelledError:
                pass
        # 释放该 token 的在途占用。置于 finally 内：客户端中断
        # （CancelledError / GeneratorExit）时同样要注销，否则计数只增不减，
        # 该账号会在此后永久显得最忙，破坏负载均衡。
        if lease is not None:
            lease.release()


def _responses_input_to_messages(inputs):
    if isinstance(inputs, (str, dict)):
        inputs = [inputs]
    messages = []
    call_names = {}
    for item in inputs:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            continue
        if item_type == "function_call":
            raw_args = item.get("arguments", "{}")
            args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
            call_id = (
                item.get("call_id")
                or item.get("id")
                or ("call_" + uuid.uuid4().hex[:8])
            )
            fn_name = item.get("name", "")
            if call_id and fn_name:
                call_names[call_id] = fn_name
            tc = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": args_str,
                },
            }
            if messages and messages[-1].get("role") == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tc)
            else:
                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": [tc]}
                )
            continue
        if item_type == "function_call_output":
            out_val = item.get("output", "")
            if isinstance(out_val, list):
                out_val = " ".join(
                    p.get("text", "")
                    for p in out_val
                    if isinstance(p, dict)
                    and p.get("type") in ("input_text", "output_text", "text")
                )
            elif not isinstance(out_val, str):
                out_val = json.dumps(out_val)
            call_id = item.get("call_id") or item.get("id") or ""
            tool_msg = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": out_val,
            }
            t_name = item.get("name") or call_names.get(call_id)
            if t_name:
                tool_msg["name"] = t_name
            messages.append(tool_msg)
            continue

        role = item.get("role", "user")
        if role == "developer":
            role = "system"
        content = item.get("content", [])
        msg_content = []
        if isinstance(content, str):
            msg_content = content
        else:
            for c in content:
                if isinstance(c, str):
                    msg_content.append({"type": "text", "text": c})
                elif not isinstance(c, dict):
                    continue
                elif c.get("type") in ("input_text", "output_text"):
                    msg_content.append({"type": "text", "text": c.get("text")})
                elif c.get("type") == "refusal":
                    msg_content.append({"type": "text", "text": c.get("refusal", "")})
                elif c.get("type") == "input_file":
                    if c.get("file_data"):
                        msg_content.append(
                            {
                                "type": "file",
                                "file_data": c.get("file_data"),
                                "filename": c.get("filename"),
                            }
                        )
                    else:
                        msg_content.append(
                            {"type": "file", "file_id": c.get("file_id")}
                        )
                elif c.get("type") == "input_image":
                    url = c.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if url:
                        msg_content.append(
                            {"type": "image_url", "image_url": {"url": url}}
                        )
                    elif c.get("file_id"):
                        msg_content.append(
                            {"type": "file", "file_id": c.get("file_id")}
                        )
                    else:
                        msg_content.append(c)
                else:
                    msg_content.append(c)
        if (
            role == "assistant"
            and isinstance(msg_content, list)
            and all(
                isinstance(p, dict) and p.get("type") == "text" for p in msg_content
            )
        ):
            msg_content = "\n".join(p.get("text", "") for p in msg_content)
        if role == "assistant" and messages and messages[-1].get("role") == "assistant":
            prev_c = messages[-1].get("content")
            if not prev_c:
                messages[-1]["content"] = msg_content
            elif isinstance(prev_c, str) and isinstance(msg_content, str):
                messages[-1]["content"] = (
                    f"{prev_c}\n{msg_content}" if msg_content else prev_c
                )
            elif isinstance(prev_c, list) and isinstance(msg_content, list):
                prev_c.extend(msg_content)
            continue
        messages.append({"role": role, "content": msg_content})
    return messages


@app.post("/v1/responses")
@app.post("/responses")
async def openai_responses(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await _json_body(request)
    model = resolve_model(body.get("model"))
    inputs = body.get("input", [])
    if isinstance(inputs, (str, dict)):
        inputs = [inputs]

    messages = _responses_input_to_messages(inputs)

    instructions = body.get("instructions")
    if isinstance(instructions, list):
        inst_msgs = _responses_input_to_messages(instructions)
        inst_text = "\n".join(
            _messages_text([m]) for m in inst_msgs if _messages_text([m])
        )
    elif isinstance(instructions, str):
        inst_text = instructions
    else:
        inst_text = ""

    text_cfg = body.get("text")
    if (
        isinstance(text_cfg, dict)
        and isinstance(text_cfg.get("format"), dict)
        and text_cfg["format"].get("type") == "json_schema"
    ):
        json_schema = text_cfg["format"].get("schema")
        if json_schema:
            schema_inst = f"You MUST return valid JSON adhering strictly to this JSON Schema:\n{json.dumps(json_schema)}"
            inst_text = f"{inst_text}\n\n{schema_inst}" if inst_text else schema_inst

    prev_resp_id = body.get("previous_response_id")
    if (
        prev_resp_id
        and prev_resp_id in _response_history
        and not any(m.get("role") == "assistant" for m in messages)
    ):
        prev_msgs = [dict(m) for m in _response_history[prev_resp_id]]
        if inst_text and not (
            prev_msgs
            and prev_msgs[0].get("role") == "system"
            and prev_msgs[0].get("content") == inst_text
        ):
            if prev_msgs and prev_msgs[0].get("role") == "system":
                prev_msgs[0] = {"role": "system", "content": inst_text}
            else:
                prev_msgs.insert(0, {"role": "system", "content": inst_text})
        messages = prev_msgs + messages
    elif inst_text:
        messages.insert(0, {"role": "system", "content": inst_text})

    thinking = is_thinking_enabled(body, request)
    search = bool(body.get("search", False))
    stream = bool(body.get("stream", False))
    raw_tools = body.get("tools", None)
    chat_tools = None
    if isinstance(raw_tools, list):
        chat_tools = []
        for t in raw_tools:
            if not isinstance(t, dict):
                continue
            t_type = str(t.get("type", ""))
            if t_type.startswith("web_search"):
                search = True
            else:
                chat_tools.append(t)
        if not chat_tools:
            chat_tools = None

    response_opts = dict(body)
    response_opts["thinking"] = thinking
    response_opts["_resp_id"] = f"resp_{uuid.uuid4().hex}"

    result = await handle_chat(
        messages,
        model,
        thinking,
        search,
        stream,
        chat_tools,
        scope=get_api_key(request),
        is_responses=True,
        response_opts=response_opts,
    )

    if stream:
        return result

    if isinstance(result, dict) and "choices" in result:
        return format_responses_response(
            result, result.get("model", model), response_opts
        )
    return result


def convert_anthropic_messages(messages):
    """Translate Anthropic message dicts into OpenAI-style dicts.

    tool_use blocks become assistant tool_calls and tool_result blocks become
    role="tool" messages so that build_prompt()/extract_tool_results() see real
    tool results and the signature cache can match across turns.
    """
    openai_msgs = []
    for m in messages:
        content = m.get("content", "")
        tool_calls = []
        tool_results = []
        if isinstance(content, list):
            parts = []
            image_parts = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif c.get("type") == "image":
                    image_parts.append(c)
                elif c.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": c.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                            "type": "function",
                            "function": {
                                "name": c.get("name", ""),
                                "arguments": json.dumps(c.get("input", {})),
                            },
                        }
                    )
                elif c.get("type") == "tool_result":
                    res_content = c.get("content", "")
                    if isinstance(res_content, list):
                        for item in res_content:
                            if isinstance(item, dict) and item.get("type") == "image":
                                image_parts.append(item)
                        res_content = " ".join(
                            item.get("text", "")
                            for item in res_content
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                    elif not isinstance(res_content, str):
                        res_content = str(res_content)
                    tool_results.append(
                        {
                            "tool_call_id": c.get("tool_use_id", ""),
                            "content": res_content,
                        }
                    )
            if image_parts:
                content = [
                    {"type": "text", "text": s} for s in parts if s
                ] + image_parts
            else:
                content = "\n".join(p for p in parts if p)
        if m.get("role") == "system":
            if content:
                openai_msgs.append({"role": "system", "content": content})
            continue
        if m.get("role") == "assistant":
            if isinstance(content, str) and (
                not content.strip() or content.strip() == "(no content)"
            ):
                content = None
            msg = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if msg["content"] is None and not tool_calls:
                continue
            openai_msgs.append(msg)
            continue
        for tr in tool_results:
            openai_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"],
                }
            )
        has_content = (
            bool(content) if not isinstance(content, list) else len(content) > 0
        )
        if has_content or not tool_results:
            openai_msgs.append({"role": m.get("role", "user"), "content": content})
    return openai_msgs


@app.post("/v1/messages")
@app.post("/messages")
async def anthropic_messages(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await _json_body(request)
    system = body.get("system", "")

    messages = body.get("messages", [])
    model = resolve_model(body.get("model"))

    thinking = is_thinking_enabled(body, request)
    stream = body.get("stream", False)
    tools = body.get("tools", [])

    openai_msgs = []
    if system:
        if isinstance(system, list):
            system_str = " ".join(
                c.get("text", "")
                for c in system
                if isinstance(c, dict) and c.get("type") == "text"
            )
        else:
            system_str = str(system)
        if system_str:
            openai_msgs.append({"role": "system", "content": system_str})

    openai_msgs.extend(convert_anthropic_messages(messages))

    openai_tools = []
    server_search = False
    for t in tools:
        if t.get("type") in {"web_search_20250305", "web_search", "web_search_preview"}:
            # 服务端搜索工具必须由上游联网检索执行，不能作为客户端 function
            # tool 转发：转发只会换回一个 tool_use 声明块，调用方拿不到来源，
            # 而 Anthropic 客户端要求服务端执行后的 web_search_tool_result。
            server_search = True
            continue
        if t.get("type") == "function":
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", "NO DESCRIPTION"),
                        "parameters": t.get("input_schema", {}),
                    },
                }
            )
        elif "name" in t:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", t.get("parameters", {})),
                    },
                }
            )

    output_config = body.get("output_config")
    if (
        isinstance(output_config, dict)
        and output_config.get("format", {}).get("type") == "json_schema"
    ):
        json_schema = output_config["format"].get("schema")
        if json_schema:
            openai_msgs.insert(
                0,
                {
                    "role": "system",
                    "content": f"You MUST return valid JSON adhering strictly to this JSON Schema:\n{json.dumps(json_schema)}",
                },
            )

    req_model = body.get("model")
    search_sink = []
    if stream:
        # 流式路径下搜索结果与正文交错到达，content_block 索引在开流时即已固定，
        # 无法在响应头发出后追加结果块；需要结构化来源的调用方应走非流式路径。
        # 此处仍透传 server_search，确保上游确实执行联网检索。
        return await handle_chat(
            openai_msgs,
            model,
            thinking,
            server_search,
            True,
            openai_tools or None,
            is_anthropic=True,
            req_model=req_model,
            scope=get_api_key(request),
            search_sink=search_sink,
        )

    result = await handle_chat(
        openai_msgs,
        model,
        thinking,
        server_search,
        False,
        openai_tools or None,
        is_anthropic=True,
        req_model=req_model,
        scope=get_api_key(request),
        search_sink=search_sink,
    )
    if not isinstance(result, dict) or "choices" not in result:
        return result
    return format_anthropic_response(result, req_model, search_results=search_sink)


@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)

    # Declared limits reflect OBSERVED DeepSeek web behavior (issue #22), not
    # guaranteed upstream API limits: a single first message passes up to ~1M
    # tokens, remembered in-session context reaches ~393K input tokens before
    # the bridge summarizes and rolls the conversation into a fresh chat, and
    # observed per-response output is ~4,000-8,192 tokens.
    base_models = [
        {
            "id": SINGLE_MODEL,
            "object": "model",
            "type": "model",
            "name": SINGLE_MODEL,
            "display_name": "DeepSeek V4.1 Flash",
            "created": 1785456000,
            "created_at": "2026-07-31T00:00:00Z",
            "owned_by": "deeperseeker",
            "context_window": context_window_tokens(),
            "max_output_tokens": max_output_tokens(),
            # OpenAI-compatible modality declaration
            "input_modalities": ["text", "image", "pdf"],
            "output_modalities": ["text"],
            "capabilities": {
                "batch": {"supported": True},
                "code_execution": {"supported": True},
                "image_input": {"supported": True},
                "pdf_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {
                    "supported": True,
                    "types": {
                        "enabled": {"supported": True},
                        "adaptive": {"supported": True},
                    },
                },
                "effort": {
                    "supported": True,
                    "low": {"supported": True},
                    "medium": {"supported": True},
                },
                "context_management": {
                    "clear_thinking_20251015": {"supported": True},
                    "compact_20260112": {"supported": True},
                    "supported": True,
                },
            },
        }
    ]

    # Single model, plus the anthropic/claude-* alias for Claude Desktop
    # auto-discovery. Both IDs serve the same upstream v4.1flash model.
    claude_aliases = []
    for m in base_models:
        alias = dict(m)
        alias["id"] = f"anthropic/claude-{m['id']}"
        alias["name"] = f"anthropic/claude-{m['name']}"
        alias["display_name"] = f"Claude {m['display_name']}"
        claude_aliases.append(alias)

    all_models = base_models + claude_aliases

    return {
        "object": "list",
        "data": all_models,
        "has_more": False,
        "first_id": all_models[0]["id"],
        "last_id": all_models[-1]["id"],
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


def _prune_admin_sessions():
    """B12：过期的后台会话此前会驻留内存直到重启，此处改为登录时顺带清理，
    使该字典只保留仍有效的会话。"""
    now = time.time()
    expired = [sid for sid, ts in SESSIONS.items() if now - ts > SESSION_TTL]
    for sid in expired:
        SESSIONS.pop(sid, None)


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")
    _prune_admin_sessions()
    if time.time() < _login_fails["locked_until"]:
        return templates.TemplateResponse(
            request, "login.html", {"error": "登录尝试次数过多，请 5 分钟后再试。"}
        )
    if secrets.compare_digest(
        username.encode("utf-8"), ADMIN_USER.encode("utf-8")
    ) and secrets.compare_digest(
        password.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8")
    ):
        _login_fails["count"] = 0
        sid = str(uuid.uuid4())
        SESSIONS[sid] = time.time()
        resp = HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")
        resp.set_cookie("session_id", sid, httponly=True, samesite="lax")
        return resp
    _login_fails["count"] += 1
    if _login_fails["count"] >= 5:
        _login_fails["locked_until"] = time.time() + 300
        _login_fails["count"] = 0
    return templates.TemplateResponse(
        request, "login.html", {"error": "用户名或密码错误。"}
    )


@app.get("/logout")
async def logout(request: Request):
    sid = request.cookies.get("session_id")
    SESSIONS.pop(sid, None)
    resp = HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    resp.delete_cookie("session_id")
    return resp


@app.get("/dashboard")
async def dashboard(request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    tokens = await _db(get_tokens)
    accounts = await _db(get_accounts)
    account_error = request.query_params.get("account_error")
    account_message = request.query_params.get("account_message")
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "tokens": tokens,
            "accounts": accounts,
            "account_error": account_error,
            "account_message": account_message,
            "token_check_state": _token_check_state,
            "check_interval": TOKEN_CHECK_INTERVAL,
        },
    )


# 图表降采样阈值：小时点超过这个数（约 30 天及以上区间）就退回按天聚合，
# 否则 Chart.js 要在页面上画上千根柱子，渲染会明显卡顿。
CHART_MAX_HOUR_POINTS = 720


def _chart_series(stats, from_hour):
    """挑图表的出点粒度，返回 (序列, "hour" / "day")。

    24 小时窗口逐小时出点才能还原峰谷两档；再长的区间按小时出点会把
    图挤成一团，超过 CHART_MAX_HOUR_POINTS 就改用逐日序列。
    """
    if from_hour and len(stats["hourly"]) <= CHART_MAX_HOUR_POINTS:
        return stats["hourly"], "hour"
    return stats["daily"], "day"


def _chart_from_stats(stats, from_hour):
    """把区间统计折成 Chart.js 需要的多序列。

    峰谷两档成本各自成列、叠成同一根柱，便于一眼看出高峰时段多花的钱；
    token 按输入（未缓存）/ 输出 / 缓存读拆三条折线走右轴，与顶部卡片的
    口径一致（缓存写不单独出线，它已计入区间总量）。

    命中率序列也在这里出：本函数才知道图表最终采用哪个粒度，模板不该
    自己再按分母算一遍。

    @param from_hour 区间是否为滚动 24 小时窗口（resolve_range 的小时下界
        非 None）。只有它才值得逐小时出点，其余区间一律按天。
    """
    series, granularity = _chart_series(stats, from_hour)
    if granularity == "hour":
        labels = [s["hour"][11:] + ":00" for s in series]
    else:
        labels = [s["day"][5:] for s in series]
    hit_rate = [round(v, 6) for v in hit_rate_series(series)]
    return {
        "labels": labels,
        "cost_peak": [round(s["cost_peak"], 6) for s in series],
        "cost_off_peak": [round(s["cost_off_peak"], 6) for s in series],
        "input": [s["cache_miss"] for s in series],
        "output": [s["output"] for s in series],
        "cache_read": [s["cache_hit"] for s in series],
        "cache_hit_rate": hit_rate,
        "granularity": granularity,
    }


@app.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    # resolve_range 读的是 parse_qs 形状（值一律是列表），QueryParams 直接转过去会
    # 拿到裸字符串，故在这里包一层。
    params = {k: [v] for k, v in request.query_params.items()}
    # 「全部」预设的起点只能从库里读：resolve_range 是纯函数，读不到库，
    # 所以先取最早一天再喂给它。
    earliest = await _db(earliest_day)
    from_day, to_day, from_hour, to_hour = resolve_range(params, earliest)
    if from_hour:
        rows = await _db(query_hours, from_hour, to_hour)
    else:
        rows = await _db(query_hours, from_day + "T00", to_day + "T23")
    # 号池成员用别名显示，别名为空才退回 "Token #id"。
    tokens = await _db(get_tokens)
    labels = {
        "tokens": {t["id"]: (t["alias"] or f"Token #{t['id']}") for t in tokens},
        "keys": {},
    }
    stats = aggregate(rows, from_day, to_day, labels)
    return templates.TemplateResponse(
        request,
        "stats.html",
        {
            "stats": stats,
            "chart": _chart_from_stats(stats, bool(from_hour)),
            "heatmap": build_heatmap(stats["daily"], from_day, to_day),
            "range_preset": params.get("range", ["30"])[0],
            "range_labels": RANGE_LABELS,
        },
    )


@app.post("/tokens/add")
async def tokens_add(request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    form = await request.form()
    auth_token = form.get("auth_token", "").strip().strip("'\"")
    alias = form.get("alias", "").strip() or None
    if auth_token:
        await _db(add_token, auth_token, alias)
    return HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")


@app.post("/accounts/add")
async def accounts_add(request: Request):
    """Log in an account and add only its returned bearer token to the pool."""
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    form = await request.form()
    email = str(form.get("email", "")).strip()
    password = str(form.get("password", ""))
    alias = str(form.get("alias", "")).strip() or None
    if not email or not password:
        return RedirectResponse(
            "/dashboard?account_error=" + quote("邮箱和密码不能为空"), status_code=303
        )
    try:
        auth_token = await login_deepseek_account(email, password)
        save_account_login(email, alias, auth_token)
        return RedirectResponse(
            "/dashboard?account_message=" + quote("账号登录成功，已加入 Token 号池"),
            status_code=303,
        )
    except Exception as e:
        logger.warning("DeepSeek account login failed for %s: %s", email, str(e)[:300])
        return RedirectResponse(
            "/dashboard?account_error=" + quote("登录失败：" + str(e)[:180]),
            status_code=303,
        )


@app.post("/accounts/{account_id}/delete")
async def accounts_delete(account_id: int, request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    delete_account(account_id)
    return RedirectResponse(
        "/dashboard?account_message=" + quote("账号及其 Token 已删除"), status_code=303
    )


@app.post("/tokens/check")
async def tokens_check(request: Request, token_id: int | None = None):
    """Run a manual check for one token or the complete token pool."""
    try:
        get_current_admin(request)
    except HTTPException:
        return JSONResponse({"error": "未登录或登录已过期"}, status_code=401)
    if token_id is None:
        results = await _run_token_checks()
    else:
        async with _token_check_lock:
            results = [await check_token_status(token_id)]
            _token_check_state["last_results"] = results
            _token_check_state["last_run_at"] = time.time()
    # The page reloads to show updated rows; never echo raw bearer tokens in JSON.
    return JSONResponse(
        {"results": results, "checked_at": _token_check_state["last_run_at"]}
    )


@app.post("/tokens/{token_id}/delete")
async def tokens_delete(token_id: int, request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    await _db(delete_token, token_id)
    return HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return await dashboard(request)


@app.get("/health")
async def health(request: Request):
    active = sum(1 for t in await _db(get_tokens) if t["status"] == "ACTIVE")
    cookies_valid = False
    try:
        with open(cookie_file_path()) as f:
            c = json.load(f)
        exp = c.get("expiry")
        cookies_valid = bool(exp and exp > time.time())
    except Exception:
        cookies_valid = False
    # WAF cookies are deprecated in favor of Android headers (which do not require cookies).
    # Token presence determines service readiness; cookie status is kept for backup visibility.
    ok = active > 0
    data = {"status": "ok" if ok else "degraded"}
    if check_key(request):
        data["active_tokens"] = active
        data["cookies_valid"] = cookies_valid
    return JSONResponse(data, status_code=200 if ok else 503)


def main():
    """Entry point for the console script."""
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "4000")),
    )


if __name__ == "__main__":
    main()
