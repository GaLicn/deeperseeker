FROM python:3.13-slim


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app


# ca-certificates 必须显式安装：下面的源端点改为 HTTPS，而 --no-install-recommends
# 不会随 curl 带入该包，缺失时 apt 无法完成 TLS 握手。
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# 本机代理转发 deb.debian.org 时会出现 500/502 与连接重置，导致
# `playwright install --with-deps` 拉不齐 Debian 包并以退出码 100 失败。此处
# 提高 apt 的重试次数与超时，并把源端点由 HTTP 改为 HTTPS，规避代理对明文
# HTTP 的干扰。该配置对本文件后续所有 apt 调用（含 Playwright 的内部调用）生效。
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries
RUN echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries
RUN echo 'Acquire::https::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries
RUN for f in /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list; do \
      if [ -f "$f" ]; then sed -i 's|http://deb.debian.org|https://deb.debian.org|g' "$f" || true; fi; \
    done


# requirements.txt is not copied: the install step below uses the self-contained
# waf-backup lock, which already pins every base dependency.
COPY pyproject.toml uv.lock requirements-waf-backup.txt ./


# Upstream switched this step to `uv sync`, which installs only the default
# dependency group. The local login risk check still needs headless Chromium,
# so install the self-contained waf-backup superset (base deps + playwright)
# instead of the plain lock, then fetch the matching browser build.
RUN uv pip install --system --no-cache -r requirements-waf-backup.txt

# Headless Chromium is required at runtime: the DeepSeek login risk check only
# accepts the device ID published by the web client's fingerprint SDK. The
# browser install pulls dozens of Debian packages through the same unstable
# proxy as the base image, so it is retried; if the mirrors stay unreachable it
# is skipped with a warning, because only the optional Playwright fallback needs
# Chromium while the primary Android-header login path does not.
RUN ok=0; for attempt in 1 2 3; do \
      if playwright install --with-deps chromium; then ok=1; break; fi; \
      echo "playwright install attempt $attempt failed; retrying in 10s"; \
      sleep 10; \
    done; \
    if [ "$ok" = "1" ]; then echo "chromium ready"; \
    else echo "WARN: chromium unavailable (mirror unreachable); Playwright fallback disabled"; fi

COPY . .


RUN mkdir -p /app/data && \
    ln -sf /app/data/deeperseeker.db /app/deeperseeker.db && \
    ln -sf /app/data/aws_cookies_deepseek.json /app/aws_cookies_deepseek.json

EXPOSE 4000

ENV DB_PATH=/app/data/deeperseeker.db
ENV DEEPSEEKER_COOKIE_PATH=/app/data/aws_cookies_deepseek.json
ENV HOST=0.0.0.0

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -sf http://localhost:4000/health || exit 1

# If reverting to backup Playwright cookie generation, run under xvfb:
# CMD ["sh", "-c", "xvfb-run -a -s '-screen 0 1280x720x24' python3 app.py"]
# might also need requirements-waf-backup edition
# Upstream runs `uv sync`, which also installs this project and creates the
# `deeperseeker` console script. This image installs dependencies only (from the
# waf-backup lock), so invoke the entrypoint module directly instead.
CMD ["python3", "app.py"]
