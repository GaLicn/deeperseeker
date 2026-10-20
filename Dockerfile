FROM python:3.13-slim


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian
ARG DEBIAN_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security

# 同时覆盖 Debian 主源与安全源，兼容 deb822 和传统 sources.list。
RUN set -eux; \
    printf 'Acquire::Retries "3";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' > /etc/apt/apt.conf.d/80-retries; \
    rm -f /etc/apt/apt.conf.d/docker-clean; \
    for source in /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list; do \
      if [ -f "$source" ]; then \
        sed -i \
          -e "s|https\?://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
          -e "s|https\?://security.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
          -e "s|https\?://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
          "$source"; \
      fi; \
    done

# 缓存索引和 DEB，重建时复用下载；缓存目录不会写入最终镜像。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# requirements.txt is not copied: the install step below uses the self-contained
# waf-backup lock, which already pins every base dependency.
COPY pyproject.toml uv.lock requirements-waf-backup.txt ./


# Upstream switched this step to `uv sync`, which installs only the default
# dependency group. The local login risk check still needs headless Chromium,
# so install the self-contained waf-backup superset (base deps + playwright)
# instead of the plain lock, then fetch the matching browser build.
RUN uv pip install --system --no-cache -r requirements-waf-backup.txt

# 分开安装系统依赖和浏览器，浏览器下载重试不会重复更新 APT 索引。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    playwright install-deps chromium

# 使用 Playwright 自带的下载重试；失败直接报错，避免镜像缺少登录所需浏览器。
RUN playwright install chromium

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
