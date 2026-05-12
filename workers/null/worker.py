import asyncio
import logging
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from aiohttp import web

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

log = logging.getLogger(__file__)

TARGET_PERF = 100.0
BENCHMARK_SENTINEL = "__null_worker_benchmark__"

INTERNAL_HOST = "127.0.0.1"
INTERNAL_PORT = int(os.environ.get("NULL_CONTROL_PORT", 18999))
STUB_HEALTH_PATH = "/health"

BACKEND_HEALTH_URL = os.environ.get("BACKEND_HEALTH_URL", "").strip()
if BACKEND_HEALTH_URL:
    _p = urlsplit(BACKEND_HEALTH_URL)
    if not _p.scheme or not _p.hostname:
        raise ValueError(f"BACKEND_HEALTH_URL must be absolute, got: {BACKEND_HEALTH_URL!r}")
    HEALTH_BASE_URL = f"{_p.scheme}://{_p.hostname}"
    HEALTH_PORT = _p.port or (443 if _p.scheme == "https" else 80)
    HEALTH_PATH = _p.path or "/"
    USE_STUB_HEALTH = False
else:
    HEALTH_BASE_URL = f"http://{INTERNAL_HOST}"
    HEALTH_PORT = INTERNAL_PORT
    HEALTH_PATH = STUB_HEALTH_PATH
    USE_STUB_HEALTH = True


_backend_ref: dict = {"backend": None}


def _build_internal_app() -> web.Application:
    app = web.Application()

    async def release_handler(_request: web.Request) -> web.Response:
        # Closes the singleton session. Uses name-mangled __close_session
        # to bypass the session_auth check — safe because this server is
        # bound to 127.0.0.1, and it spares the consumer from threading
        # session_auth through its queue.
        backend = _backend_ref.get("backend")
        if backend is None:
            return web.json_response({"released": False, "reason": "backend not ready"}, status=503)
        sids = list(backend.sessions.keys())
        if not sids:
            return web.json_response({"released": False, "reason": "no active session"}, status=200)
        closed = []
        for sid in sids:
            try:
                if await backend._Backend__close_session(sid):
                    closed.append(sid)
            except Exception as e:
                log.warning(f"Error closing session {sid}: {e}")
        return web.json_response({"released": bool(closed), "session_ids": closed}, status=200)

    app.router.add_post("/release", release_handler)

    if USE_STUB_HEALTH:
        async def stub_health(_request: web.Request) -> web.Response:
            return web.Response(status=200, text="ok")
        app.router.add_get(STUB_HEALTH_PATH, stub_health)

    return app


@asynccontextmanager
async def null_lifecycle():
    # Pin max_throughput to TARGET_PERF exactly — the framework's
    # __run_benchmark short-circuits to float(file_contents) if this exists.
    try:
        with open(".has_benchmark", "w") as fh:
            fh.write(str(int(TARGET_PERF)))
    except OSError as e:
        log.warning(f"Could not pin benchmark cache: {e}")

    runner = web.AppRunner(_build_internal_app())
    await runner.setup()
    await web.TCPSite(runner, INTERNAL_HOST, INTERNAL_PORT).start()

    log.info(
        "Null pyworker control server: http://%s:%d (POST /release%s)",
        INTERNAL_HOST,
        INTERNAL_PORT,
        f", GET {STUB_HEALTH_PATH}" if USE_STUB_HEALTH else "",
    )
    if not USE_STUB_HEALTH:
        log.info("Framework healthcheck → %s", BACKEND_HEALTH_URL)

    try:
        yield
    finally:
        await runner.cleanup()


async def ping(**params: object) -> dict:
    # Exists only to satisfy the framework's "at least one handler with a
    # BenchmarkConfig" requirement. Sleep 1s on the benchmark path as a
    # fallback in case the .has_benchmark cache pin failed; otherwise the
    # benchmark cache short-circuits and this never runs.
    if params.get(BENCHMARK_SENTINEL):
        await asyncio.sleep(1.0)
        return {"ok": True, "benchmark": True}
    return {"ok": True}


worker_config = WorkerConfig(
    model_server_url=HEALTH_BASE_URL,
    model_server_port=HEALTH_PORT,
    model_healthcheck_url=HEALTH_PATH,
    lifecycle=null_lifecycle(),
    max_sessions=1,
    handlers=[
        HandlerConfig(
            route="/ping",
            allow_parallel_requests=True,
            remote_function=ping,
            workload_calculator=lambda _payload: TARGET_PERF,
            benchmark_config=BenchmarkConfig(
                generator=lambda: {BENCHMARK_SENTINEL: True},
                runs=1,
                concurrency=1,
                do_warmup=False,
            ),
        ),
    ],
    log_action_config=LogActionConfig(),
)

_worker = Worker(worker_config)
_backend_ref["backend"] = _worker.backend
_worker.run()
