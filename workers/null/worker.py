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

# Performance value pinned in the benchmark cache; sent to autoscaler as
# max_perf. Standardized at 100 — the conventional default the rest of the
# serverless system expects.
TARGET_PERF = 100.0

# Marker the benchmark path sets so the fallback /ping path returns
# immediately during the framework's startup benchmark.
BENCHMARK_SENTINEL = "__null_worker_benchmark__"

# Internal control server. Hosts:
#   * POST /release  — releases the active reservation by closing the
#     singleton session on this worker. Called by the user's queue
#     consumer when its work is done.
#   * GET  /health   — only when BACKEND_HEALTH_URL is unset; gives the
#     framework's healthcheck loop something live to talk to.
# Bound to 127.0.0.1 so only processes on the instance can reach it.
INTERNAL_HOST = "127.0.0.1"
INTERNAL_PORT = int(os.environ.get("NULL_CONTROL_PORT", 18999))
STUB_HEALTH_PATH = "/health"

BACKEND_HEALTH_URL = os.environ.get("BACKEND_HEALTH_URL", "").strip()

if BACKEND_HEALTH_URL:
    _parsed = urlsplit(BACKEND_HEALTH_URL)
    if not _parsed.scheme or not _parsed.hostname:
        raise ValueError(
            f"BACKEND_HEALTH_URL must be an absolute URL, got: {BACKEND_HEALTH_URL!r}"
        )
    HEALTH_BASE_URL = f"{_parsed.scheme}://{_parsed.hostname}"
    HEALTH_PORT = _parsed.port or (443 if _parsed.scheme == "https" else 80)
    HEALTH_PATH = _parsed.path or "/"
    USE_STUB_HEALTH = False
else:
    HEALTH_BASE_URL = f"http://{INTERNAL_HOST}"
    HEALTH_PORT = INTERNAL_PORT
    HEALTH_PATH = STUB_HEALTH_PATH
    USE_STUB_HEALTH = True


# Stashed after Worker(...) is constructed so /release can reach the
# framework's session machinery. Dict so the lifecycle closure picks up
# the assignment that happens before .run().
_backend_ref: dict = {"backend": None}


def _build_internal_app() -> web.Application:
    app = web.Application()

    async def release_handler(_request: web.Request) -> web.Response:
        """End the active reservation (the singleton session on this worker).

        max_sessions=1 means at most one session is active here. We call
        the framework's internal __close_session via name-mangling to
        bypass the session_auth check that /session/end normally requires.
        That's intentional: this endpoint is localhost-only so trust is
        assumed, and the user's consumer can release without having to
        plumb session_auth through their queue.

        __close_session reports the session metrics as a success, fires
        on_close_route if configured, and pops the session from the dict.
        """
        backend = _backend_ref.get("backend")
        if backend is None:
            return web.json_response(
                {"released": False, "reason": "backend not ready"},
                status=503,
            )
        sids = list(backend.sessions.keys())
        if not sids:
            return web.json_response(
                {"released": False, "reason": "no active session"},
                status=200,
            )
        closed = []
        for sid in sids:
            try:
                if await backend._Backend__close_session(sid):
                    closed.append(sid)
            except Exception as e:
                log.warning(f"Error closing session {sid}: {e}")
        return web.json_response(
            {"released": bool(closed), "session_ids": closed},
            status=200,
        )

    app.router.add_post("/release", release_handler)

    if USE_STUB_HEALTH:
        async def stub_health(_request: web.Request) -> web.Response:
            return web.Response(status=200, text="ok")

        app.router.add_get(STUB_HEALTH_PATH, stub_health)

    return app


@asynccontextmanager
async def null_lifecycle():
    # Pin max_throughput to exactly TARGET_PERF by pre-populating the
    # framework's benchmark cache file. __run_benchmark short-circuits to
    # float(file_contents) when this file exists.
    try:
        with open(".has_benchmark", "w") as fh:
            fh.write(str(int(TARGET_PERF)))
    except OSError as e:
        log.warning(f"Could not pin benchmark cache: {e}")

    app = _build_internal_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, INTERNAL_HOST, INTERNAL_PORT)
    await site.start()

    lines = [
        f"Null pyworker internal control server: http://{INTERNAL_HOST}:{INTERNAL_PORT}",
        f"  POST /release  - end the active reservation (call from your queue consumer)",
    ]
    if USE_STUB_HEALTH:
        lines.append(
            f"  GET  {STUB_HEALTH_PATH}      - stub healthcheck (override with BACKEND_HEALTH_URL)"
        )
    else:
        lines.append(f"Framework healthcheck pointed at: {BACKEND_HEALTH_URL}")
    lines.append(
        "Reservations use the framework session model. Clients POST to "
        "/session/create via the SDK to acquire a worker; max_sessions=1 "
        "so each worker holds at most one reservation."
    )
    log.info("\n".join(lines))

    try:
        yield
    finally:
        await runner.cleanup()


async def ping(**params: object) -> dict:
    """Trivial handler. Exists to satisfy the framework's requirement that
    at least one HandlerConfig has a BenchmarkConfig, and to give clients
    a route they can hit with session_id to extend their session TTL.
    """
    if params.get(BENCHMARK_SENTINEL):
        # Fallback only — the lifecycle pre-pins .has_benchmark so
        # __run_benchmark normally short-circuits and this never runs. If
        # the cache write failed, sleep ~1s so the time-based throughput
        # math lands near TARGET_PERF.
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
