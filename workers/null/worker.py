import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
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

# Safety cap: if the user's queue consumer never calls /release, the
# reservation is auto-released after this many seconds so a forgotten /release
# can't pin a worker indefinitely. Override with MAX_RESERVATION_SECONDS.
MAX_RESERVATION_SECONDS = float(os.environ.get("MAX_RESERVATION_SECONDS", 3600))

# Marker the benchmark path sets so the same remote function can return
# immediately during capacity estimation instead of sleeping.
BENCHMARK_SENTINEL = "__null_worker_benchmark__"

# Internal control server. Hosts:
#   * POST /release  — always available, marks the active reservation as
#     done so the held /reserve returns 200 (success in metrics, not a
#     cancellation).
#   * GET  /health   — only when no external BACKEND_HEALTH_URL is set; the
#     framework's healthcheck loop polls it so the worker has a live signal.
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


# Singleton active reservation. `allow_parallel_requests=False` on the
# /reserve handler guarantees the framework only runs one at a time per
# worker, so a single slot is enough.
_active_reservation: Optional[asyncio.Event] = None


def _build_internal_app() -> web.Application:
    app = web.Application()

    async def release_handler(_request: web.Request) -> web.Response:
        event = _active_reservation
        if event is None:
            return web.json_response(
                {"released": False, "reason": "no active reservation"},
                status=200,
            )
        event.set()
        return web.json_response({"released": True}, status=200)

    app.router.add_post("/release", release_handler)

    if USE_STUB_HEALTH:
        async def stub_health(_request: web.Request) -> web.Response:
            return web.Response(status=200, text="ok")

        app.router.add_get(STUB_HEALTH_PATH, stub_health)

    return app


@asynccontextmanager
async def null_lifecycle():
    # Pin max_throughput to exactly 100 by pre-populating the framework's
    # benchmark cache file. The framework's __run_benchmark short-circuits
    # to `float(file_contents)` when this file exists, bypassing the
    # time-based calculation that would otherwise drift to ~99.x due to
    # asyncio scheduling overhead. The filename matches the framework
    # constant BENCHMARK_INDICATOR_FILE in
    # vastai.serverless.server.lib.backend.
    try:
        with open(".has_benchmark", "w") as fh:
            fh.write("100")
    except OSError as e:
        log.warning(f"Could not pin benchmark cache to 100: {e}")

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
    log.info("\n".join(lines))

    try:
        yield
    finally:
        await runner.cleanup()


async def reserve_worker(**params: object) -> dict:
    global _active_reservation

    if params.get(BENCHMARK_SENTINEL):
        # The framework computes max_throughput = workload / time during the
        # startup benchmark. A null worker has no real throughput concept,
        # so we deliberately take ~1s with workload=100 to pin
        # max_throughput to ~100. Without this the near-instant benchmark
        # would report hundreds of thousands of workload/sec, distorting
        # any downstream capacity math.
        await asyncio.sleep(1.0)
        return {"ok": True, "benchmark": True}

    requested = params.get("duration")
    if requested is None:
        duration = MAX_RESERVATION_SECONDS
    else:
        try:
            duration = max(0.0, min(float(requested), MAX_RESERVATION_SECONDS))
        except (TypeError, ValueError):
            duration = MAX_RESERVATION_SECONDS

    event = asyncio.Event()
    _active_reservation = event
    log.info(
        f"Reservation acquired; awaiting POST /release on "
        f"http://{INTERNAL_HOST}:{INTERNAL_PORT}/release "
        f"(auto-release after {duration:.1f}s)"
    )
    try:
        try:
            await asyncio.wait_for(event.wait(), timeout=duration)
            log.info("Reservation released via /release")
            return {"released": "explicit", "duration_cap": duration}
        except asyncio.TimeoutError:
            log.warning(
                f"Reservation hit duration cap of {duration:.1f}s without "
                f"explicit /release; releasing automatically"
            )
            return {"released": "duration_elapsed", "duration": duration}
    finally:
        if _active_reservation is event:
            _active_reservation = None


worker_config = WorkerConfig(
    model_server_url=HEALTH_BASE_URL,
    model_server_port=HEALTH_PORT,
    model_healthcheck_url=HEALTH_PATH,
    lifecycle=null_lifecycle(),
    handlers=[
        HandlerConfig(
            route="/reserve",
            allow_parallel_requests=False,
            # Reject (429) any /reserve that arrives while the worker is
            # already busy. A held reservation lasts up to MAX_RESERVATION_
            # SECONDS, so queueing behind it would mean hours of wait —
            # better to bounce the request immediately so serverless routes
            # it to a free worker (or spins up a new one).
            max_queue_time=0.0,
            remote_function=reserve_worker,
            workload_calculator=lambda _payload: 100.0,
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

Worker(worker_config).run()
