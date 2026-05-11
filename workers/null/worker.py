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

# Safety cap: if a client never disconnects and never sets `duration`, the
# reservation is auto-released after this many seconds so a stuck client
# can't pin a worker indefinitely. Override with MAX_RESERVATION_SECONDS.
MAX_RESERVATION_SECONDS = float(os.environ.get("MAX_RESERVATION_SECONDS", 3600))

# Marker the benchmark path sets so the same remote function can return
# immediately during capacity estimation instead of sleeping.
BENCHMARK_SENTINEL = "__null_worker_benchmark__"

# Healthcheck wiring. The framework periodically GETs
# `<model_server_url>:<model_server_port><model_healthcheck_url>` and marks the
# worker errored if that ever fails after the first success. For the null
# worker we either:
#   * point at a URL the user supplies via BACKEND_HEALTH_URL — typically
#     their own queue-consumer's health endpoint, so the autoscaler sees the
#     worker as broken if the consumer dies, or
#   * run a tiny built-in stub that always returns 200, so the framework has
#     something live to talk to until the user wires up a real consumer.
BACKEND_HEALTH_URL = os.environ.get("BACKEND_HEALTH_URL", "").strip()
STUB_HEALTH_HOST = "127.0.0.1"
STUB_HEALTH_PORT = int(os.environ.get("NULL_STUB_HEALTH_PORT", 18999))
STUB_HEALTH_PATH = "/health"

if BACKEND_HEALTH_URL:
    _parsed = urlsplit(BACKEND_HEALTH_URL)
    if not _parsed.scheme or not _parsed.hostname:
        raise ValueError(
            f"BACKEND_HEALTH_URL must be an absolute URL, got: {BACKEND_HEALTH_URL!r}"
        )
    HEALTH_BASE_URL = f"{_parsed.scheme}://{_parsed.hostname}"
    HEALTH_PORT = _parsed.port or (443 if _parsed.scheme == "https" else 80)
    HEALTH_PATH = _parsed.path or "/"
    USE_STUB = False
else:
    HEALTH_BASE_URL = f"http://{STUB_HEALTH_HOST}"
    HEALTH_PORT = STUB_HEALTH_PORT
    HEALTH_PATH = STUB_HEALTH_PATH
    USE_STUB = True


@asynccontextmanager
async def null_lifecycle():
    runner = None
    if USE_STUB:
        async def stub_health(_request: web.Request) -> web.Response:
            return web.Response(status=200, text="ok")

        app = web.Application()
        app.router.add_get(STUB_HEALTH_PATH, stub_health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, STUB_HEALTH_HOST, STUB_HEALTH_PORT)
        await site.start()
        log.info(
            f"Null pyworker stub healthcheck listening on "
            f"http://{STUB_HEALTH_HOST}:{STUB_HEALTH_PORT}{STUB_HEALTH_PATH} "
            f"(override by setting BACKEND_HEALTH_URL)"
        )
    else:
        log.info(f"Null pyworker healthcheck pointing at {BACKEND_HEALTH_URL}")

    try:
        yield
    finally:
        if runner is not None:
            await runner.cleanup()


async def reserve_worker(**params: object) -> dict:
    if params.get(BENCHMARK_SENTINEL):
        return {"ok": True, "benchmark": True}

    requested = params.get("duration")
    if requested is None:
        duration = MAX_RESERVATION_SECONDS
    else:
        try:
            duration = max(0.0, min(float(requested), MAX_RESERVATION_SECONDS))
        except (TypeError, ValueError):
            duration = MAX_RESERVATION_SECONDS

    log.info(
        f"Reservation acquired; holding worker busy for up to {duration:.1f}s "
        f"(release early by disconnecting the HTTP request)"
    )
    try:
        await asyncio.sleep(duration)
        log.info("Reservation duration elapsed; releasing worker")
        return {"released": "duration_elapsed", "duration": duration}
    except asyncio.CancelledError:
        log.info("Reservation released by client disconnect")
        raise


worker_config = WorkerConfig(
    model_server_url=HEALTH_BASE_URL,
    model_server_port=HEALTH_PORT,
    model_healthcheck_url=HEALTH_PATH,
    lifecycle=null_lifecycle(),
    handlers=[
        HandlerConfig(
            route="/reserve",
            allow_parallel_requests=False,
            max_queue_time=30.0,
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
