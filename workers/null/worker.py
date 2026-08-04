"""Null PyWorker — PROOF OF CONCEPT.

Holds a Vast serverless reservation open so a customer's own queue consumer drives
autoscaling without forwarding work through Vast. See README.md and the design note.

Not GA. Open items, deliberately NOT worked around with private SDK internals:
  - RELEASE is the reserving client's job, via the public Session.close() (off-instance;
    the demo client's `async with session` already does this). The SDK exposes no public
    worker-side session close, so a box cannot end its own reservation without a private
    call — which this worker no longer makes. SDK ask: a public loopback-release primitive.
  - SESSION TTL renews only on a forwarded request; this worker forwards none, so the
    reservation lifetime must be sized by the client up front. SDK ask: a renew primitive
    — or use the hybrid dispatch model (README) where work rides Session.request().
  - THROUGHPUT is pinned via the SDK-internal `.has_benchmark` sentinel file, the only way
    to fix perf today. SDK ask: a public fixed-perf / skip-benchmark config field.
  - CREDENTIALS for the customer's workload are out of scope here: the consumer obtains its
    own, short-lived and in-memory (see README).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from aiohttp import web

from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig

log = logging.getLogger(__file__)

TARGET_PERF = 100.0
BENCHMARK_SENTINEL = "__null_worker_benchmark__"
INTERNAL_HOST = "127.0.0.1"
INTERNAL_PORT = int(os.environ.get("NULL_CONTROL_PORT", 18999))

# Point the framework healthcheck at the consumer's readiness endpoint via BACKEND_HEALTH_URL
# (recommended — a real signal). Absent, fall back to a loopback stub that always returns 200;
# that HIDES a wedged consumer and is PoC-only.
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
    HEALTH_PATH = "/health"
    USE_STUB_HEALTH = True


@asynccontextmanager
async def null_lifecycle():
    # Pin max_throughput to TARGET_PERF by writing the SDK-internal `.has_benchmark` sentinel
    # (float on first line), which __run_benchmark short-circuits on. This is the only way to
    # fix perf today; replace when the SDK exposes a public fixed-perf config (PoC coupling).
    try:
        with open(".has_benchmark", "w") as fh:
            fh.write(str(int(TARGET_PERF)))
    except OSError as e:
        log.warning("Could not pin benchmark: %s", e)

    runner = None
    if USE_STUB_HEALTH:
        log.warning(
            "BACKEND_HEALTH_URL unset — using a stub /health that always returns 200. "
            "This hides a wedged consumer; set BACKEND_HEALTH_URL to a real readiness endpoint."
        )
        app = web.Application()

        async def stub_health(_request: web.Request) -> web.Response:
            return web.Response(status=200, text="ok")

        app.router.add_get(HEALTH_PATH, stub_health)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, INTERNAL_HOST, INTERNAL_PORT).start()
        log.info("Stub health at http://%s:%d%s (PoC)", INTERNAL_HOST, INTERNAL_PORT, HEALTH_PATH)
    else:
        log.info("Framework healthcheck → %s", BACKEND_HEALTH_URL)

    try:
        yield
    finally:
        if runner is not None:
            await runner.cleanup()


async def ping(**params: object) -> dict:
    # Exists only to satisfy the framework's "one handler with a BenchmarkConfig" rule. The
    # benchmark path sleeps 1s as a fallback if the perf pin above didn't take.
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

Worker(worker_config).run()
