import asyncio
import logging
import os
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def null_lifecycle():
    log.info("Null pyworker active (no model server)")
    try:
        yield
    finally:
        log.info("Null pyworker shutting down")


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
    model_server_url="http://127.0.0.1",
    model_server_port=1,
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
