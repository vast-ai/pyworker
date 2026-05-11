import argparse
import asyncio
import logging
import os
import sys
import time

from vastai import Serverless

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

ENDPOINT_NAME = "null-prod"


async def reserve(
    client: Serverless,
    *,
    endpoint_name: str,
    duration: float,
    label: str = "reservation",
) -> dict:
    """Hold a Vast worker open for `duration` seconds (or until we disconnect).

    The worker counts itself busy for the lifetime of this call. Returning
    here means the reservation has ended — either /release was called on
    the worker's internal control port, or the duration cap fired, or the
    HTTP request was cancelled.
    """
    endpoint = await client.get_endpoint(name=endpoint_name)
    payload = {"duration": duration}
    start = time.monotonic()
    log.info("[%s] POST /reserve duration=%ss", label, duration)
    try:
        resp = await endpoint.request("/reserve", payload, cost=150)
        elapsed = time.monotonic() - start
        log.info("[%s] returned after %.1fs: %s", label, elapsed, resp.get("response"))
        return resp["response"]
    except asyncio.CancelledError:
        elapsed = time.monotonic() - start
        log.info("[%s] cancelled after %.1fs (HTTP connection dropped)", label, elapsed)
        raise


async def run_demo(
    client: Serverless,
    *,
    endpoint_name: str,
    interval: float,
    plateau: float,
) -> None:
    """Trapezoidal load: ramp up three reservations, plateau, then scale down.

    Start three reservations spaced `interval` seconds apart. Pick the
    duration so that the first release fires `plateau` seconds *after the
    last reservation started*, giving the autoscaler time to actually have
    all three workers running before any of them begin to scale down.
    Releases then fire `interval` seconds apart, matching the ramp-up.

    Each reservation ends via its duration cap (a 200 success).
    """
    n = 3
    hold = (n - 1) * interval + plateau
    tasks: list[asyncio.Task] = []
    for i in range(1, n + 1):
        label = f"res-{i}"
        log.info("[%s] starting (auto-release after %.0fs)", label, hold)
        task = asyncio.create_task(
            reserve(
                client,
                endpoint_name=endpoint_name,
                duration=hold,
                label=label,
            ),
            name=label,
        )
        tasks.append(task)
        if i < n:
            log.info("Waiting %.0fs before next reservation...", interval)
            await asyncio.sleep(interval)

    log.info(
        "All %d reservations in flight; holding plateau for %.0fs, "
        "then scaling down %.0fs apart",
        n,
        plateau,
        interval,
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for task, result in zip(tasks, results):
        log.info("[%s] final: %r", task.get_name(), result)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vast Null PyWorker demo client")
    p.add_argument(
        "--endpoint",
        default=os.environ.get("VAST_ENDPOINT", ENDPOINT_NAME),
        help=f"Vast endpoint name (default: {ENDPOINT_NAME})",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="Seconds to hold each worker busy (default: 180)",
    )

    modes = p.add_mutually_exclusive_group(required=False)
    modes.add_argument(
        "--reserve",
        action="store_true",
        help="Make a single /reserve call (default if no mode given)",
    )
    modes.add_argument(
        "--demo",
        action="store_true",
        help="Run the staggered 3-reservation demo, cancelling one mid-way",
    )

    p.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Demo mode: seconds between reservation steps (default: 30)",
    )
    p.add_argument(
        "--plateau",
        type=float,
        default=300.0,
        help=(
            "Demo mode: seconds to hold all 3 reservations active before "
            "scale-down starts. Gives the autoscaler time to fully spin "
            "up the third worker (default: 300)"
        ),
    )
    return p


async def main_async():
    args = build_arg_parser().parse_args()

    print("=" * 60)
    print(f"Endpoint: {args.endpoint}")
    print("=" * 60)

    try:
        async with Serverless() as client:
            if args.demo:
                await run_demo(
                    client,
                    endpoint_name=args.endpoint,
                    interval=args.interval,
                    plateau=args.plateau,
                )
            else:
                response = await reserve(
                    client,
                    endpoint_name=args.endpoint,
                    duration=args.duration,
                    label="reservation",
                )
                print(f"Reservation result: {response}")
    except KeyboardInterrupt:
        log.info("Interrupted; dropping any in-flight reservations")
    except Exception as e:
        log.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
