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
        resp = await endpoint.request("/reserve", payload, cost=1)
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
    duration: float,
    interval: float,
) -> None:
    """Reserve, wait, reserve, wait, reserve, wait, cancel one.

    All three reservations run concurrently as separate held HTTP requests.
    After all three are in flight, we cancel the first to demonstrate the
    early-release path. The remaining two are left to run to their natural
    duration cap (or you can ctrl-c to drop them).
    """
    tasks: list[asyncio.Task] = []
    for i in range(1, 4):
        label = f"res-{i}"
        task = asyncio.create_task(
            reserve(
                client,
                endpoint_name=endpoint_name,
                duration=duration,
                label=label,
            ),
            name=label,
        )
        tasks.append(task)
        if i < 3:
            log.info("Waiting %.0fs before starting next reservation...", interval)
            await asyncio.sleep(interval)

    log.info(
        "All 3 reservations in flight. Waiting %.0fs, then cancelling res-1...",
        interval,
    )
    await asyncio.sleep(interval)

    log.info("Cancelling res-1 (drops the HTTP connection — produces a 499)")
    tasks[0].cancel()

    log.info(
        "res-2 and res-3 left running. They will end at their duration cap "
        "(%.0fs), or you can ctrl-c to drop them.",
        duration,
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
                    duration=args.duration,
                    interval=args.interval,
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
