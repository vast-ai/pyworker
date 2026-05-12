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
DEFAULT_SESSION_COST = 100  # matches the worker's max_perf


async def reserve(
    client: Serverless,
    *,
    endpoint_name: str,
    hold_for: float,
    session_cost: int,
    label: str = "session",
) -> None:
    endpoint = await client.get_endpoint(name=endpoint_name)
    lifetime = hold_for + 60  # outlast the hold; no keepalives sent
    start = time.monotonic()
    log.info("[%s] creating session (cost=%d, hold=%.0fs)", label, session_cost, hold_for)
    async with await endpoint.session(cost=session_cost, lifetime=lifetime) as s:
        log.info("[%s] session %s open", label, s.session_id)
        try:
            await asyncio.sleep(hold_for)
        except asyncio.CancelledError:
            log.info("[%s] cancelled after %.1fs", label, time.monotonic() - start)
            raise
    log.info("[%s] closed cleanly after %.1fs", label, time.monotonic() - start)


async def run_demo(
    client: Serverless,
    *,
    endpoint_name: str,
    interval: float,
    plateau: float,
    session_cost: int,
) -> None:
    n = 3
    hold = (n - 1) * interval + plateau
    tasks: list[asyncio.Task] = []
    for i in range(1, n + 1):
        label = f"res-{i}"
        tasks.append(asyncio.create_task(
            reserve(client, endpoint_name=endpoint_name, hold_for=hold,
                    session_cost=session_cost, label=label),
            name=label,
        ))
        if i < n:
            await asyncio.sleep(interval)
    log.info(
        "All %d sessions in flight; plateau %.0fs, scale-down %.0fs apart",
        n, plateau, interval,
    )
    await asyncio.gather(*tasks, return_exceptions=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vast Null PyWorker demo client")
    p.add_argument("--endpoint", default=os.environ.get("VAST_ENDPOINT", ENDPOINT_NAME),
                   help=f"endpoint name (default: {ENDPOINT_NAME})")
    p.add_argument("--instance", choices=("prod", "alpha", "candidate", "local"),
                   default=os.environ.get("VAST_INSTANCE", "prod"),
                   help="serverless instance (default: prod)")
    p.add_argument("--duration", type=float, default=180.0,
                   help="single-reserve mode: seconds to hold (default: 180)")

    modes = p.add_mutually_exclusive_group(required=False)
    modes.add_argument("--reserve", action="store_true",
                       help="single session (default if no mode given)")
    modes.add_argument("--demo", action="store_true",
                       help="staggered 3-session trapezoid")

    p.add_argument("--interval", type=float, default=30.0,
                   help="demo: seconds between sessions (default: 30)")
    p.add_argument("--plateau", type=float, default=300.0,
                   help="demo: seconds to hold all 3 active (default: 300)")
    p.add_argument("--session-cost", type=int, default=DEFAULT_SESSION_COST,
                   help=f"cost reported at session-create (default: {DEFAULT_SESSION_COST})")
    return p


async def main_async():
    args = build_arg_parser().parse_args()
    print("=" * 60)
    print(f"Endpoint: {args.endpoint}  (instance: {args.instance})")
    print("=" * 60)

    try:
        async with Serverless(instance=args.instance) as client:
            if args.demo:
                await run_demo(client, endpoint_name=args.endpoint,
                               interval=args.interval, plateau=args.plateau,
                               session_cost=args.session_cost)
            else:
                await reserve(client, endpoint_name=args.endpoint,
                              hold_for=args.duration,
                              session_cost=args.session_cost,
                              label="reservation")
    except KeyboardInterrupt:
        log.info("Interrupted; dropping in-flight sessions")
    except Exception as e:
        log.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
