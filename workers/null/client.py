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
# Default cost passed to /session/create. 100 matches the worker's
# max_perf for clean unit-occupancy semantics: one session = one worker.
# If you hit autoscaler scale-up issues (queueing past the 2nd active
# worker), --session-cost 200 is a temporary over-provisioning workaround
# until the known autoscaler fixes land.
DEFAULT_SESSION_COST = 100


async def reserve(
    client: Serverless,
    *,
    endpoint_name: str,
    hold_for: float,
    session_cost: int,
    label: str = "session",
) -> None:
    """Open a session, hold the worker for `hold_for` seconds, close cleanly.

    Uses the framework's session model — each session counts as one worker
    occupied, but unlike a held HTTP request it isn't poisoning the
    worker's throughput math. max_sessions=1 on the worker side means a
    second /session/create against the same worker gets 429, so serverless
    routes the second reservation to a free worker or scales a new one up.
    """
    endpoint = await client.get_endpoint(name=endpoint_name)
    # Session lifetime must outlast the hold. The framework expires sessions
    # whose `expiration` (set to now + lifetime at creation) has passed; we
    # don't make any keepalive requests so no extension happens.
    lifetime = hold_for + 60
    start = time.monotonic()
    log.info(
        "[%s] creating session (cost=%d, lifetime=%.0fs, hold=%.0fs)",
        label, session_cost, lifetime, hold_for,
    )
    async with await endpoint.session(cost=session_cost, lifetime=lifetime) as s:
        log.info("[%s] session %s open", label, s.session_id)
        try:
            await asyncio.sleep(hold_for)
            log.info("[%s] hold complete, closing session", label)
        except asyncio.CancelledError:
            elapsed = time.monotonic() - start
            log.info("[%s] cancelled after %.1fs, closing session", label, elapsed)
            raise
    elapsed = time.monotonic() - start
    log.info("[%s] session closed cleanly after %.1fs", label, elapsed)


async def run_demo(
    client: Serverless,
    *,
    endpoint_name: str,
    interval: float,
    plateau: float,
    session_cost: int,
) -> None:
    """Trapezoidal load: ramp up three sessions, plateau, then scale down.

    Start three sessions spaced `interval` seconds apart. Each holds for
    `(n-1)*interval + plateau` seconds, so the first release fires
    `plateau` seconds after the last session started — giving the
    autoscaler time to actually have all three workers running before any
    scale-down begins. Releases then fire `interval` seconds apart,
    matching the ramp-up.

    Each session ends via the SDK's `session.close()` on `async with` exit,
    which posts to /session/end with proper auth — counted as a normal
    success in metrics.
    """
    n = 3
    hold = (n - 1) * interval + plateau
    tasks: list[asyncio.Task] = []
    for i in range(1, n + 1):
        label = f"res-{i}"
        log.info("[%s] starting (hold=%.0fs)", label, hold)
        task = asyncio.create_task(
            reserve(
                client,
                endpoint_name=endpoint_name,
                hold_for=hold,
                session_cost=session_cost,
                label=label,
            ),
            name=label,
        )
        tasks.append(task)
        if i < n:
            log.info("Waiting %.0fs before next session...", interval)
            await asyncio.sleep(interval)

    log.info(
        "All %d sessions in flight; holding plateau for %.0fs, "
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
        help="Single-reserve mode: seconds to hold the worker (default: 180)",
    )

    modes = p.add_mutually_exclusive_group(required=False)
    modes.add_argument(
        "--reserve",
        action="store_true",
        help="Make a single session (default if no mode given)",
    )
    modes.add_argument(
        "--demo",
        action="store_true",
        help="Run the staggered 3-reservation trapezoid demo",
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
    p.add_argument(
        "--session-cost",
        type=int,
        default=DEFAULT_SESSION_COST,
        help=(
            f"Cost reported to the autoscaler for each /session/create. "
            f"Setting this above the worker's max_perf (100) over-provisions "
            f"slightly, keeping an extra active worker warm so the next "
            f"session lands without queueing (default: {DEFAULT_SESSION_COST})"
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
                    session_cost=args.session_cost,
                )
            else:
                await reserve(
                    client,
                    endpoint_name=args.endpoint,
                    hold_for=args.duration,
                    session_cost=args.session_cost,
                    label="reservation",
                )
    except KeyboardInterrupt:
        log.info("Interrupted; dropping any in-flight sessions")
    except Exception as e:
        log.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
