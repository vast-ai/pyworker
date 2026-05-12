import argparse
import asyncio
import logging
import os
import sys

from vastai import Serverless

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


async def reserve(client: Serverless, endpoint_name: str, hold: float, cost: int, label: str):
    endpoint = await client.get_endpoint(name=endpoint_name)
    async with await endpoint.session(cost=cost, lifetime=hold + 60) as s:
        sid = s.session_id
        log.info("[%s] %s open, holding %.0fs", label, sid, hold)
        await asyncio.sleep(hold)
    log.info("[%s] %s closed", label, sid)


async def main_async():
    p = argparse.ArgumentParser(description="Vast Null PyWorker demo client")
    p.add_argument("--endpoint", default=os.environ.get("VAST_ENDPOINT", "null-prod"))
    p.add_argument("--instance", choices=("prod", "alpha", "candidate", "local"),
                   default=os.environ.get("VAST_INSTANCE", "prod"))
    p.add_argument("--count", type=int, default=1,
                   help="concurrent sessions to open (default: 1)")
    p.add_argument("--interval", type=float, default=30.0,
                   help="seconds between session starts when count>1 (default: 30)")
    p.add_argument("--hold", type=float, default=180.0,
                   help="seconds to hold each session (default: 180)")
    p.add_argument("--cost", type=int, default=100,
                   help="cost reported at session-create (default: 100)")
    args = p.parse_args()

    print(f"endpoint={args.endpoint} instance={args.instance} "
          f"count={args.count} hold={args.hold}s cost={args.cost}")

    try:
        async with Serverless(instance=args.instance) as client:
            tasks = []
            for i in range(args.count):
                label = f"res-{i+1}" if args.count > 1 else "reservation"
                tasks.append(asyncio.create_task(
                    reserve(client, args.endpoint, args.hold, args.cost, label),
                    name=label,
                ))
                if i + 1 < args.count:
                    await asyncio.sleep(args.interval)
            await asyncio.gather(*tasks, return_exceptions=True)
    except KeyboardInterrupt:
        log.info("Interrupted")
    except Exception as e:
        log.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
