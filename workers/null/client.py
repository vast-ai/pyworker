import argparse
import asyncio
import logging
import os
import sys

from vastai import Serverless

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

ENDPOINT_NAME = "null-prod"


async def reserve(client: Serverless, *, endpoint_name: str, duration: float) -> dict:
    """Hold a Vast worker open for `duration` seconds (or until we disconnect).

    The worker counts itself busy for the lifetime of this call, so the
    autoscaler will keep it provisioned. Returning here means the reservation
    has ended — either the worker hit its duration cap or the request errored.
    """
    endpoint = await client.get_endpoint(name=endpoint_name)
    payload = {"duration": duration}
    log.info("POST /reserve duration=%ss", duration)
    resp = await endpoint.request("/reserve", payload, cost=100)
    return resp["response"]


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
        default=60.0,
        help="Seconds to hold the worker busy (default: 60)",
    )
    return p


async def main_async():
    args = build_arg_parser().parse_args()

    print("=" * 60)
    print(f"Reserving 1 worker on endpoint '{args.endpoint}' for {args.duration}s")
    print("=" * 60)

    try:
        async with Serverless() as client:
            response = await reserve(
                client=client,
                endpoint_name=args.endpoint,
                duration=args.duration,
            )
            print(f"Reservation result: {response}")
    except Exception as e:
        log.error("Error during reservation: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
