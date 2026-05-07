"""ComfyUI worker for the vast.ai PyWorker SDK.

Each worker runs a benchmark on warm-up. The payload is selected as follows:

  1. If ``misc/benchmark.json`` exists, it is used as a custom ComfyUI
     workflow (recommended: match the workflow your endpoint will actually
     serve, so the autoscaler's performance estimate is meaningful).
  2. Otherwise an SD1.5 Text2Image fallback runs, parameterised by the
     ``BENCHMARK_TEST_{WIDTH,HEIGHT,STEPS}`` env vars and a random prompt
     from ``misc/test_prompts.txt``.

``__RANDOM_INT__`` placeholders in custom workflows are substituted
server-side by ai-dock/comfyui-api-wrapper, so this worker does not handle
them itself.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# ComfyUI model configuration
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18288
MODEL_LOG_FILE             = '/var/log/portal/comfyui.log'
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# ComfyUI-specific log messages
MODEL_LOAD_LOG_MSG = [
    "To see the GUI go to: "
]

MODEL_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",
    "Value not in list: ",
    "[ERROR] Provisioning Script failed"
]

MODEL_INFO_LOG_MSGS = [
    '"message":"Downloading'
]

# Benchmark assets shipped alongside this worker. Resolved relative to this
# file so the worker keeps working regardless of the launch cwd.
MISC_DIR       = Path(__file__).parent / "misc"
BENCHMARK_FILE = MISC_DIR / "benchmark.json"
TEST_PROMPTS   = MISC_DIR / "test_prompts.txt"

log = logging.getLogger(__name__)


def _custom_workflow_payload() -> dict | None:
    """Build a payload from ``misc/benchmark.json``, or None if unavailable."""
    if not BENCHMARK_FILE.exists():
        return None
    try:
        with open(BENCHMARK_FILE) as f:
            workflow = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error("Failed to load %s: %s; falling back to default benchmark", BENCHMARK_FILE, e)
        return None
    log.info("Using custom benchmark workflow from %s", BENCHMARK_FILE)
    return {
        "input": {
            "request_id": f"test-{random.randint(1000, 99999)}",
            "workflow_json": workflow,
        }
    }


def _default_payload() -> dict:
    """Build the SD1.5 Text2Image fallback payload."""
    with open(TEST_PROMPTS) as f:
        prompts = [line.strip() for line in f if line.strip()]
    return {
        "input": {
            "request_id": f"test-{random.randint(1000, 99999)}",
            "modifier": "Text2Image",
            "modifications": {
                "prompt": random.choice(prompts),
                "width":  int(os.getenv("BENCHMARK_TEST_WIDTH",  512)),
                "height": int(os.getenv("BENCHMARK_TEST_HEIGHT", 512)),
                "steps":  int(os.getenv("BENCHMARK_TEST_STEPS",  20)),
                "seed":   random.randint(0, sys.maxsize),
            }
        }
    }


def make_benchmark_payload() -> dict:
    """Build one benchmark request payload.

    Called once per benchmark run by the SDK; using a generator (rather
    than a static ``dataset=``) lets each run re-pick a prompt and re-roll
    the seed, and avoids holding multiple copies of a large workflow JSON
    in memory.
    """
    return _custom_workflow_payload() or _default_payload()


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=10.0,
            benchmark_config=BenchmarkConfig(
                generator=make_benchmark_payload,
            )
        )
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

Worker(worker_config).run()
