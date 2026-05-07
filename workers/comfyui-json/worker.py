"""ComfyUI worker for the vast.ai PyWorker SDK.

Each worker runs a benchmark on warm-up. The payload is selected as follows:

  1. If ``misc/benchmark.json`` exists in the cloned worker tree, it is
     used as a custom ComfyUI workflow. Use this if you fork the repo and
     bake in your workflow.
  2. Else, if ``$BENCHMARK_JSON_PATH`` is set and points at a readable
     file, it is used. Use this from a provisioning script — provisioning
     runs before pyworker is cloned, so it cannot write into ``misc/``,
     but it can drop the workflow elsewhere (e.g. ``/workspace/``) and
     export this env var.
  3. Else, if the well-known path
     ``/opt/comfyui-api-wrapper/workflows/pyworker_benchmark.json`` exists,
     it is used. The vast.ai ComfyUI base image's ``convert-workflows.sh``
     maintains this as a symlink to the first provisioned workflow, so on
     that image no env var is needed.
  4. Otherwise an SD1.5 Text2Image fallback runs, parameterised by the
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

# Well-known location maintained by the vast.ai ComfyUI base image.
# convert-workflows.sh symlinks this to the first provisioned workflow,
# letting the base image work out-of-the-box without any env var.
WELLKNOWN_BENCHMARK = Path("/opt/comfyui-api-wrapper/workflows/pyworker_benchmark.json")

log = logging.getLogger(__name__)


def _resolve_benchmark_path() -> Path | None:
    """Return the path to the custom benchmark workflow, or None if absent.

    See module docstring for the precedence rule. A set-but-broken
    ``$BENCHMARK_JSON_PATH`` logs a warning then falls through to the
    well-known path, so a typo in the env var doesn't silently mask a
    provisioned benchmark sitting at the standard location.
    """
    if BENCHMARK_FILE.exists():
        return BENCHMARK_FILE
    env_path = os.getenv("BENCHMARK_JSON_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        log.warning("BENCHMARK_JSON_PATH=%s does not exist; trying fallbacks", path)
    if WELLKNOWN_BENCHMARK.exists():
        return WELLKNOWN_BENCHMARK
    return None


def _custom_workflow_payload() -> dict | None:
    """Build a payload from a custom benchmark workflow JSON, or None if unavailable."""
    path = _resolve_benchmark_path()
    if path is None:
        return None
    try:
        with open(path) as f:
            workflow = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error("Failed to load %s: %s; falling back to default benchmark", path, e)
        return None
    log.info("Using custom benchmark workflow from %s", path)
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
