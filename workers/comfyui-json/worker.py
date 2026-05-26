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
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# ComfyUI model configuration. The model server is ai-dock's
# comfyui-api-wrapper sitting in front of ComfyUI itself, not ComfyUI's
# own port (18188). We tail the api-wrapper's log rather than ComfyUI's
# and key off the api-wrapper's own structured readiness/fault signals:
#
#   BACKENDS_READY            — api-wrapper has confirmed every ComfyUI
#                               backend passes HTTP+WS probes. Until
#                               this fires, posting to /generate/sync
#                               can hit "Cannot connect to host" inside
#                               the api-wrapper, which the SDK can't
#                               recover from since __call_backend
#                               doesn't retry connection-refused.
#   BACKENDS_READY_TIMEOUT    — backends never reachable within
#                               api-wrapper's deadline. Worker is
#                               unrecoverable; mark errored.
#   BACKEND_UNRECOVERABLE     — CUDA fault / illegal memory access on a
#                               backend's GPU. Same fate.
#   Application startup failed — uvicorn's own ASGI lifespan failed.
#
# These tokens are emitted by ai-dock/comfyui-api-wrapper >= the
# "feat/backend-readiness-log-signals" change. Older wrappers won't
# emit BACKENDS_READY natively; for them, a background readiness shim
# (see _readiness_shim below) probes the stack and synthesises the
# token so warm-up triggers regardless of wrapper version.
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18288
MODEL_LOG_FILE             = '/var/log/portal/api-wrapper.log'
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# Trigger benchmark only after the full stack (api-wrapper + ComfyUI
# backends) is reachable. See BACKENDS_READY in the comment above.
MODEL_LOAD_LOG_MSG = [
    "BACKENDS_READY",
]

# LogAction.ModelError is fatal: the SDK calls backend_errored() and
# locks the worker into a permanent error state. Patterns must
# therefore only match conditions where the api-wrapper genuinely
# cannot serve any request — supervisord restarts on uvicorn exit, so
# a real failure self-heals rather than dragging the worker down.
#
# Notably *not* matched here:
#   - per-request errors (PreprocessWorker failures, ComfyUI workflow
#     validation, "Value not in list:") — one malformed client payload
#     would otherwise kill the worker
#   - "CUDA out of memory" — surfaces both as a misconfigured GPU
#     (which the benchmark-failure path already catches via
#     backend_errored) and as a too-greedy client request, which is
#     indistinguishable from a substring match
#   - convert-workflows.sh warnings — that script is not load-bearing
#     for serving
MODEL_ERROR_LOG_MSGS = [
    "BACKENDS_READY_TIMEOUT",       # backends never reachable
    "BACKEND_UNRECOVERABLE",        # CUDA fault latched per backend
    "Application startup failed",   # uvicorn ASGI lifespan startup failed
]

# LogAction.Info is purely informational (echoes log lines into the vast
# console). Nothing in api-wrapper.log is currently worth surfacing —
# model downloads are upstream in provisioning, per-request logs are
# too noisy.
MODEL_INFO_LOG_MSGS = []

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

# Used when test_prompts.txt is unreadable or empty. Bare and generic
# on purpose — this is a benchmark seed, not a creative output.
_FALLBACK_PROMPT = "a still life on a wooden table, soft daylight"


def _env_int(name: str, default: int) -> int:
    """Read an integer env var, warning + falling back on bad values."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("ignoring %s=%r (not an int); using default %d", name, raw, default)
        return default


def _try_load_workflow(path: Path) -> dict | None:
    """Load and return a benchmark workflow from ``path``.

    Returns None on any failure (path missing, not a regular file,
    unreadable, invalid JSON) so the caller can fall through to the
    next tier rather than dropping straight to the SD1.5 default.
    """
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to load %s: %s; trying next tier", path, e)
        return None


def _custom_workflow_payload() -> dict | None:
    """Try each benchmark workflow tier in order; return the first one
    that loads cleanly as a payload, or None if every tier is absent /
    unreadable. Tiers (in order): in-tree ``misc/benchmark.json``,
    ``$BENCHMARK_JSON_PATH``, well-known base-image symlink.
    """
    env_path = os.getenv("BENCHMARK_JSON_PATH")
    candidates = [("misc", BENCHMARK_FILE)]
    if env_path:
        candidates.append(("env", Path(env_path)))
    candidates.append(("well-known", WELLKNOWN_BENCHMARK))

    for label, path in candidates:
        # Surface a warning specifically when the operator pointed
        # BENCHMARK_JSON_PATH at something we can't use — silent
        # fall-through there is a footgun (typo => SD1.5 fallback,
        # operator wonders why custom benchmark didn't take).
        if not path.is_file():
            if label == "env":
                log.warning(
                    "BENCHMARK_JSON_PATH=%s is not a readable file; trying fallbacks", path
                )
            continue
        workflow = _try_load_workflow(path)
        if workflow is None:
            continue
        log.info("Using custom benchmark workflow from %s (%s)", path, label)
        return {
            "input": {
                "request_id": f"test-{random.randint(1000, 99999)}",
                "workflow_json": workflow,
            }
        }
    return None


def _load_prompts() -> list[str]:
    """Read misc/test_prompts.txt; defensive against missing/empty file."""
    try:
        with open(TEST_PROMPTS) as f:
            prompts = [line.strip() for line in f if line.strip()]
    except OSError as e:
        log.warning("could not read %s: %s; using built-in fallback prompt", TEST_PROMPTS, e)
        return [_FALLBACK_PROMPT]
    if not prompts:
        log.warning("%s is empty; using built-in fallback prompt", TEST_PROMPTS)
        return [_FALLBACK_PROMPT]
    return prompts


def _default_payload() -> dict:
    """Build the SD1.5 Text2Image fallback payload."""
    prompts = _load_prompts()
    return {
        "input": {
            "request_id": f"test-{random.randint(1000, 99999)}",
            "modifier": "Text2Image",
            "modifications": {
                "prompt": random.choice(prompts),
                "width":  _env_int("BENCHMARK_TEST_WIDTH",  512),
                "height": _env_int("BENCHMARK_TEST_HEIGHT", 512),
                "steps":  _env_int("BENCHMARK_TEST_STEPS",  20),
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


# --- Readiness shim ----------------------------------------------------
#
# Older api-wrapper versions (pre feat/backend-readiness-log-signals)
# never emit BACKENDS_READY, so the SDK's log tail would hang on warm-up
# indefinitely on those images. Forks of the base image can't be repinned
# for us, so we probe the stack ourselves and append BACKENDS_READY to
# the log the SDK is tailing once the stack is reachable end-to-end.
#
# On a current wrapper the real token appears first; ours is then a
# harmless duplicate. The probe re-implements the gate the new wrapper
# enforces (ComfyUI reachable AND api-wrapper /health 200) — the same
# end-to-end condition that motivated the move to BACKENDS_READY — so
# correctness is preserved on old wrappers and unchanged on new ones.
_COMFY_BACKEND_URL  = "http://127.0.0.1:18188/system_stats"
_WRAPPER_HEALTH_URL = f"{MODEL_SERVER_URL}:{MODEL_SERVER_PORT}{MODEL_HEALTHCHECK_ENDPOINT}"

# Upper bound roughly matches the api-wrapper's own backend-readiness
# deadline. If we exceed it, we deliberately do NOT write the token —
# the SDK's warm-up will time out naturally and surface a real failure
# rather than us papering over a stuck stack.
_READINESS_DEADLINE_S       = 600
_READINESS_PROBE_INTERVAL_S = 5

# Small head-start so Worker(...).run() has time to open its tail
# before we could plausibly write. Stack startup is many seconds even
# in the hot-cache case, so this only matters in pathological "already
# fully warm" restarts.
_READINESS_PROBE_GRACE_S = 2


def _probe(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError):
        return False


def _readiness_shim() -> None:
    time.sleep(_READINESS_PROBE_GRACE_S)
    deadline = time.monotonic() + _READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        if _probe(_COMFY_BACKEND_URL) and _probe(_WRAPPER_HEALTH_URL):
            try:
                # O_APPEND single-line writes are atomic under PIPE_BUF
                # on Linux, so no interleaving risk with the wrapper's
                # own writer.
                with open(MODEL_LOG_FILE, "a") as f:
                    f.write("BACKENDS_READY (synthesised by pyworker readiness shim)\n")
                log.info("readiness shim: stack reachable; emitted BACKENDS_READY")
            except OSError as e:
                log.warning("readiness shim: could not write to %s: %s", MODEL_LOG_FILE, e)
            return
        time.sleep(_READINESS_PROBE_INTERVAL_S)
    log.warning(
        "readiness shim: stack not reachable within %ds; "
        "letting SDK warm-up time out naturally",
        _READINESS_DEADLINE_S,
    )


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

threading.Thread(target=_readiness_shim, name="readiness-shim", daemon=True).start()

Worker(worker_config).run()
