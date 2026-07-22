import inspect
import math
import nltk
import random
import os

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig


def _env_lines(name, default):
    """A newline-delimited env var -> list[str], or `default` if unset/empty. Each line
    is stripped (a YAML block-scalar / heredoc value can carry leading indentation, and
    the log grammar is substring-matched, so an unstripped pattern would never match).
    Lets the image (base-image) supply per-backend log grammar; absent -> default."""
    raw = os.environ.get(name)
    return [s for ln in raw.splitlines() if (s := ln.strip())] if raw else default


def _env_float(name, default):
    """A POSITIVE-float env var with a safe fallback. A malformed, non-finite, or
    non-positive value -> default: these feed asyncio timeouts, so 0/negative would
    instantly fail readiness and inf/nan would hang it."""
    try:
        v = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) and v > 0 else float(default)


# Per-worker configuration. Every value is env-overridable with the previous
# hardcoded value as the default, so the image (base-image) can bake per-backend
# values while absent-env reproduces today's behaviour exactly. See CON-1612.
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18000
MODEL_LOG_FILE             = os.environ.get("MODEL_LOG", "/var/log/portal/vllm.log")
# `MODEL_HEALTH_ENDPOINT` is the established env var (the framework reads it too, and
# the vLLM serverless templates set it). A path resolves against MODEL_SERVER_URL; a
# full URL is used as-is (the worker venv is built fresh, so aiohttp is current).
MODEL_HEALTHCHECK_ENDPOINT = os.environ.get("MODEL_HEALTH_ENDPOINT", "/health")

# Log-action grammar — vLLM defaults, overridable per backend by the image.
MODEL_LOAD_LOG_MSG = _env_lines("MODEL_LOAD_LOG_MSG", [
    "Application startup complete.",
])

MODEL_ERROR_LOG_MSGS = _env_lines("MODEL_ERROR_LOG_MSGS", [
    "INFO exited: vllm",
    "RuntimeError: Engine",
    "Traceback (most recent call last):",
])

MODEL_INFO_LOG_MSGS = _env_lines("MODEL_INFO_LOG_MSGS", [
    '"message":"Download',
])

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

def request_parser(request):
    data = request
    if request.get("input") is not None:
        data = request.get("input")
    return data


def completions_benchmark_generator() -> dict:
    prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
    model = os.environ.get("MODEL_NAME")
    if not model:
        raise ValueError("MODEL_NAME environment variable not set")

    benchmark_data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 500,
    }

    return benchmark_data

_config = dict(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/v1/completions",
            workload_calculator= lambda data: data.get("max_tokens", 0),
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
            benchmark_config=BenchmarkConfig(
                generator=completions_benchmark_generator,
                concurrency=10,
                runs=3
            )
        ),
        HandlerConfig(
            route="/v1/chat/completions",
            workload_calculator= lambda data: data.get("max_tokens", 0),
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
        )
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

# Readiness config (health-gated readiness, vast-cli WS2). Feature-detected: only
# passed when the installed vastai supports it, so a new worker on an OLD framework
# degrades to log mode instead of a TypeError. Absent READINESS defaults to 'logs'
# (current behaviour) — nothing changes until an image/template sets it.
if "readiness" in inspect.signature(WorkerConfig).parameters:
    _config["readiness"] = os.environ.get("READINESS", "logs")
    _config["readiness_timeout"] = _env_float("READINESS_TIMEOUT", 300)
    _config["healthcheck_probe_timeout"] = _env_float("HEALTHCHECK_PROBE_TIMEOUT", 10)

Worker(WorkerConfig(**_config)).run()
