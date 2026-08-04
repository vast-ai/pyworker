"""Shared core for the OpenAI-compatible workers (vllm/sglang/llama/openai).

They all proxy the same /v1/completions + /v1/chat/completions API, so the logic lives
here and the per-engine adapters just pass an EngineDefaults. Every default is
env-overridable: the image is version-locked to the engine, so it owns the
engine/version-specific values (log path, health endpoint, log grammar)."""

import os
import random
from dataclasses import dataclass, field
from typing import List

import nltk

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig


def _env_lines(name, default):
    """Newline-delimited env var -> list of stripped lines; default if unset/empty."""
    raw = os.environ.get(name)
    return [s for ln in raw.splitlines() if (s := ln.strip())] if raw else default


# One template serves both lanes; on-demand templates set only the engine var, so the
# benchmark recovers the model id from it. Only one is ever set.
_MODEL_NAME_VARS = ("MODEL_NAME", "VLLM_MODEL", "SGLANG_MODEL", "LLAMA_MODEL")


def _resolve_model_name():
    return next((v for var in _MODEL_NAME_VARS if (v := os.environ.get(var))), None)


@dataclass(frozen=True)
class EngineDefaults:
    """Per-engine baked defaults; each is overridden by the matching env var if set."""

    name: str                 # engine id for the startup banner
    model_log_file: str       # MODEL_LOG
    load_log_msgs: List[str]  # MODEL_LOAD_LOG_MSG — model-loaded markers
    error_log_msgs: List[str]  # MODEL_ERROR_LOG_MSGS — failed-load markers
    info_log_msgs: List[str] = field(default_factory=lambda: ['"message":"Download'])  # MODEL_INFO_LOG_MSGS


MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()


def request_parser(request):
    return request["input"] if request.get("input") is not None else request


def completions_benchmark_generator() -> dict:
    model = _resolve_model_name()
    if not model:
        raise ValueError("No model set: MODEL_NAME / VLLM_MODEL / SGLANG_MODEL / LLAMA_MODEL all empty")
    prompt = " ".join(random.choices(WORD_LIST, k=250))
    return {"model": model, "prompt": prompt, "temperature": 0.7, "max_tokens": 500}


def run(defaults: EngineDefaults) -> None:
    """Build the WorkerConfig from defaults (env-overridable) and run the worker."""

    # BACKEND=openai aliases vllm, so report the real engine and note the alias.
    backend = os.environ.get("BACKEND")
    alias = f" (BACKEND={backend})" if backend and backend != defaults.name else ""
    print(f"Using worker backend: {defaults.name}{alias}", flush=True)

    # Relative path resolves against the server url+port; a full URL is used as-is.
    healthcheck_url = os.environ.get("MODEL_HEALTH_ENDPOINT", "/health")

    config = dict(
        model_server_url=MODEL_SERVER_URL,
        model_server_port=MODEL_SERVER_PORT,
        model_log_file=os.environ.get("MODEL_LOG", defaults.model_log_file),
        model_healthcheck_url=healthcheck_url,
        handlers=[
            HandlerConfig(
                route="/v1/completions",
                workload_calculator=lambda data: data.get("max_tokens", 0),
                allow_parallel_requests=True,
                request_parser=request_parser,
                max_queue_time=600.0,
                benchmark_config=BenchmarkConfig(
                    generator=completions_benchmark_generator, concurrency=10, runs=3
                ),
            ),
            HandlerConfig(
                route="/v1/chat/completions",
                workload_calculator=lambda data: data.get("max_tokens", 0),
                allow_parallel_requests=True,
                request_parser=request_parser,
                max_queue_time=600.0,
            ),
        ],
        log_action_config=LogActionConfig(
            on_load=_env_lines("MODEL_LOAD_LOG_MSG", defaults.load_log_msgs),
            on_error=_env_lines("MODEL_ERROR_LOG_MSGS", defaults.error_log_msgs),
            on_info=_env_lines("MODEL_INFO_LOG_MSGS", defaults.info_log_msgs),
        ),
    )
    Worker(WorkerConfig(**config)).run()
