# Vast PyWorker Examples

This repository contains **example PyWorkers** used by Vast.ai’s default Serverless templates (e.g., vLLM, TGI, ComfyUI, Wan, ACE). A PyWorker is a lightweight Python HTTP proxy that runs alongside your model server and:

- Exposes one or more HTTP routes (e.g., `/v1/completions`, `/generate/sync`)
- Optionally validates/transforms request payloads
- Computes per-request **workload** for autoscaling
- Forwards requests to the local model server
- Optionally supports FIFO queueing when the backend cannot process concurrent requests
- Detects readiness/failure from model logs and runs a benchmark to estimate throughput

> Important: The **core PyWorker framework** (Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig) is provided by the **`vastai`** Python package (https://github.com/vast-ai/vast-cli). This repo focuses on *worker implementations and examples*, not the framework internals.

## Repository Purpose

Use this repository as:

- A reference for how Vast templates wire up `worker.py`
- A starting point for implementing your own custom Serverless PyWorker
- A collection of working examples for common model backends

If you are looking for the framework code itself, refer to the Vast.ai SDK.

## Project Structure

Typical layout:

- `workers/`
  - Example worker implementations (each worker is usually a self-contained folder)
  - Each example typically includes:
    - `worker.py` (the entrypoint used by Serverless)
    - Optional sample workflows / payloads (for ComfyUI-based workers)
    - Optional local test harness scripts

## How Serverless launches worker.py

On each worker instance, the template’s startup script typically:

1. Clones your repository from `PYWORKER_REPO`
2. Installs dependencies from `requirements.txt`
3. Starts the **model server** (vLLM, TGI, ComfyUI, etc.)
4. Runs:
   ```bash
   python worker.py
   ```

Your `worker.py` builds a `WorkerConfig`, constructs a `Worker`, and starts the PyWorker HTTP server.

## worker.py

A PyWorker is usually a single `worker.py` that uses SDK configuration objects:

```python
from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=18000,
    model_log_file="/var/log/model/server.log",
    handlers=[
        HandlerConfig(
            route="/v1/completions",
            allow_parallel_requests=True,
            max_queue_time=60.0,
            workload_calculator=lambda payload: float(payload.get("max_tokens", 0)),
            benchmark_config=BenchmarkConfig(
                generator=lambda: {"prompt": "hello", "max_tokens": 128},
                runs=8,
                concurrency=10,
            ),
        )
    ],
    log_action_config=LogActionConfig(
        on_load=["Application startup complete."],
        on_error=["Traceback (most recent call last):", "RuntimeError:"],
        on_info=['"message":"Download'],
    ),
)

Worker(worker_config).run()
```

## Included Examples

This repository contains example PyWorkers corresponding to common Vast templates, including:

- **vLLM**: OpenAI-compatible completions/chat endpoints with parallel request support
- **TGI (Text Generation Inference)**: OpenAI-compatible endpoints and log-based readiness
- **ComfyUI (Image / JSON workflows)**: `/generate/sync` for ComfyUI workflow execution
- **ComfyUI Wan 2.2 (T2V)**: ComfyUI workflow execution producing video outputs
- **ComfyUI ACE Step (Text-to-Music)**: ComfyUI workflow execution producing audio outputs

Exact worker paths and naming may vary by template; use the `workers/` directory as the source of truth.

## Getting Started (Local)

1. Install Python dependencies for the examples you plan to run:
   ```bash
   pip install -r requirements.txt
   ```

2. Start your model server locally (vLLM, TGI, ComfyUI, etc.) and ensure:
   - You know the model server URL/port
   - You have a log file path you can tail for readiness/error detection

3. Run the worker:
   ```bash
   python worker.py
   ```
   or, if running an example from a subfolder:
   ```bash
   python workers/<example>/worker.py
   ```

> Note: Many examples assume they are running inside Vast templates (ports, log paths, model locations). You may need to adjust `model_server_port` and `model_log_file` for local usage.

## Deploying on Vast Serverless

To use a custom PyWorker with Serverless:

1. Create a public Git repository containing:
   - `worker.py`
   - `requirements.txt`

2. In your Serverless template / endpoint configuration, set:
   - `PYWORKER_REPO` to your Git repository URL
   - (Optional) `PYWORKER_REF` to a git ref (branch, tag, or commit)

3. The template startup script will clone/install and run your `worker.py`.

## Guidance for Custom Workers

When implementing your own worker:

- Define one `HandlerConfig` per route you want to expose.
- Choose a workload function that correlates with compute cost:
  - LLMs: prompt tokens + max output tokens (or `max_tokens` as a simpler proxy)
  - Non-LLMs: constant cost per request (e.g., `100.0`) is often sufficient
- Set `allow_parallel_requests=False` for backends that cannot handle concurrency (e.g., many ComfyUI deployments).
- Configure exactly **one** `BenchmarkConfig` across all handlers to enable capacity estimation.
- Use `LogActionConfig` to reliably detect “model loaded” and “fatal error” log lines.

## Community & Support

- Vast.ai Discord: https://discord.gg/Pa9M29FFye
- Vast.ai Subreddit: https://reddit.com/r/vastai/