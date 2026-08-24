"""vLLM adapter — logic is in workers/openai/core.py; this just supplies vLLM's defaults."""

from workers.openai.core import EngineDefaults, run

run(EngineDefaults(
    name="vllm",
    model_log_file="/var/log/portal/vllm.log",
    load_log_msgs=["Application startup complete."],
    error_log_msgs=["INFO exited: vllm", "RuntimeError: Engine", "Traceback (most recent call last):"],
))
