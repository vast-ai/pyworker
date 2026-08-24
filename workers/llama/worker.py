"""llama.cpp adapter — logic is in workers/openai/core.py; this just supplies its defaults."""

from workers.openai.core import EngineDefaults, run

run(EngineDefaults(
    name="llama",
    model_log_file="/var/log/portal/llama.log",
    # "model loaded" prints after /health flips to 200; don't use "listening on" (prints during 503).
    load_log_msgs=["model loaded"],
    error_log_msgs=["INFO exited: llama", "error loading model", "failed to load model"],
))
