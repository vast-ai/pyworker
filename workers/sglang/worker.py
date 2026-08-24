"""SGLang adapter — logic is in workers/openai/core.py; this just supplies SGLang's defaults."""

from workers.openai.core import EngineDefaults, run

run(EngineDefaults(
    name="sglang",
    model_log_file="/var/log/portal/sglang.log",
    load_log_msgs=["The server is fired up and ready to roll!"],
    error_log_msgs=["INFO exited: sglang", "Traceback (most recent call last):"],
))
