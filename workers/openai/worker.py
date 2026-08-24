"""BACKEND=openai — alias for vllm. Running this module runs the vLLM worker, so openai
and vllm share one definition of the defaults (no copy to drift). Existing openai
templates keep working on a pyworker that has this split."""

import workers.vllm.worker  # noqa: F401  — imported for side effect: runs the vLLM worker
