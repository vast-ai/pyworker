# vLLM PyWorker

Serverless worker for [vLLM](https://github.com/vllm-project/vllm) (`BACKEND=vllm`).

vLLM serves the OpenAI-compatible API, so all worker logic lives in
[`workers/openai/core.py`](../openai/core.py). `worker.py` here is a thin adapter that
supplies vLLM's default log grammar; the image overrides every value via env
(`MODEL_LOG`, `MODEL_LOAD_LOG_MSG`, `MODEL_HEALTH_ENDPOINT`, …).

The test client is shared — see [`workers/openai/README.md`](../openai/README.md) and run
`python -m workers.openai.client`.
