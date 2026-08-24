# llama.cpp PyWorker

Serverless worker for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`
(`BACKEND=llama`).

llama-server serves the OpenAI-compatible API, so all worker logic lives in
[`workers/openai/core.py`](../openai/core.py). `worker.py` here is a thin adapter that
supplies llama.cpp's default log grammar; the image overrides every value via env
(`MODEL_LOG`, `MODEL_LOAD_LOG_MSG`, `MODEL_HEALTH_ENDPOINT`, …).

The test client is shared — see [`workers/openai/README.md`](../openai/README.md) and run
`python -m workers.openai.client`.
