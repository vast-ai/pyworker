# SGLang PyWorker

Serverless worker for [SGLang](https://github.com/sgl-project/sglang) (`BACKEND=sglang`).

SGLang serves the OpenAI-compatible API, so all worker logic lives in
[`workers/openai/core.py`](../openai/core.py). `worker.py` here is a thin adapter that
supplies SGLang's default log grammar; the image overrides every value via env
(`MODEL_LOG`, `MODEL_LOAD_LOG_MSG`, `MODEL_HEALTH_ENDPOINT`, …).

The test client is shared — see [`workers/openai/README.md`](../openai/README.md) and run
`python -m workers.openai.client`.
