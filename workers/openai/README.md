# OpenAI Compatible PyWorker

This is the base PyWorker for OpenAI compatible inference servers.  See the [Serverless documentation](https://docs.vast.ai/serverless) for guides and how-to's.

All worker logic lives in `core.py`. The per-engine backends `vllm`, `sglang` and `llama` are thin adapters over this core, differing only in their baked default log grammar (every value is env-overridable by the image). `BACKEND=openai` is a backwards-compatible **alias for `vllm`** — `openai/worker.py` runs the vLLM worker directly, so there is one definition of the vLLM defaults and no second copy to drift; templates declaring `openai` must run a pyworker new enough to contain this split. The demo test client in `client.py` is shared — run it as `python -m workers.openai.client` regardless of which engine backend you deployed.

## Instance Setup

1. Pick a template

This worker is compatible with any backend API that properly implements the `/v1/completions` and `/v1/chat/completions` endpoints.  We currently have three templates you can choose from but you can also create your own without having to modify the PyWorker.

- [vLLM](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=vLLM%20(Serverless)) (recommended)
- [Ollama](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=Ollama%20%2B%20Qwen3%3A32b%20(Serverless))


All of these templates can be configured via the template interface.  You may want to change the model or startup arguments, depending on the template you selected.

2. Follow the [getting started guide](https://docs.vast.ai/documentation/serverless/quickstart) for help with configuring your serverless setup.  For testing, we recommend that you use the default options presented by the web interface.

## Client Setup (Demo)

1. Clone the PyWorker repository to your local machine and install the necessary requirements for running the test client.

```bash
git clone https://github.com/vast-ai/pyworker
cd pyworker
pip install uv
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Using the Test Client

Several examples have been provided in the client to help you get started with your own implementation.

First, set your API key as an environment variable:

```bash
export VAST_API_KEY=<your_api_key>
```

The `--model` and `--endpoint` flags are optional. If not provided, they default to `Qwen/Qwen3-8B` and `my-vllm-endpoint` respectively.

### Chat Completion (streaming)

Call to `/v1/chat/completions` with streaming response

```bash
python -m workers.openai.client --chat-stream --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Interactive Chat (streaming)

Interactive session with calls to `/v1/chat/completions`.

Type `clear` to clear the chat history or `quit` to exit.

```bash
python -m workers.openai.client --interactive --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Chat Completion (json)

Call to `/v1/chat/completions` with json response

```bash
python -m workers.openai.client --chat --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Tool Use (json)

Call to `/v1/chat/completions` with tool and json response.

This test defines a simple tool which will list the contents of the local pyworker directory.  The output is then analysed by the model.

```bash
python -m workers.openai.client --tools --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Completions

Call to `/v1/completions` with json response

```bash
python -m workers.openai.client --completion --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

