# OpenAI Compatible PyWorker

This is the base PyWorker for OpenAI compatible inference servers.  See the [Serverless documentation](https://docs.vast.ai/serverless) for guides and how-to's.

## Instance Setup

1. Pick a template

This worker is compatible with any backend API that properly implements the `/v1/completions` and `/v1/chat/completions` endpoints.  We currently have four templates you can choose from but you can also create your own without having to modify the PyWorker.

- [vLLM](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=vLLM%20%2B%20Qwen%2FQwen3-8B%20(Serverless)) (recommended)
- [Ollama](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=Ollama%20%2B%20Qwen3%3A32b%20(Serverless))
- [HuggingFace TGI](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=TGI%20%2B%20Qwen3-8B%20(Serverless))
- llama.cpp


All of these templates can be configured via the template interface.  You may want to change the model or startup arguments, depending on the template you selected.

2. Follow the [getting started guide](https://docs.vast.ai/serverless/getting-started) for help with configuring your serverless setup.  For testing, we recommend that you use the default options presented by the web interface.

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

### Completions

Call to `/v1/completions` with json response

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --completion --model <MODEL_NAME>
```

### Chat Completion (json)

Call to `/v1/chat/completions` with json response

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --chat --model <MODEL_NAME>
```

### Chat Completion (streaming)

Call to `/v1/chat/completions` with streaming response

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --chat-stream --model <MODEL_NAME>
```

### Tool Use (json)

Call to `/v1/chat/completions` with tool and json response.

This test defines a simple tool which will list the contents of the local pyworker directory.  The output is then analysed by the model.

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --tools --model <MODEL_NAME>
```

### Interactive Chat (streaming)

Interactive session with calls to `/v1/chat/completions`.

Type `clear` to clear the chat history or `quit` to exit.

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --interactive --model <MODEL_NAME>
```

