# OpenAI Compatible PyWorker

This is the base PyWorker for OpenAI compatible inference servers.  See the [Serverless documentation](https://docs.vast.ai/serverless) for guides and how-to's.

All worker logic lives in `core.py`. The per-engine backends `vllm`, `sglang` and `llama` are thin adapters over this core, differing only in their baked default log grammar (every value is env-overridable by the image). `BACKEND=openai` is a backwards-compatible **alias for `vllm`** — `openai/worker.py` runs the vLLM worker directly, so there is one definition of the vLLM defaults and no second copy to drift; templates declaring `openai` must run a pyworker new enough to contain this split. The demo test client in `client.py` is shared — run it as `python -m workers.openai.client` regardless of which engine backend you deployed.

## Routes

| Route | Request | Response |
|---|---|---|
| `/v1/completions`, `/v1/chat/completions` | JSON | JSON or SSE |
| `/v1/chat/completions/batch` | JSON (`messages`: a list of conversations) | JSON |
| `/v1/embeddings` | JSON | JSON |
| `/v1/rerank` | JSON (`query`, `documents`) | JSON |
| `/v1/score` | JSON (`text_1`/`text_2`, `queries`/`items` or `data_1`/`data_2`) | JSON |
| `/v1/audio/speech` | JSON (voice-clone `ref_audio`: http(s) URL or `data:` URI) | audio bytes |
| `/v1/audio/speech/batch` | JSON (`items`, each a speech request; `ref_audio` on the batch or an item) | JSON, audio base64'd per item |
| `/v1/audio/generate` | JSON (`input` is the prompt; `audio_length` in seconds) | audio bytes |
| `/v1/audio/transcriptions`, `/v1/audio/translations` | JSON with the file base64'd in `file` | JSON or text |
| `/v1/images/generations` | JSON | JSON |
| `/v1/images/edits` | JSON with `image` (base64, or a list), or `url` (http(s) or `data:`) | JSON |
| `/v1/videos/sync` | JSON: form fields, plus uploads base64'd (`input_reference`, `input_references`, `source_video`, ...) | `video/mp4` bytes |

The worker envelope is JSON, so uploads arrive base64-encoded (`file`, `image`, `mask`, and
the video file fields) and are sent to the engine as multipart form data. `filename` /
`mask_filename` / `<field>_filename` set the file type. Requires a `vastai` SDK with
multipart support; on an older SDK the upload routes are not served.

`/v1/videos/sync` holds the request open until the video is done (the engine gives up after
`VLLM_OMNI_VIDEO_SYNC_TIMEOUT`, 600 s by default). The noise masks are JSON files. The
asynchronous `/v1/videos` job API is not served: it is polled with GETs, and the worker
takes only POSTs.

References (`url` on edits, `ref_audio` on speech and speech batches, and the URL in each
video reference) are passed to the engine, not uploaded,
so they must be an http(s) URL or a `data:` URI; anything else, a file path included, is
refused. http(s) URLs are fetched by the engine from inside the instance and are not
filtered here, as with `image_url` on chat.

`lora` and `frame_interpolation_model_path` name files already on the instance and are
passed through as given: bake a LoRA into a custom image (recommended), or fetch it with
the provisioning script.

vLLM-Omni reads an edit mask from `mask_image`, not the `mask` the OpenAI API names, so
a `mask` is not applied there.

**Served by default: completions and chat, as before, plus the `BENCHMARK_ROUTE`.** Other
routes are opt-in with `OPENAI_ROUTES`, since the worker cannot know what the loaded model
supports. A route the model does not serve returns the engine's 404, and is still counted
as work against the queue estimate, so list only the routes the model serves.

What each engine actually serves depends on the engine and the loaded model:

| Engine | Typically serves |
|---|---|
| vLLM, SGLang, llama.cpp | completions, chat; embeddings with an embedding model; rerank with a reranker |
| vLLM with Whisper or Voxtral | transcriptions, translations (and no completions at all) |
| vLLM-Omni (`--omni`) | image generations and edits, speech and speech batches, audio generation (Stable Audio), video (`/v1/videos/sync`), chat batches, alongside text |

SGLang's `/v1/score` is a different API (label-token probabilities, `query`/`items`/
`label_token_ids`); the score benchmark sends vLLM's shape, so benchmark an SGLang
reranker on `/v1/rerank`.

## Benchmarking

`BENCHMARK_ROUTE` names the route to benchmark, and defaults to `/v1/completions`, so an
LLM deployment benchmarks exactly what it did before. Any route above can be benchmarked,
so a deployment serving only one (an edit-only image model, say) can still become ready.
Every benchmark request weighs one reference request, so the score means the same thing
whichever route is benchmarked.

If the model does not serve the benchmarked route, the benchmark fails and the worker does
not become ready. The startup log names the route and the variable to change:

```
benchmarking: /v1/completions. If this model does not serve it, set BENCHMARK_ROUTE to
the route it does.
```

The transcription and translation benchmarks send real speech -- a bundled 10.7 s sample,
tiled to 30 s -- because noise leaves the decoder idle and overstates throughput.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `BENCHMARK_ROUTE` | `/v1/completions` | The route to benchmark. Served by default; the worker refuses to start if `OPENAI_ROUTES` leaves it out. |
| `OPENAI_ROUTES` | completions, chat | Comma-separated routes to serve. Must include `BENCHMARK_ROUTE`. |
| `BENCHMARK_SPEECH_VOICE` | none | `voice` to send when benchmarking speech. |
| `BENCHMARK_EMBED_CHARS` | 600 | Characters the embeddings benchmark sends. Sized for a 256-token encoder; raise it for a long-context model. |

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


### The other routes

One call to the route, printing the result or saving returned media to `--out` (default:
the current directory). `--model` is optional here: the worker fills in its own.

```bash
python -m workers.openai.client --embeddings --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --rerank --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --score --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --speech --voice <VOICE> --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --speech-batch --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --transcribe clip.wav --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --translate clip.wav --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --image --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --edit photo.png --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --audio-generate --seconds 5 --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --video [--image-file first-frame.png] --endpoint <ENDPOINT_NAME>
python -m workers.openai.client --chat-batch --endpoint <ENDPOINT_NAME>
```
