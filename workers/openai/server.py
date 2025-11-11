import os
import logging
from .data_types.server import CompletionsHandler, ChatCompletionsHandler
from aiohttp import web
from lib.backend import Backend, LogAction
from lib.server import start_server

# This line indicates that the inference server is listening
MODEL_SERVER_START_LOG_MSG = [
    "Application startup complete.",  # vLLM
    "llama runner started",  # Ollama
    '"message":"Connected","target":"text_generation_router"',  # TGI
    '"message":"Connected","target":"text_generation_router::server"',  # TGI
    "starting the main loop",  # llama.cpp
]

MODEL_SERVER_ERROR_LOG_MSGS = [
    "INFO exited: vllm",  # vLLM
    "RuntimeError: Engine",  # vLLM
    "Error: pull model manifest:",  # Ollama
    "stalled; retrying",  # Ollama
    "Error: WebserverFailed",  # TGI
    "Error: DownloadError",  # TGI
    "Error: ShardCannotStart",  # TGI
    "main: exiting due to model loading error",  # llama.cpp
    "Aborted (core dumped)",  # llama.cpp
]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

backend = Backend(
    model_server_url=os.environ["MODEL_SERVER_URL"],
    model_log_file=os.environ["MODEL_LOG"],
    allow_parallel_requests=True,
    benchmark_handler=CompletionsHandler(benchmark_runs=3, benchmark_words=256),
    log_actions=[
        *[(LogAction.ModelLoaded, info_msg) for info_msg in MODEL_SERVER_START_LOG_MSG],
        (LogAction.Info, '"message":"Download'),
        *[
            (LogAction.ModelError, error_msg)
            for error_msg in MODEL_SERVER_ERROR_LOG_MSGS
        ],
    ],
)


async def handle_ping(_):
    return web.Response(body="pong")


routes = [
    web.post("/v1/completions", backend.create_handler(CompletionsHandler())),
    web.post("/v1/chat/completions", backend.create_handler(ChatCompletionsHandler())),
    web.get("/ping", handle_ping),
]

if __name__ == "__main__":
    start_server(backend, routes)
