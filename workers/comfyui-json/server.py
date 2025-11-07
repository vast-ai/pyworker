import os
import logging
import dataclasses
import base64
from typing import Optional, Union, Type

from aiohttp import web, ClientResponse

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler
from lib.server import start_server
from .data_types import ComfyWorkflowData


MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:18288")

# This is the last log line that gets emitted once comfyui+extensions have been fully loaded
MODEL_SERVER_START_LOG_MSG = "To see the GUI go to: "
MODEL_SERVER_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",  # This error is emitted when the downloaded model is corrupted
    "Value not in list: ",  # This error is emitted when the model file is not there at all
    "[ERROR] Provisioning Script failed", # Error inserted by provisioning script if models/nodes fail to download
]


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


async def generate_client_response(
        client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        # Check if the response is actually streaming based on response headers/content-type
        is_streaming_response = (
            model_response.content_type == "text/event-stream"
            or model_response.content_type == "application/x-ndjson"
            or model_response.headers.get("Transfer-Encoding") == "chunked"
            or "stream" in model_response.content_type.lower()
        )

        if is_streaming_response:
            log.debug("Detected streaming response...")
            res = web.StreamResponse()
            res.content_type = model_response.content_type
            await res.prepare(client_request)
            async for chunk in model_response.content:
                await res.write(chunk)
            await res.write_eof()
            log.debug("Done streaming response")
            return res
        else:
            log.debug("Detected non-streaming response...")
            content = await model_response.read()
            return web.Response(
                body=content,
                status=model_response.status,
                content_type=model_response.content_type
            )
            

@dataclasses.dataclass
class ComfyWorkflowHandler(EndpointHandler[ComfyWorkflowData]):

    @property
    def endpoint(self) -> str:
        return "/generate/sync"

    @property
    def healthcheck_endpoint(self) -> Optional[str]:
        return f"{MODEL_SERVER_URL}/health"

    @classmethod
    def payload_cls(cls) -> Type[ComfyWorkflowData]:
        return ComfyWorkflowData

    def make_benchmark_payload(self) -> ComfyWorkflowData:
        return ComfyWorkflowData.for_test()

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        return await generate_client_response(client_request, model_response)


backend = Backend(
    model_server_url=MODEL_SERVER_URL,
    model_log_file=os.environ["MODEL_LOG"],
    allow_parallel_requests=False,
    benchmark_handler=ComfyWorkflowHandler(
        benchmark_runs=3, benchmark_words=100
    ),
    log_actions=[
        (LogAction.ModelLoaded, MODEL_SERVER_START_LOG_MSG),
        (LogAction.Info, "Downloading:"),
        *[
            (LogAction.ModelError, error_msg)
            for error_msg in MODEL_SERVER_ERROR_LOG_MSGS
        ],
    ],
)


async def handle_ping(_):
    return web.Response(body="pong")


routes = [
    web.post("/generate/sync", backend.create_handler(ComfyWorkflowHandler())),
    web.get("/ping", handle_ping),
]

if __name__ == "__main__":
    start_server(backend, routes)
