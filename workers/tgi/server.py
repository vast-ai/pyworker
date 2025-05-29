import os
import logging
from typing import Union, Type, Dict
import dataclasses

from aiohttp import web, ClientResponse, ClientSession # type: ignore

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler, MODELLOADEDSTATUS
from lib.server import start_server
from .data_types import InputData


MODEL_SERVER_URL = "http://0.0.0.0:5001"

# This is the last log line that gets emitted once comfyui+extensions have been fully loaded
MODEL_SERVER_START_LOG_MSG = ['"message":"Connected","target":"text_generation_router"', '"message":"Connected","target":"text_generation_router::server"']
MODEL_SERVER_ERROR_LOG_MSGS = ["Error: WebserverFailed", "Error: DownloadError", "Error: ShardCannotStart"]


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


@dataclasses.dataclass
class GenerateHandler(EndpointHandler[InputData]):

    @property
    def endpoint(self) -> str:
        return "/generate"

    @classmethod
    def payload_cls(cls) -> Type[InputData]:
        return InputData

    def make_benchmark_payload(self) -> InputData:
        return InputData.for_test()

    def health_check(self) -> MODELLOADEDSTATUS:
        """Simple health check that returns the status enum directly"""
        # This is a synchronous version that can be used for basic health checks
        # For now, we'll return READY - in the future this could be enhanced
        # to do a simple check without requiring async/await
        return MODELLOADEDSTATUS.READY

    async def model_health_check(self, model_server_url: str) -> Dict[str, str]:
        """TGI-specific health check implementation"""
        url = f'{model_server_url}/health'                
        try:
            async with ClientSession() as session:
                async with session.get(url) as health_response:
                    status_code = health_response.status
                    if status_code == 200:
                        message = await health_response.text()
                        return {'status': MODELLOADEDSTATUS.READY.value, 'reason': message}
                    elif status_code == 503:
                        try:
                            error_response = await health_response.json()
                            error = error_response.get("error", "")
                            error_type = error_response.get("error_type", "")
                            reason = f'{error} {error_type}'.strip()
                        except Exception:
                            reason = "Unhealthy (invalid JSON error response)"
                        return {'status': MODELLOADEDSTATUS.FAILED.value, 'reason': reason}
                    else:
                        return {
                            'status': MODELLOADEDSTATUS.DEFERRED_TO_LOG_FILE.value,
                            'reason': f'Model health endpoint not ready (status: {status_code})'
                        }
        except Exception as e:
            log.debug(f"Health check exception: {str(e)}")
            return {
                'status': MODELLOADEDSTATUS.FAILED.value,
                'reason': f'Exception during health check: {str(e)}'
            }

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        _ = client_request
        match model_response.status:
            case 200:
                log.debug("SUCCESS")
                data = await model_response.json()
                return web.json_response(data=data)
            case code:
                log.debug("SENDING RESPONSE: ERROR: unknown code")
                return web.Response(status=code)


class GenerateStreamHandler(EndpointHandler[InputData]):
    @property
    def endpoint(self) -> str:
        return "/generate_stream"

    @classmethod
    def payload_cls(cls) -> Type[InputData]:
        return InputData

    def make_benchmark_payload(self) -> InputData:
        return InputData.for_test()

    def health_check(self) -> MODELLOADEDSTATUS:
        """Simple health check that returns the status enum directly"""
        # This is a synchronous version that can be used for basic health checks
        # For now, we'll return READY - in the future this could be enhanced
        # to do a simple check without requiring async/await
        return MODELLOADEDSTATUS.READY

    async def model_health_check(self, model_server_url: str) -> Dict[str, str]:
        """TGI-specific health check implementation (same as GenerateHandler)"""
        url = f'{model_server_url}/health'                
        try:
            async with ClientSession() as session:
                async with session.get(url) as health_response:
                    status_code = health_response.status
                    if status_code == 200:
                        message = await health_response.text()
                        return {'status': MODELLOADEDSTATUS.READY.value, 'reason': message}
                    elif status_code == 503:
                        try:
                            error_response = await health_response.json()
                            error = error_response.get("error", "")
                            error_type = error_response.get("error_type", "")
                            reason = f'{error} {error_type}'.strip()
                        except Exception:
                            reason = "Unhealthy (invalid JSON error response)"
                        return {'status': MODELLOADEDSTATUS.FAILED.value, 'reason': reason}
                    else:
                        return {
                            'status': MODELLOADEDSTATUS.DEFERRED_TO_LOG_FILE.value,
                            'reason': f'Model health endpoint not ready (status: {status_code})'
                        }
        except Exception as e:
            log.debug(f"Health check exception: {str(e)}")
            return {
                'status': MODELLOADEDSTATUS.FAILED.value,
                'reason': f'Exception during health check: {str(e)}'
            }

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        match model_response.status:
            case 200:
                log.debug("Streaming response...")
                res = web.StreamResponse()
                res.content_type = "text/event-stream"
                await res.prepare(client_request)
                async for chunk in model_response.content:
                    await res.write(chunk)
                await res.write_eof()
                log.debug("Done streaming response")
                return res
            case code:
                log.debug("SENDING RESPONSE: ERROR: unknown code")
                return web.Response(status=code)


backend = Backend(
    model_server_url=MODEL_SERVER_URL,
    model_log_file=os.environ["MODEL_LOG"],
    allow_parallel_requests=True,
    model_type= 'tgi',
    benchmark_handler=GenerateHandler(benchmark_runs=3, benchmark_words=256),
    log_actions=[
         *[
            (LogAction.ModelLoaded, info_msg)
            for info_msg in MODEL_SERVER_START_LOG_MSG
        ],
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
    web.post("/generate", backend.create_handler(GenerateHandler())),
    web.post("/generate_stream", backend.create_handler(GenerateStreamHandler())),
    web.get("/ping", handle_ping),
]

if __name__ == "__main__":
    start_server(backend, routes)
