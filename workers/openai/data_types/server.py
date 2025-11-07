import os, json, random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from lib.data_types import EndpointHandler, ApiPayload, JsonDataException
from typing import Union, Type, Dict, Any, Optional
from aiohttp import web, ClientResponse
import nltk
import logging

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()
log = logging.getLogger(__name__)

"""
Generic dataclass accepts any dictionary in input.
"""


@dataclass
class GenericData(ApiPayload, ABC):
    input: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenericData":
        return cls(input=data["input"])

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "GenericData":
        errors = {}

        # Validate required parameters
        required_params = ["input"]
        for param in required_params:
            if param not in json_msg:
                errors[param] = "missing parameter"

        if errors:
            raise JsonDataException(errors)

        try:
            # Create clean data dict and delegate to from_dict
            clean_data = {"input": json_msg["input"]}

            return cls.from_dict(clean_data)

        except (json.JSONDecodeError, JsonDataException) as e:
            errors["parameters"] = str(e)
            raise JsonDataException(errors)

    @classmethod
    @abstractmethod
    def for_test(cls) -> "GenericData":
        pass

    def generate_payload_json(self) -> Dict[str, Any]:
        return self.input

    def count_workload(self) -> int:
        return self.input.get("max_tokens", 0)


@dataclass
class GenericHandler(EndpointHandler[GenericData], ABC):

    @property
    @abstractmethod
    def endpoint(self) -> str:
        pass

    @property
    def healthcheck_endpoint(self) -> Optional[str]:
        return os.environ.get("MODEL_HEALTH_ENDPOINT")

    @classmethod
    def payload_cls(cls) -> Type[GenericData]:
        return GenericData

    @abstractmethod
    def make_benchmark_payload(self) -> GenericData:
        pass

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        match model_response.status:
            case 200:
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
                        status=200,
                        content_type=model_response.content_type,
                    )
            case code:
                log.debug("SENDING RESPONSE: ERROR: unknown code")
                return web.Response(status=code)


@dataclass
class CompletionsData(GenericData):
    @classmethod
    def for_test(cls) -> "CompletionsData":
        system_prompt = """You are a helpful AI assistant. You have access to the following knowledge base:
    
        Zebras (US: /ˈziːbrəz/, UK: /ˈzɛbrəz, ˈziː-/)[2] (subgenus Hippotigris) are African equines 
        with distinctive black-and-white striped coats. There are three living species: Grévy's zebra 
        (Equus grevyi), the plains zebra (E. quagga), and the mountain zebra (E. zebra). Zebras share the 
        genus Equus with horses and asses, the three groups being the only living members of the family 
        Equidae. Zebra stripes come in different patterns, unique to each individual. Zebras inhabit eastern 
        and southern Africa and can be found in a variety of habitats such as savannahs, grasslands, 
        woodlands, shrublands, and mountainous areas.
        
        Please answer the following question based on the above context."""
        unique_question = " ".join(random.choices(WORD_LIST, k=int(100)))
        model = os.environ.get("MODEL_NAME")
        if not model:
            raise ValueError("MODEL_NAME environment variable not set")

        test_input = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{unique_question}",
            "temperature": 0.7,
            "max_tokens": 500,
        }
        return cls(input=test_input)


@dataclass
class CompletionsHandler(GenericHandler):
    @property
    def endpoint(self) -> str:
        return "/v1/completions"

    @classmethod
    def payload_cls(cls) -> Type[CompletionsData]:
        return CompletionsData

    def make_benchmark_payload(self) -> CompletionsData:
        return CompletionsData.for_test()


@dataclass
class ChatCompletionsData(GenericData):
    """Chat completions-specific data implementation"""

    @classmethod
    def for_test(cls) -> "ChatCompletionsData":
        system_prompt = """You are a helpful AI assistant. You have access to the following knowledge base:
    
        Zebras (US: /ˈziːbrəz/, UK: /ˈzɛbrəz, ˈziː-/)[2] (subgenus Hippotigris) are African equines 
        with distinctive black-and-white striped coats. There are three living species: Grévy's zebra 
        (Equus grevyi), the plains zebra (E. quagga), and the mountain zebra (E. zebra). Zebras share the 
        genus Equus with horses and asses, the three groups being the only living members of the family 
        Equidae. Zebra stripes come in different patterns, unique to each individual. Zebras inhabit eastern 
        and southern Africa and can be found in a variety of habitats such as savannahs, grasslands, 
        woodlands, shrublands, and mountainous areas.
        
        Please answer the following question based on the above context."""
        unique_question = " ".join(random.choices(WORD_LIST, k=int(100)))
        model = os.environ.get("MODEL_NAME")
        if not model:
            raise ValueError("MODEL_NAME environment variable not set")

        # Chat completions use messages format instead of prompt
        test_input = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},  # Shared prefix
                {"role": "user", "content": unique_question}   # Unique per request
            ],
            "temperature": 0.7,
            "max_tokens": 500,
        }
        return cls(input=test_input)


@dataclass
class ChatCompletionsHandler(GenericHandler):
    @property
    def endpoint(self) -> str:
        return "/v1/chat/completions"

    @classmethod
    def payload_cls(cls) -> Type[ChatCompletionsData]:
        return ChatCompletionsData

    def make_benchmark_payload(self) -> ChatCompletionsData:
        return ChatCompletionsData.for_test()
