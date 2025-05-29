import dataclasses
import random
import inspect
from typing import Dict, Any

from transformers import AutoTokenizer
import nltk
from aiohttp import ClientSession

from lib.data_types import ApiPayload, JsonDataException, MODELLOADEDSTATUS

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")


@dataclasses.dataclass
class InputParameters:
    max_new_tokens: int = 256

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputParameters":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        return cls(
            **{
                k: v
                for k, v in json_msg.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclasses.dataclass
class InputData(ApiPayload):
    inputs: str
    parameters: InputParameters

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputData":
        return cls(
            inputs=data["inputs"], parameters=InputParameters(**data["parameters"])
        )

    @classmethod
    def for_test(cls) -> "InputData":
        prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
        return cls(inputs=prompt, parameters=InputParameters())

    def generate_payload_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def count_workload(self) -> int:
        return self.parameters.max_new_tokens

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputData":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        try:
            parameters = InputParameters.from_json_msg(json_msg["parameters"])
            return cls(inputs=json_msg["inputs"], parameters=parameters)
        except JsonDataException as e:
            errors["parameters"] = e.message
            raise JsonDataException(errors)

async def tgi_health_check(model_server_url: str) -> Dict[str, str]:
    """
    Check the health status of the TGI model server.
    """
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
        return {
            'status': MODELLOADEDSTATUS.FAILED.value,
            'reason': f'Exception during health check: {str(e)}'
        }
