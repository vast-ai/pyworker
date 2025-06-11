import dataclasses
import random
import inspect
from typing import Dict, Any

from transformers import OpenAIGPTTokenizer
import nltk

from lib.data_types import ApiPayload, JsonDataException

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

# used to count to count tokens and workload for LLM
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")


@dataclasses.dataclass
class InputData(ApiPayload):
    prompt: str
    max_response_tokens: int

    @classmethod
    def for_test(cls) -> "InputData":
        prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
        return cls(prompt=prompt, max_response_tokens=300)

    def generate_payload_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def count_workload(self) -> int:
        return len(tokenizer.tokenize(self.prompt))

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputData":
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
