import os
import sys
import random
import dataclasses
from typing import Dict, Any
from functools import cache
from math import ceil
from pathlib import Path
import json
import logging

from lib.data_types import ApiPayload, JsonDataException

log = logging.getLogger(__file__)

def count_workload() -> float:
    # Always 100.0 where there is a single instance of ComfyUI handling requests
    # Results will indicate % or a job completed per second.  Avoids sub 0.1 sec performance indication
    return 100.0

@dataclasses.dataclass
class ComfyWorkflowData(ApiPayload):
    input: dict

    @classmethod
    def for_test(cls):
        """
        If the user has provided a benchmark workflow we can use it here to properly gauge performance.
        Otherwise, use the variables available to simulate workflows of the required running time
        Example: SD1.5, simple image gen 10000 steps, 512px x 512px will run for approximately 9 minutes @ ~18 it/s (RTX 4090)
        """
        # Try to load benchmark.json
        benchmark_file = Path("workers/comfyui-json/misc/benchmark.json")
        
        if benchmark_file.exists():
            try:
                with open(benchmark_file, "r") as f:
                    benchmark_workflow = json.load(f)
                return cls(
                    input={
                        "request_id": f"test-{random.randint(1000, 99999)}",
                        "workflow_json": benchmark_workflow
                    }
                )
            except (json.JSONDecodeError, IOError):
                # JSON is malformed or file can't be read, fall through to default
                log.error(f"Failed to benchmark using {benchmark_file}")
        
        # Fallback: read prompts and construct payload
        log.info("Using fallback method for benchmarking")
        with open("workers/comfyui-json/misc/test_prompts.txt", "r") as f:
            test_prompts = f.readlines()
        
        test_prompt = random.choice(test_prompts).rstrip()
        return cls(
            input={
                "request_id": f"test-{random.randint(1000, 99999)}",
                "modifier": "Text2Image",
                "modifications": {
                    "prompt": test_prompt,
                    "width": os.getenv('BENCHMARK_TEST_WIDTH', 512),
                    "height": os.getenv('BENCHMARK_TEST_HEIGHT', 512),
                    "steps": os.getenv('BENCHMARK_TEST_STEPS', 20),
                    "seed": random.randint(0, sys.maxsize),
                }
            }
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        # input is already a dict, just return it wrapped in the expected structure
        return {"input": self.input}

    def count_workload(self) -> float:
        return count_workload()

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "ComfyWorkflowData":
        # Extract required fields
        if "input" not in json_msg:
            raise JsonDataException({"input": "missing parameter"})
        
        return cls(
            input=json_msg["input"]
        )