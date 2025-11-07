import logging
import uuid
import random
from urllib.parse import urljoin
import json

import requests

from lib.test_utils import print_truncate_res
from utils.endpoint_util import Endpoint
from utils.ssl import get_cert_file_path
from .data_types import count_workload

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


def call_text2image_workflow(
    endpoint_group_name: str, api_key: str, server_url: str
) -> None:
    """Simple Text2Image using the new modifier-based approach"""
    
    def make_request(url: str, payload: dict, timeout: int = None, verify=True, context: str = "request"):
        """Helper function for making requests with consistent error handling"""
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                verify=verify
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as http_err:
            log.error(f"HTTP error occurred during {context}: {http_err}")
            log.error(f"Status Code: {response.status_code}")
            log.error("Response content:", response.text)
            return None
        except requests.exceptions.Timeout:
            log.error(f"Timeout occurred during {context}: {url}")
            return None
        except requests.exceptions.ConnectionError:
            log.error(f"Connection error occurred during {context}: {url}")
            return None
        except json.JSONDecodeError as json_err:
            log.error(f"Failed to decode JSON response during {context}: {json_err}")
            if 'response' in locals():
                print("Response content:", response.text)
            return None
        except Exception as err:
            log.error(f"An unexpected error occurred during {context}: {err}")
            if 'response' in locals():
                log.error("Response content (if available):", response.text)
            return None
    
    WORKER_ENDPOINT = "/generate/sync"

    # This worker has concurrency = 1.  All workloads have cost value 1.0
    COST = count_workload()
    
    # Route to get worker URL
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    
    # First request - get routing information
    route_response = make_request(
        url=urljoin(server_url, "/route/"),
        payload=route_payload,
        timeout=4,
        context="route request"
    )
    
    if route_response is None:
        return None
    
    if "url" not in route_response or not route_response["url"]:
        log.error("Error: No worker in 'Ready' state. Please wait while the serverless engine removes errored workers or finishes loading new workers.")
        return None
    
    if "status" in route_response:
        print(f"Autoscaler status: {route_response['status']}")
        return None
    
    # Extract data from route response
    url = route_response["url"]
    auth_data = dict(
        signature=route_response["signature"],
        cost=route_response["cost"],
        endpoint=route_response["endpoint"],
        reqnum=route_response["reqnum"],
        url=route_response["url"],
        request_idx=route_response["request_idx"],
    )
    
    # Build the payload for the worker request
    worker_payload = {
        "input": {
            "request_id": str(uuid.uuid4()),
            "modifier": "Text2Image",
            "modifications": {
                "prompt": "a beautiful landscape with mountains and lakes",
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "seed": random.randint(0, 2**32 - 1)
            },
            "workflow_json": {}  # Empty since using modifier approach
        }
    }
    
    req_data = dict(payload=worker_payload, auth_data=auth_data)
    worker_url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {worker_url}")
    
    # Second request - call the worker endpoint
    worker_response = make_request(
        url=worker_url,
        payload=req_data,
        verify=get_cert_file_path(),
        context="worker request"
    )
    
    return worker_response


if __name__ == "__main__":
    from lib.test_utils import test_args

    args = test_args.parse_args()
    endpoint_api_key = Endpoint.get_endpoint_api_key(
        endpoint_name=args.endpoint_group_name,
        account_api_key=args.api_key,
        instance=args.instance,
    )

    if endpoint_api_key:
        result = call_text2image_workflow(
            api_key=endpoint_api_key,
            endpoint_group_name=args.endpoint_group_name,
            server_url=args.server_url,
        )
        if result is None:
            log.error("Text2Image workflow failed")
        else:   
            print(result)
    else:
        log.error(f"Failed to get API key for endpoint {args.endpoint_group_name}")
