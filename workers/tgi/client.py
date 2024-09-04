import sys
import json
from urllib.parse import urljoin

import requests


def call_generate(endpoint_group_name: str, api_key: str, server_url: str) -> None:
    WORKER_ENDPOINT = "/generate"
    COST = 100
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    response = requests.post(
        urljoin(server_url, "/route/"),
        json=route_payload,
        timeout=4,
    )
    message = response.json()
    url = message["url"]
    auth_data = dict(
        signature=message["signature"],
        cost=message["cost"],
        endpoint=message["endpoint"],
        reqnum=message["reqnum"],
        url=message["url"],
    )
    payload = dict(inputs="tell me about cats", parameters=dict(max_new_tokens=500))
    req_data = dict(payload=payload, auth_data=auth_data)
    url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {url}")
    response = requests.post(
        url,
        json=req_data,
    )
    res = response.json()
    print(res)


def call_generate_stream(endpoint_group_name: str, api_key: str, server_url: str):
    WORKER_ENDPOINT = "/generate_stream"
    COST = 100
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    response = requests.post(
        urljoin(server_url, "/route/"),
        json=route_payload,
        timeout=4,
    )
    message = response.json()
    url = message["url"]
    print(f"url: {url}")
    auth_data = dict(
        signature=message["signature"],
        cost=message["cost"],
        endpoint=message["endpoint"],
        reqnum=message["reqnum"],
        url=message["url"],
    )
    payload = dict(inputs="tell me about dogs", parameters=dict(max_new_tokens=500))
    req_data = dict(payload=payload, auth_data=auth_data)
    url = urljoin(url, WORKER_ENDPOINT)
    response = requests.post(url, json=req_data, stream=True)
    for line in response.iter_lines():
        payload = line.decode().lstrip("data:").rstrip()
        if payload:
            data = json.loads(payload)
            print(data["token"]["text"], end="")
            sys.stdout.flush()
    print()


if __name__ == "__main__":
    from lib.test_utils import test_args

    args = test_args.parse_args()
    call_generate(
        api_key=args.api_key,
        endpoint_group_name=args.endpoint_group_name,
        server_url=args.server_url,
    )
    call_generate_stream(
        api_key=args.api_key,
        endpoint_group_name=args.endpoint_group_name,
        server_url=args.server_url,
    )
