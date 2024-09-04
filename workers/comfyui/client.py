from urllib.parse import urljoin

import requests

from lib.test_utils import print_truncate_res

"""
NOTE: this client example uses a custom comfy workflow compatible with SD3 only
"""


def call_default_workflow(
    endpoint_group_name: str, api_key: str, server_url: str
) -> None:
    WORKER_ENDPOINT = "/prompt"
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
    payload = dict(
        prompt="a fat fluffy cat", width=1024, height=1024, steps=20, seed=123456789
    )
    req_data = dict(payload=payload, auth_data=auth_data)
    url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {url}")
    response = requests.post(
        url,
        json=req_data,
    )
    print_truncate_res(str(response.json()))


def call_custom_workflow_for_sd3(
    endpoint_group_name: str, api_key: str, server_url: str
) -> None:
    WORKER_ENDPOINT = "/custom-workflow"
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
    workflow = {
        "3": {
            "inputs": {
                "seed": 156680208700286,
                "steps": 20,
                "cfg": 8,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
            "class_type": "KSampler",
        },
        "4": {
            "inputs": {"ckpt_name": "sd3_medium_incl_clips_t5xxlfp16.safetensors"},
            "class_type": "CheckpointLoaderSimple",
        },
        "5": {
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        },
        "6": {
            "inputs": {
                "text": "beautiful scenery nature glass bottle landscape, purple galaxy bottle",
                "clip": ["4", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "7": {
            "inputs": {"text": "text, watermark", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
        },
        "8": {
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
        },
        "9": {
            "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
            "class_type": "SaveImage",
        },
    }
    # these values should match the values in the custom workflow above,
    # they are used to calculate workload
    custom_fields = dict(
        steps=20,
        width=512,
        height=512,
    )
    req_data = dict(
        payload=dict(custom_fields=custom_fields, workflow=workflow),
        auth_data=auth_data,
    )
    url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {url}")
    response = requests.post(
        url,
        json=req_data,
    )
    print_truncate_res(str(response.json()))


if __name__ == "__main__":
    from lib.test_utils import test_args

    args = test_args.parse_args()
    call_default_workflow(
        api_key=args.api_key,
        endpoint_group_name=args.endpoint_group_name,
        server_url=args.server_url,
    )
    call_custom_workflow_for_sd3(
        api_key=args.api_key,
        endpoint_group_name=args.endpoint_group_name,
        server_url=args.server_url,
    )
