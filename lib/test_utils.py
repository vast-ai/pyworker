import os
import time
import argparse
from typing import Callable, List, Dict, Tuple, Dict, Any
from time import sleep
import threading
from enum import Enum
from collections import Counter
from dataclasses import dataclass, field, asdict
from urllib.parse import urljoin

import requests

from lib.data_types import AuthData, ApiPayload


class ClientStatus(Enum):
    FetchEndpoint = 1
    Generating = 2
    Done = 3
    Error = 4


total_success = 0
last_res = []

start_time = time.time()
test_args = argparse.ArgumentParser(description="Test inference endpoint")
test_args.add_argument(
    "-k", dest="api_key", type=str, required=True, help="Your vast account API key"
)
test_args.add_argument(
    "-e",
    dest="endpoint_group_name",
    type=str,
    required=True,
    help="Endpoint group name",
)
test_args.add_argument(
    "-l",
    dest="server_url",
    action="store_const",
    const="http://localhost:8081",
    default="https://run.vast.ai",
    help="Call local autoscaler instead of prod, for dev use only",
)

GetPayloadAndWorkload = Callable[[], Tuple[Dict[str, Any], float]]


def print_truncate_res(res: str):
    if len(res) > 50:
        print(f"{res[:50]}....{res[-100:]}")
    else:
        print(res)


@dataclass
class ClientState:
    endpoint_group_name: str
    api_key: str
    server_url: str
    worker_endpoint: str
    payload: ApiPayload
    url: str = ""
    status: ClientStatus = ClientStatus.FetchEndpoint
    as_error: List[str] = field(default_factory=list)
    infer_error: List[str] = field(default_factory=list)
    conn_errors: Counter = field(default_factory=Counter)

    def make_call(self):
        self.status = ClientStatus.FetchEndpoint
        route_payload = {
            "endpoint": self.endpoint_group_name,
            "api_key": self.api_key,
            "cost": self.payload.count_workload(),
        }
        response = requests.post(
            urljoin(self.server_url, "/route/"),
            json=route_payload,
            timeout=4,
        )
        if response.status_code != 200:
            self.as_error.append(
                f"code: {response.status_code}, body: {response.text}",
            )
            self.status = ClientStatus.Error
            return
        message = response.json()
        worker_address = message["url"]
        req_data = dict(
            payload=asdict(self.payload),
            auth_data=asdict(AuthData.from_json_msg(message)),
        )
        self.url = worker_address
        url = urljoin(worker_address, self.worker_endpoint)
        self.status = ClientStatus.Generating
        response = requests.post(
            url,
            json=req_data,
        )
        if response.status_code != 200:
            self.infer_error.append(
                f"code: {response.status_code}, body: {response.text}, url: {url}",
            )
            self.status = ClientStatus.Error
            return
        res = str(response.json())
        global total_success
        global last_res
        total_success += 1
        if len(res) > 50:
            last_res.append(f"{res[:50]}....{res[-50:]}")
        else:
            last_res.append(res)
        self.status = ClientStatus.Done

    def simulate_user(self) -> None:
        try:
            self.make_call()
        except Exception as e:
            self.status = ClientStatus.Error
            _ = e
            self.conn_errors[self.url] += 1


def print_state(clients: List[ClientState], num_clients: int) -> None:
    print("starting up...")
    sleep(2)
    center_size = 14
    global start_time
    while len(clients) < num_clients or (
        any(
            map(
                lambda client: client.status
                in [ClientStatus.FetchEndpoint, ClientStatus.Generating],
                clients,
            )
        )
    ):
        sleep(0.5)
        os.system("clear")
        print(
            " | ".join(
                [member.name.center(center_size) for member in ClientStatus]
                + [
                    item.center(center_size)
                    for item in [
                        "urls",
                        "as_error",
                        "infer_error",
                        "conn_error",
                        "total_success",
                    ]
                ]
            )
        )
        unique_urls = len(set([c.url for c in clients if c.url != ""]))
        as_errors = sum(
            map(
                lambda client: len(client.as_error),
                [client for client in clients],
            )
        )
        infer_errors = sum(
            map(
                lambda client: len(client.infer_error),
                [client for client in clients],
            )
        )
        conn_errors = sum([client.conn_errors for client in clients], start=Counter())
        conn_errors_str = ",".join(map(str, conn_errors.values())) or "0"
        elapsed = time.time() - start_time
        print(
            " | ".join(
                map(
                    lambda item: str(item).center(center_size),
                    [
                        len(list(filter(lambda x: x.status == member, clients)))
                        for member in ClientStatus
                    ]
                    + [
                        unique_urls,
                        as_errors,
                        infer_errors,
                        conn_errors_str,
                        f"{total_success}({((total_success/elapsed) * 60):.2f}/minute)",
                    ],
                )
            )
        )
        if conn_errors:
            print("conn_errors:")
            for url, count in conn_errors.items():
                print(url.ljust(28), ": ", str(count))
        elapsed = time.time() - start_time
        print(f"\n elapsed: {int(elapsed // 60)}:{int(elapsed % 60)}")
        if last_res:
            for i, res in enumerate(last_res[-10:]):
                print_truncate_res(f"res #{1+i+max(len(last_res )-10,0)}: {res}")


def run_test(
    num_requests: int,
    requests_per_second: int,
    endpoint_group_name: str,
    api_key: str,
    server_url: str,
    worker_endpoint: str,
    payload: ApiPayload,
):
    threads = []

    clients = []
    print_thread = threading.Thread(target=print_state, args=(clients, num_requests))
    threads.append(print_thread)
    print_thread.start()
    for _ in range(num_requests):
        client = ClientState(
            endpoint_group_name=endpoint_group_name,
            api_key=api_key,
            server_url=server_url,
            worker_endpoint=worker_endpoint,
            payload=payload,
        )
        clients.append(client)
        thread = threading.Thread(target=client.simulate_user, args=())
        thread.daemon = True  # makes threads get killed on program exit
        threads.append(thread)
        thread.start()
        sleep(1 / requests_per_second)
    print("done spawning workers")


def test_load_cmd(
    payload: ApiPayload, endpoint: str, arg_parser: argparse.ArgumentParser
):
    arg_parser.add_argument(
        "-n",
        dest="num_requests",
        type=int,
        required=True,
        help="total number of requests",
    )
    arg_parser.add_argument(
        "-rps",
        dest="requests_per_second",
        type=float,
        required=True,
        help="requests per second",
    )
    args = arg_parser.parse_args()
    if hasattr(args, "comfy_model"):
        os.environ["COMFY_MODEL"] = args.comfy_model
    payload = payload.for_test()
    run_test(
        num_requests=args.num_requests,
        requests_per_second=args.requests_per_second,
        api_key=args.api_key,
        server_url=args.server_url,
        endpoint_group_name=args.endpoint_group_name,
        worker_endpoint=endpoint,
        payload=payload,
    )
