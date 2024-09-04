import os
import time
import logging
import json
from asyncio import sleep
from dataclasses import dataclass, asdict, field
from functools import cache
from urllib.parse import urljoin

import requests

from lib.data_types import AutoScalaerData, SystemMetrics, ModelMetrics
from typing import Awaitable, NoReturn

METRICS_UPDATE_INTERVAL = 1

log = logging.getLogger(__file__)


@cache
def get_url() -> str:
    use_ssl = os.environ.get("USE_SSL", "false") == "true"
    worker_port = os.environ[f"VAST_TCP_PORT_{os.environ['WORKER_PORT']}"]
    public_ip = os.environ["PUBLIC_IPADDR"]
    return f"http{'s' if use_ssl else ''}://{public_ip}:{worker_port}"


@dataclass
class Metrics:
    last_metric_update: float = 0.0
    update_pending: bool = False
    id: int = field(default_factory=lambda: int(os.environ["CONTAINER_ID"]))
    report_addr: str = field(default_factory=lambda: os.environ["REPORT_ADDR"])
    url: str = field(default_factory=get_url)
    system_metrics: SystemMetrics = field(default_factory=SystemMetrics.empty)
    model_metrics: ModelMetrics = field(default_factory=ModelMetrics.empty)

    def _request_start(self, workload: float, reqnum: int) -> None:
        """
        this function is called prior to forwarding a request to a model API.
        """
        log.debug("request start")
        self.model_metrics.workload_pending += workload
        self.model_metrics.workload_received += workload
        self.model_metrics.requests_recieved.add(reqnum)
        self.model_metrics.requests_working.add(reqnum)

    def _request_end(
        self, workload: float, req_response_time: float, reqnum: int
    ) -> None:
        """
        this function is called after a response from model API is received.
        """
        self.model_metrics.workload_served += workload
        self.model_metrics.workload_pending -= workload
        self.model_metrics.requests_working.discard(reqnum)
        self.model_metrics.cur_perf = workload / req_response_time
        self.update_pending = True

    def _request_errored(self, workload: float, reqnum: int) -> None:
        """
        this function is called if model API returns an error
        """
        self.model_metrics.workload_pending -= workload
        self.model_metrics.workload_errored += workload
        self.model_metrics.requests_working.discard(reqnum)

    def _request_canceled(self, workload: float, reqnum: int) -> None:
        """
        this function is called if client drops connection before model API has responded
        """
        self.model_metrics.workload_pending -= workload
        self.model_metrics.workload_cancelled += workload
        self.model_metrics.requests_working.discard(reqnum)

    async def _send_metrics_loop(self) -> Awaitable[NoReturn]:
        while True:
            await sleep(METRICS_UPDATE_INTERVAL)
            elapsed = time.time() - self.last_metric_update
            if self.system_metrics.model_is_loaded is False and elapsed >= 10:
                log.debug(f"sending loading model metrics after {int(elapsed)}s wait")
                self.__send_metrics_and_reset(elapsed)
            elif self.update_pending or elapsed > 10:
                log.debug(f"sending loaded model metrics after {int(elapsed)}s wait")
                self.__send_metrics_and_reset(elapsed)

    def _model_loaded(self, max_throughput: float) -> None:
        self.system_metrics.model_loading_time = (
            time.time() - self.system_metrics.model_loading_start
        )
        self.system_metrics.model_is_loaded = True
        self.model_metrics.max_throughput = max_throughput

    def _model_errored(self, error_msg: str) -> None:
        self.model_metrics.set_errored(error_msg)
        self.system_metrics.model_is_loaded = True

    #######################################Private#######################################

    def __send_metrics_and_reset(self, elapsed):

        def compute_autoscaler_data() -> AutoScalaerData:
            return AutoScalaerData(
                id=self.id,
                loadtime=(self.system_metrics.model_loading_time or 0.0),
                cur_load=(self.model_metrics.workload_processing / elapsed),
                max_perf=self.model_metrics.max_throughput,
                cur_perf=self.model_metrics.cur_perf,
                error_msg=self.model_metrics.error_msg or "",
                num_requests_working=len(self.model_metrics.requests_working),
                num_requests_recieved=len(self.model_metrics.requests_recieved),
                additional_disk_usage=self.system_metrics.additional_disk_usage,
                cur_capacity=0,
                max_capacity=0,
                url=self.url,
            )

        def send_data() -> None:
            data = compute_autoscaler_data()
            full_path = urljoin(self.report_addr, "/worker_status/")
            log.debug(
                "\n".join(
                    [
                        "#" * 60,
                        f"sending data to autoscaler",
                        f"{json.dumps((asdict(data)), indent=2)}",
                        "#" * 60,
                    ]
                )
            )
            for attempt in range(1, 4):
                try:
                    requests.post(full_path, json=asdict(data), timeout=1)
                    break
                except requests.Timeout:
                    log.debug(f"autoscaler status update timed out")
                except Exception as e:
                    log.debug(f"autoscaler status update failed with error: {e}")
                time.sleep(2)
                log.debug(f"retrying autoscaler status update, attempt: {attempt}")

        ###########

        self.system_metrics.update_disk_usage()
        send_data()
        self.update_pending = False
        self.model_metrics.reset()
        self.system_metrics.reset()
        self.last_metric_update = time.time()
