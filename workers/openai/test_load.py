from lib.test_utils import test_args
from utils.endpoint_util import Endpoint
from utils.ssl import get_cert_file_path
from lib.data_types import AuthData
from .data_types.server import CompletionsData

import os
import time
import threading
import requests
from dataclasses import dataclass
from collections import Counter
from urllib.parse import urljoin, urlparse
import re

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from requests.adapters import HTTPAdapter

def get_incremented_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    i = 1
    while os.path.exists(f"{base}-{i}{ext}"):
        i += 1
    return f"{base}-{i}{ext}"

WORKER_ENDPOINT = "/v1/completions" # This will return the full text output at once. Latency metrics reflect that (ie not measuring TTFT)

@dataclass
class ReqResult:
    worker_url: str
    route_ms: float
    worker_ms: float
    total_ms: float
    ok: bool
    error: str = ""
    status_code: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    workload: float = 0.0

def do_one(endpoint_name: str,
           endpoint_id: int,
           endpoint_api_key: str,
           server_url: str,
           worker_endpoint: str,
           payload,
           results_list,
           t0,
           status_samples,
           route_session,
           worker_session):
    try:
        workload = payload.count_workload()
        route_payload = {"endpoint": endpoint_name, "api_key": endpoint_api_key, "cost": workload}
        headers = {"Authorization": f"Bearer {endpoint_api_key}"}
        start = time.time()
        r0 = route_session.post(urljoin(server_url, "/route/"), json=route_payload, headers=headers, timeout=4)
        t_after_route = time.time()
        if r0.status_code != 200:
            results_list.append(ReqResult(worker_url="", 
                                            route_ms=(t_after_route - start) * 1000.0, 
                                            worker_ms=0.0, 
                                            total_ms=(t_after_route - start) * 1000.0, 
                                            ok=False, 
                                            error=f"route error {r0.reason} {r0.text}",
                                            status_code=r0.status_code,
                                            t_start=start - t0, 
                                            t_end=t_after_route - t0, 
                                            workload=workload))
            return
        msg = r0.json()

        # 1) Check if we got a worker back from route
        worker_url = msg.get("url", "")
        if not worker_url:
            status = msg.get("status", "")
            m = re.search(r"total workers:\s*(\d+).*loading workers:\s*(\d+).*standby workers:\s*(\d+).*error workers:\s*(\d+)", status, re.I | re.S)
            if m:
                tot, loading, standby, err = map(int, m.groups())
                idle = max(tot - loading - standby - err, 0)
                status_samples.append((time.time() - t0, idle))

        # 2) If we got a worker, send the request
        if worker_url:
            req = dict(payload=payload.__dict__, auth_data=AuthData.from_json_msg(msg).__dict__)
            t_before_worker = time.time()
            r1 = worker_session.post(
                urljoin(worker_url, worker_endpoint),
                json=req,
                verify=get_cert_file_path(),
                timeout=(4, 120),
            )
            t_after_worker = time.time()
            if r1.status_code != 200:
                results_list.append(ReqResult(worker_url=worker_url, 
                                                route_ms=(t_after_route - start) * 1000.0, 
                                                worker_ms=(t_after_worker - t_before_worker) * 1000.0,
                                                total_ms=(t_after_worker - start) * 1000.0, 
                                                ok=False,
                                                error=f"worker inference error {r1.reason} {r1.text}",
                                                status_code=r1.status_code,
                                                t_start=start - t0, 
                                                t_end=t_after_worker - t0, 
                                                workload=workload))
                return
            # Success case
            results_list.append(ReqResult(worker_url=worker_url, 
                                            route_ms=(t_after_route - start) * 1000.0, 
                                            worker_ms=(t_after_worker - t_before_worker) * 1000.0, 
                                            total_ms=(t_after_worker - start) * 1000.0,
                                            ok=True,
                                            error="",
                                            status_code=200,
                                            t_start=start - t0, 
                                            t_end=t_after_worker - t0, 
                                            workload=workload))

        # 3) If so, sample via /get_endpoint_workers/ for eligible (idle) worker tracking
        if worker_url:
            try:
                r_status = route_session.post(
                    urljoin(server_url, "/get_endpoint_workers/"),
                    json={"id": endpoint_id},
                    headers={"Authorization": f"Bearer {endpoint_api_key}"},
                    timeout=3,
                )
                if r_status.status_code == 200:
                    workers = r_status.json()
                    idle = 0
                    for w in workers:
                        st = str(w.get("status", "")).lower()
                        if (st in ("idle")):
                            idle += 1
                    status_samples.append((time.time() - t0, idle))
            except Exception:
                pass
    except Exception as e:
        t = time.time()
        results_list.append(ReqResult(worker_url="", 
                                        route_ms=0.0, 
                                        worker_ms=0.0, 
                                        total_ms=0.0, 
                                        ok=False, 
                                        error=f"unknown error {e}",
                                        status_code=0,
                                        t_start=t - t0, 
                                        t_end=t - t0, 
                                        workload=0.0))

def run_load_with_metrics(num_requests: int,
                          requests_per_second: float,
                          endpoint_group_name: str,
                          account_api_key: str,
                          server_url: str,
                          worker_endpoint: str,
                          instance: str,
                          out_path: str):

    ep_info = Endpoint.get_endpoint_info(endpoint_name=endpoint_group_name,
                                         account_api_key=account_api_key,
                                         instance=instance)
    if not ep_info or not ep_info.get("api_key") or not ep_info.get("id"):
        print(f"Endpoint {endpoint_group_name} not found for API key")
        return
    endpoint_id = int(ep_info["id"])
    endpoint_api_key = ep_info["api_key"]

    t0 = time.time()
    results = []
    status_samples = []
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "8192"))
    submit_queue_factor = 2  # cap queued tasks to reduce memory

    # Shared HTTP sessions with connection pooling (persistent connections)
    def make_session(pool_connections: int, pool_maxsize: int) -> requests.Session:
        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize, max_retries=0)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        return sess

    # Router: mostly single host, small connection pool is sufficient
    route_session = make_session(pool_connections=1, pool_maxsize=max_concurrency)
    # Workers: many hosts; allow many pools and per-host concurrency up to max_concurrency
    worker_session = make_session(pool_connections=64, pool_maxsize=max_concurrency // 8)

    # Fire requests using a thread pool, scheduling at requested RPS
    inflight = set()
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        for i in range(num_requests):
            # Pace submissions to RPS
            target_time = t0 + i / max(requests_per_second, 1e-9)
            sleep_s = target_time - time.time()
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.5))  # sleep in chunks to stay responsive

            payload = CompletionsData.for_test()
            fut = executor.submit(
                do_one,
                endpoint_group_name,
                endpoint_id,
                endpoint_api_key,
                server_url,
                worker_endpoint,
                payload,
                results,
                t0,
                status_samples,
                route_session,
                worker_session,
            )
            inflight.add(fut)
            # Prevent unbounded queue growth
            if len(inflight) >= max_concurrency * submit_queue_factor:
                done, not_done = wait(inflight, return_when=FIRST_COMPLETED)
                inflight = not_done
        # Wait for all outstanding tasks
        if inflight:
            wait(inflight)
    # Close sessions
    try:
        route_session.close()
    finally:
        worker_session.close()

    # Aggregate results
    oks = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]
    total_reqs = len(results)
    succ = len(oks)

    total_ms = np.array([r.total_ms for r in oks]) if succ else np.array([])
    worker_ms = np.array([r.worker_ms for r in oks]) if succ else np.array([])
    route_ms = np.array([r.route_ms for r in oks]) if succ else np.array([])

    avg_total = float(np.mean(total_ms)) if succ else 0.0
    avg_worker = float(np.mean(worker_ms)) if succ else 0.0
    avg_route = float(np.mean(route_ms)) if succ else 0.0
    p50_total, p95_total = (float(np.percentile(total_ms, 50)), float(np.percentile(total_ms, 95))) if succ else (0.0, 0.0)

    # Distribution over workers (by host:port)
    hosts = [urlparse(r.worker_url).netloc for r in oks if r.worker_url]
    dist = Counter(hosts)

    # Idle over time (mode per second)
    idle_ts, idle_vals = [], []
    if status_samples:
        buckets = {}
        for ts, idle in status_samples:
            k = int(ts)
            buckets.setdefault(k, []).append(idle)
        keys = sorted(buckets.keys())
        idle_ts = keys
        # Use the most frequent sampled value per second (mode) to keep integer counts
        idle_vals = []
        for k in keys:
            vals_k = [int(v) for v in buckets[k]]
            if vals_k:
                cnt = Counter(vals_k)
                idle_vals.append(cnt.most_common(1)[0][0])
            else:
                idle_vals.append(0)

    print(f"\nResults: total={total_reqs} success={succ} errors={len(errs)}")
    print(f"Avg latency (ms): {avg_total:.1f}  p50: {p50_total:.1f}  p95: {p95_total:.1f}")
    print(f"Avg route latency (ms): {avg_route:.1f}  Avg worker latency (ms): {avg_worker:.1f}")
    if errs:
        print("Sample errors:")
        for e in errs[:5]:
            print(f"  {e.status_code} {e.error}")

    # Plot: 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Load test: {endpoint_group_name}  n={total_reqs}, rps={requests_per_second}, success={succ}")

    # Dist per worker
    ax0 = axes[0, 0]
    if dist:
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        labels, counts = zip(*items)
        ax0.bar(range(len(labels)), counts)
        ax0.set_xticks(range(len(labels)))
        ax0.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax0.set_title("Request distribution over workers")
    ax0.set_ylabel("count")

    # Latency histogram (total)
    ax1 = axes[0, 1]
    if succ:
        ax1.hist(total_ms, bins=30)
    ax1.set_title("Total latency (ms)")
    ax1.set_xlabel("ms")
    ax1.set_ylabel("freq")

    # Eligible workers over time
    ax_idle = axes[0, 2]
    if idle_ts:
        ax_idle.plot(idle_ts, idle_vals, "-o", ms=3)
    ax_idle.set_title("Eligible workers over time")
    ax_idle.set_xlabel("time (s)")
    ax_idle.set_ylabel("eligible count")

    # Throughput over time (completions/sec)
    ax_idle = axes[1, 0]
    ax_idle.clear()
    if succ:
        per_sec = {}
        for r in oks:
            s = int(r.t_end)
            per_sec[s] = per_sec.get(s, 0) + 1
        ts = sorted(per_sec.keys())
        vals = [per_sec[t] for t in ts]
        ax_idle.plot(ts, vals, "-o", ms=3)
    ax_idle.set_title("Completions per second")
    ax_idle.set_xlabel("time (s)")
    ax_idle.set_ylabel("completions / sec")

    # Summary text
    ax3 = axes[1, 1]
    ax3.axis("off")
    text = (
        f"Total requests: {total_reqs}\n"
        f"Success: {succ}  Errors: {len(errs)}\n"
        f"Avg total latency: {avg_total:.1f} ms\n"
        f"p50: {p50_total:.1f} ms  p95: {p95_total:.1f} ms\n"
        f"Avg route latency: {avg_route:.1f} ms\n"  
        f"Avg worker latency: {avg_worker:.1f} ms\n"
        f"300 errors: {len([r for r in errs if r.status_code >= 300 and r.status_code < 400])}\n"
        f"429 errors: {len([r for r in errs if r.status_code == 429])}\n"
        f"500 errors: {len([r for r in errs if r.status_code >= 500])}\n"
        f"Other errors: {len([r for r in errs if r.status_code not in [300, 429, 500]])}\n"
    )
    ax3.set_title("Summary")
    ax3.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, transform=ax3.transAxes)

    # Error count over time
    ax_errors = axes[1, 2]
    all_end_times = [int(r.t_end) for r in results if r.t_end > 0]
    if all_end_times:
        min_second = min(all_end_times)
        max_second = max(all_end_times)
        # Count errors per second
        errors_per_second = {}
        for result in errs:
            second = int(result.t_end)
            errors_per_second[second] = errors_per_second.get(second, 0) + 1
        # Create complete timeline including zeros
        time_seconds = list(range(min_second, max_second + 1))
        error_counts = [errors_per_second.get(sec, 0) for sec in time_seconds]
        ax_errors.plot(time_seconds, error_counts, "-o", ms=3)
    ax_errors.set_title("Errors per second")
    ax_errors.set_xlabel("time (s)")
    ax_errors.set_ylabel("errors / sec")

    # Ensure unique output path and create directory if needed
    final_out_path = get_incremented_path(out_path)
    out_dir = os.path.dirname(final_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(final_out_path, dpi=120)
    print(f"Saved report to: {final_out_path}")

    # Per-worker latency boxplot (top 12 by volume)
    groups = {}
    for r in oks:
        host = urlparse(r.worker_url).netloc
        groups.setdefault(host, []).append(r.total_ms)
    items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:12]
    if items:
        labels, data = zip(*items)
        fig2, axb = plt.subplots(1, 1, figsize=(12, 5))
        axb.boxplot(data, showfliers=False)
        axb.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axb.set_title("Per-worker latency (ms)")
        axb.set_ylabel("ms")
        plt.tight_layout()
        extra_out = get_incremented_path(os.path.splitext(out_path)[0] + "-workers.png")
        plt.savefig(extra_out, dpi=120)
        fig2.tight_layout()
        fig2.savefig(extra_out, dpi=120)
        print(f"Saved worker latency plot to: {extra_out}")

if __name__ == "__main__":
    # Check if MODEL_NAME environment variable is set
    model_name_set = os.environ.get("MODEL_NAME") is not None

    # Add model argument - required only if MODEL_NAME is not set
    test_args.add_argument(
        "--model",
        dest="model",
        required=not model_name_set,
        help="Model to use for completions request (required if MODEL_NAME env var not set)",
    )

    # Parse known args to get model early, before adding load args
    known_args, _ = test_args.parse_known_args()
    if hasattr(known_args, "model") and known_args.model:
        os.environ["MODEL_NAME"] = known_args.model
        print(f"Set MODEL_NAME environment variable to: {known_args.model}")

    # Load test args
    test_args.add_argument("-n", dest="num_requests", type=int, required=True, help="total number of requests")
    test_args.add_argument("-rps", dest="requests_per_second", type=float, required=True, help="requests per second")
    test_args.add_argument("--out", dest="out_path", type=str, default="load_test_report.png", help="path to save the report image")
    args = test_args.parse_args()

    server_url = {
        "prod": "https://run.vast.ai",
        "alpha": "https://run-alpha.vast.ai",
        "candidate": "https://run-candidate.vast.ai",
        "local": "http://localhost:8080"
    }.get(args.instance, "http://localhost:8080")

    run_load_with_metrics(
        num_requests=args.num_requests,
        requests_per_second=args.requests_per_second,
        endpoint_group_name=args.endpoint_group_name,
        account_api_key=args.api_key,
        server_url=server_url,
        worker_endpoint=WORKER_ENDPOINT,
        instance=args.instance,
        out_path=args.out_path,
    )