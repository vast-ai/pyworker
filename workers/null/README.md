# Null PyWorker

A PyWorker that does **nothing** — it does not forward requests to any model
server. Each HTTP POST to `/reserve` simply marks the worker as busy and holds
the request open until the user's queue consumer (running locally on the
instance) calls `/release` on the internal control port — or a safety
timeout elapses.

## When to use it

Use this worker when you want to drive Vast Serverless autoscaling but you do
**not** want inbound requests to reach a model on the instance. Typical setup:

- You already have a job queue on your own infrastructure (Redis, SQS, NATS,
  etc.).
- A separate worker process on the Vast instance pulls work from that queue
  directly. The Vast PyWorker is not involved in the request/response path.
- You want one Vast worker per active queue consumer, and you want the
  Serverless autoscaler to spin instances up and down based on demand on
  *your* side.

A request comes in and you get a worker. Release and it scales back down.

POST to `/reserve` and serverless gives you a worker, held busy for the
lifetime of the request. When your queue consumer is done, POST to
`/release` on the internal port (`127.0.0.1:18999` by default) and the
held `/reserve` returns `200`.

## How it works

- `allow_parallel_requests=False` and `max_queue_time=0.0`, so one in-flight
  `/reserve` fully occupies the worker and any further request that lands
  on it is rejected with `429` immediately — serverless will route to a
  free worker or scale a new one up.
- `lifecycle` is used instead of `model_log_file`, so there is no log to tail
  and no model server to start. The worker reports itself ready immediately
  after the (trivial) benchmark.
- The `/reserve` handler is a `remote_function` rather than an HTTP proxy, so
  the framework never tries to forward the request anywhere — it just awaits
  an internal `asyncio.Event`.
- An internal aiohttp control server, bound to `127.0.0.1`, hosts
  `/release` (and, when no external healthcheck URL is provided, a stub
  `/health`).

## Healthchecking

The framework periodically GETs a healthcheck URL after startup; if it ever
fails after the first success, the worker is marked errored and the
autoscaler can decommission it. Two modes:

- **Stub (default)** — the internal control server also answers
  `GET /health` with `200`. This is just enough to satisfy the framework
  while you wire up real consumers.
- **Point at your queue consumer (recommended)** — set
  `BACKEND_HEALTH_URL=http://127.0.0.1:9090/health` (absolute URL) and the
  pyworker will healthcheck *your* consumer instead. If your consumer
  process crashes, the autoscaler will see the worker as broken.

Run your queue consumer on the instance alongside the PyWorker, expose a
plain `/health` endpoint on it, then set `BACKEND_HEALTH_URL` accordingly in
your template.

## API

### `POST /reserve`  (external port, signed by the autoscaler)

Holds the worker busy until the reservation ends.

Request body (all fields optional):

```json
{ "duration": 600 }
```

- `duration` (seconds, optional): safety cap on how long to hold the
  reservation if no `/release` arrives. Capped by `MAX_RESERVATION_SECONDS`
  (env var, default 3600). If omitted, defaults to that cap.

Behavior:

- Returns `200` with `{"released": "explicit", ...}` when the local consumer
  POSTs `/release` on the internal port. **This is the intended happy path
  — the request is counted as a success in metrics.**
- Returns `200` with `{"released": "duration_elapsed", "duration": <n>}` if
  the duration cap fires (safety net for a stuck consumer).
- Returns `499` if the external client disconnects (counted as cancelled in
  metrics — avoid this; use `/release` instead).
- Returns `429` immediately if the worker is already holding a reservation
  (so serverless routes the request to a free worker instead of queueing).

### `POST /release`  (internal port, localhost-only)

Marks the active reservation as done. No body required. Idempotent:

```bash
curl -X POST http://127.0.0.1:18999/release
```

Responses:

- `200 {"released": true}` — active reservation was released; the held
  `/reserve` will return `{"released": "explicit"}`.
- `200 {"released": false, "reason": "no active reservation"}` — nothing was
  in flight, no-op.

Only processes on the Vast instance can reach this port. There is no
authentication on it.

## Environment variables

- `MAX_RESERVATION_SECONDS` — upper bound on how long a single `/reserve`
  call can hold a worker if `/release` is never called. Defaults to `3600`.
- `BACKEND_HEALTH_URL` — absolute URL the framework should healthcheck
  (e.g. `http://127.0.0.1:9090/health`). When set, the stub `/health` route
  is not registered on the internal server. When unset, the built-in stub
  is used.
- `NULL_CONTROL_PORT` — port for the internal control server (hosts
  `/release` and optionally `/health`). Defaults to `18999`.

## Deploying on Vast Serverless

1. Create a Serverless endpoint and point `PYWORKER_REPO` at this repository
   (or your fork).
2. Set `BACKEND=null` in the template so `start_server.sh` runs
   `workers.null.worker`.
3. There is no model server to configure; you can omit model-related env vars
   entirely.
4. Run your own queue-consumer process on the instance alongside the
   PyWorker. When the consumer finishes its work it should:
   ```bash
   curl -X POST http://127.0.0.1:18999/release
   ```
   so the held `/reserve` returns success and the autoscaler can scale the
   worker down cleanly.

## Client example

```bash
python -m workers.null.client --endpoint <ENDPOINT_NAME> --duration 600
```

This POSTs once to `/reserve`, which causes exactly one worker to be
provisioned (if none is free) and held busy. To exercise the full flow,
shell into the worker and run `curl -X POST http://127.0.0.1:18999/release`
— the client will return with `{"released": "explicit", ...}`.

## Notes and caveats

- The HTTP connection from the external caller must stay open for the full
  reservation. Make sure your client and any intermediate proxies allow
  long-lived requests (disable idle timeouts, retries, and connection
  reuse if necessary).
- If your client retries on timeout, you may end up provisioning duplicate
  workers. Configure `duration` generously and rely on `/release` from the
  consumer to end reservations promptly.
- Avoid disconnecting the external `/reserve` request as a way to release —
  that produces a `499` and is counted as a cancellation in Vast metrics.
  Always release via `POST /release` on the internal port.
- There is no streaming / heartbeat in the response; the request returns
  exactly once, when the reservation ends.
