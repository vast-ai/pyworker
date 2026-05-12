# Null PyWorker

A PyWorker that does **nothing** — it does not forward requests to any model
server. Reservations are modelled as framework **sessions**: a request
comes in and you get a worker; release and it scales back down.

## When to use it

Use this worker when you want to drive Vast Serverless autoscaling but you do
**not** want inbound requests to reach a model on the instance. Typical setup:

- You already have a job queue on your own infrastructure (Redis, SQS, NATS,
  etc.).
- A separate worker process on the Vast instance pulls work from that queue
  directly. The Vast PyWorker is not involved in the request/response path.
  Your consumer can be any language — node, golang, python, a binary —
  this PyWorker is implementation-agnostic.
- You want one Vast worker per active queue consumer, and you want the
  Serverless autoscaler to spin instances up and down based on demand on
  *your* side.

## How it works

- Reservations use the framework's **session** model. The SDK exposes
  `endpoint.session(cost, lifetime)` which POSTs to `/session/create` (a
  built-in framework route) and returns a `Session` object usable as
  `async with`. Closing the context (or calling `await session.close()`)
  POSTs to `/session/end` — counted as a normal success in metrics.
- `max_sessions=1` on the worker side means a second `/session/create`
  against an already-occupied worker returns `429`. Serverless routes
  that request to a free worker or scales a new one up.
- Sessions are **excluded from queue-wait math** (the framework filters
  `if not request.is_session`), so an occupied worker doesn't look like
  it has a request queue piling up. The autoscaler treats a session as
  occupancy, not as work-in-progress.
- `lifecycle` is used instead of `model_log_file`, so there is no log to
  tail and no model server to start. The worker reports itself ready
  immediately after a trivial benchmark.

## Healthchecking

The framework periodically GETs a healthcheck URL after startup; if it ever
fails after the first success, the worker is marked errored and the
autoscaler can decommission it. Two modes:

- **Stub (default)** — the internal control server also answers
  `GET /health` with `200`. Just enough to satisfy the framework while
  you wire up real consumers.
- **Point at your queue consumer (recommended)** — set
  `BACKEND_HEALTH_URL=http://127.0.0.1:9090/health` (absolute URL) and
  the pyworker will healthcheck *your* consumer instead. If the consumer
  process crashes, the autoscaler will see the worker as broken.

## API

### Reservation: `POST /session/create`  (external, signed)

Not implemented here — the framework provides this route automatically on
every PyWorker. Use the SDK:

```python
from vastai import Serverless

async with Serverless() as client:
    endpoint = await client.get_endpoint(name="my-null-endpoint")
    async with endpoint.session(cost=100, lifetime=600) as s:
        # Worker is now reserved. Your queue dispatcher does whatever it
        # needs to do (typically: enqueue a job that mentions s.session_id).
        ...
    # `async with` exit posts to /session/end → 200 success in metrics
```

Or raw HTTP (the SDK takes care of autoscaler signing for you, but the
shape of the request is documented for non-Python clients):

```
POST /session/create
{
  "auth_data": { /* signed by autoscaler */ },
  "payload": {
    "lifetime": 600,
    "on_close_route": "https://your.callback/notify",
    "on_close_payload": {"job_id": "..."}
  }
}
```

### Release from a local consumer: `POST /release`  (internal, localhost-only)

Closes the active session, regardless of who created it. No body, no
auth. Use this when the queue consumer doesn't have (and shouldn't need)
the session's `session_auth`:

```bash
curl -X POST http://127.0.0.1:18999/release
```

Responses:

- `200 {"released": true, "session_ids": ["..."]}` — closed; the held
  client-side `/session/create` completes and counts as a success.
- `200 {"released": false, "reason": "no active session"}` — nothing
  active, no-op.

For setups where the dispatcher can hand the consumer `session_auth`
(e.g. as part of the queue payload), the consumer can instead POST
`/session/end` on the framework's HTTP-only port
(`$WORKER_HTTP_PORT`, default `WORKER_PORT+1`) — the standard, fully
authenticated release path.

## Environment variables

- `BACKEND_HEALTH_URL` — absolute URL the framework should healthcheck
  (e.g. `http://127.0.0.1:9090/health`). When set, the stub `/health`
  route is not registered on the internal server.
- `NULL_CONTROL_PORT` — port for the internal control server (hosts
  `/release` and optionally `/health`). Defaults to `18999`.

## Deploying on Vast Serverless

1. Create a Serverless endpoint and point `PYWORKER_REPO` at this
   repository (or your fork).
2. Set `BACKEND=null` in the template so `start_server.sh` runs
   `workers.null.worker`.
3. There is no model server to configure; you can omit model-related env
   vars entirely.
4. Run your own queue-consumer process on the instance alongside the
   PyWorker. When it finishes its work:
   ```bash
   curl -X POST http://127.0.0.1:18999/release
   ```

### Endpoint scaling parameters

The null worker reports `max_perf = 100` and each reservation is a
session of `cost = 100`. Set the endpoint accordingly:

- **`target_util = 1.0`** — required. The default of `0.9` reserves
  ~11% spare capacity, which for a unit-occupancy worker rounds up to a
  whole extra worker (e.g. `min_load = 100` becomes `100 / 0.9 = 111.1`
  → 2 active workers instead of 1). With `target_util = 1.0` the math
  is clean: `min_load = 100 * N` keeps exactly `N` workers active.
- **`min_load`** — set to `100 * N` for `N` always-on workers (with
  `target_util = 1.0`).
- **`max_workers`** — cap on total reservations the endpoint can ever
  serve concurrently.
- **`max_queue_time` / `target_queue_time`** — leave at defaults. Both
  operate on per-worker `wait_time`, which is computed *excluding*
  sessions (`backend.py:510`, `data_types.py:307-317`), so a worker
  holding a reservation reports `wait_time = 0.0`. Tuning these does
  not change null-worker scaling — additional reservations land or
  miss based on the `max_sessions = 1` rejection (429), not queue
  time.
- **`inactivity_timeout`** — works as expected: idle (no active
  sessions) for N seconds → permitted to scale down past `min_load`.

## Client example

Single reservation (holds for 180s):

```bash
python -m workers.null.client --endpoint <ENDPOINT_NAME>
```

Staggered demo:

```bash
python -m workers.null.client --endpoint <ENDPOINT_NAME> --demo
```

Starts three sessions 30s apart (all held concurrently), holds the
3-worker plateau for 5 minutes so the autoscaler has time to actually
provision the third worker before any scale-down starts, then closes
the sessions one at a time, also 30s apart, and exits. Every session
ends cleanly via the SDK's `session.close()` — `200` successes in
metrics, no cancellations.

Tune the timing with `--interval` and `--plateau`. To exercise the
local-release path, shell into a worker and run
`curl -X POST http://127.0.0.1:18999/release`.

## Notes and caveats

- The reservation's lifetime caps how long the session can live without
  client activity. Set it comfortably longer than the work you expect to
  do, or have the client periodically POST `/ping` with `session_id` to
  extend.
- The `on_close_route` payload (passed at `/session/create`) is POSTed by
  the framework when the session ends. Useful for notifying your queue
  consumer that the reservation is closing.
- `/release` on the internal port is convenient but bypasses
  `session_auth`. If you need the standard authenticated release flow,
  pass `session_auth` to your consumer (e.g. through the queue payload)
  and have it POST to `/session/end` on the framework's HTTP port
  instead.
