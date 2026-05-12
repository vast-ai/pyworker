# Null PyWorker

Holds Vast Serverless reservations open without forwarding any work to a
model. Use it when your real workload (a queue consumer in any language)
runs as a separate process on the instance and you just want to drive
Vast autoscaling: **one POST reserves a worker, one POST releases it.**

## Use case

You have a job queue on your own infrastructure (Redis, SQS, NATS, etc.)
and a consumer (node, golang, python, a binary — anything) that pulls
from it. You want one Vast worker per unit of in-flight work, scaling
elastically from zero. The null PyWorker is the autoscaling driver; your
consumer does the work.

## How it works

Reservations use the framework's session API. The SDK's
`endpoint.session(...)` POSTs `/session/create` to reserve a worker;
`session.close()` POSTs `/session/end` to release it. `max_sessions=1`
means each worker holds exactly one reservation — the next reservation
either lands on a free worker or triggers a scale-up.

The PyWorker itself does nothing functional:

- One trivial `/ping` route to satisfy the framework's benchmark
  requirement (its `max_perf` is pinned to 100).
- An internal `/release` endpoint on `127.0.0.1:18999` for the local
  consumer to end the session without needing `session_auth`.

## Endpoint parameters

Tested working configuration:

| Parameter | Value | Why |
|---|---|---|
| `target_util` | `1.0` | One session = one worker. Default `0.9` rounds up to an extra worker. |
| `min_load` | `0` | Scale-to-zero floor. |
| `max_queue_time` | `1` | Stop routing to an occupied worker after ~1s of implied queue. |
| `target_queue_time` | `0.5` | Trigger scale-up promptly once anything queues. |
| `inactivity_timeout` | `10` (seconds) | Permit scale-to-zero after 10s idle. |

## API

| Route | Where | Use |
|---|---|---|
| `POST /session/create` | endpoint, signed | Reserve a worker (`endpoint.session(...)`) |
| `POST /session/end` | endpoint, signed | Release (`session.close()`) |
| `POST /release` | `127.0.0.1:18999`, no auth | Local consumer release, no `session_auth` needed |

## Healthcheck

Default: stub on `127.0.0.1:18999/health` returning `200`. Set
`BACKEND_HEALTH_URL=http://127.0.0.1:9090/health` (absolute URL) to point
the framework at your queue consumer's health endpoint instead — if the
consumer dies, the autoscaler sees the worker as broken.

## Deploying

1. Point `PYWORKER_REPO` at this repo (or your fork).
2. Set `BACKEND=null` in the template.
3. Run your queue consumer alongside the PyWorker. When it's done with
   a unit of work:
   ```bash
   curl -X POST http://127.0.0.1:18999/release
   ```

## Client demo

```bash
# Single reservation, hold 180s
python -m workers.null.client --endpoint <NAME> --instance alpha

# Three concurrent reservations, started 30s apart, each held 360s
python -m workers.null.client --endpoint <NAME> --instance alpha --count 3 --hold 360
```

Flags: `--count` (number of concurrent sessions, default 1), `--hold`
(seconds each session is held, default 180), `--interval` (seconds
between starts when `--count > 1`, default 30), `--cost` (cost reported
at session-create, default 100 = `max_perf`), `--instance` (`prod` |
`alpha` | `candidate` | `local`).

## Environment variables

- `BACKEND_HEALTH_URL` — absolute URL the framework healthchecks. Stub
  is used when unset.
- `NULL_CONTROL_PORT` — internal control server port. Defaults to `18999`.
