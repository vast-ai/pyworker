# Null PyWorker

A PyWorker that does **nothing** — it does not forward requests to any model
server. Each HTTP POST to `/reserve` simply marks the worker as busy and holds
the request open until the caller disconnects (or a configured timeout
elapses).

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

For each job your side wants to run on a Vast instance, you POST once to
`/reserve`. The autoscaler will provision a worker if none is free; the
request stays open, keeping that worker counted as busy, until you close the
connection. When you close, the worker goes idle and the autoscaler is free
to scale it down.

## How it works

- `allow_parallel_requests=False`, so one in-flight `/reserve` fully occupies
  the worker. Any second request that lands on the same worker queues (or is
  rejected with `429` after `max_queue_time`), pushing the autoscaler to
  provision more workers.
- `lifecycle` is used instead of `model_log_file`, so there is no log to tail
  and no model server to start. The worker reports itself ready immediately
  after the (trivial) benchmark.
- The handler is a `remote_function` rather than an HTTP proxy, so the
  framework never tries to forward the request anywhere.

## API

### `POST /reserve`

Holds the worker busy for the lifetime of the request.

Request body (all fields optional):

```json
{ "duration": 60 }
```

- `duration` (seconds, optional): how long to hold the reservation if the
  client does not disconnect first. Capped by `MAX_RESERVATION_SECONDS` (env
  var, default 3600). If omitted, defaults to the cap.

Behavior:

- Returns `200` with `{"released": "duration_elapsed", "duration": <n>}` when
  the duration elapses normally.
- Returns `499` when the client disconnects (the reservation is released
  immediately).
- Returns `429` if the worker is already busy and queue wait would exceed
  `max_queue_time` (30s by default).

## Environment variables

- `MAX_RESERVATION_SECONDS` — upper bound on how long a single `/reserve`
  call can hold a worker. Defaults to `3600`. Set lower if you want a tighter
  safety cap against stuck clients.

## Deploying on Vast Serverless

1. Create a Serverless endpoint and point `PYWORKER_REPO` at this repository
   (or your fork).
2. Set `BACKEND=null` in the template so `start_server.sh` runs
   `workers.null.worker`.
3. There is no model server to configure; you can omit model-related env vars
   entirely.
4. Run your own queue-consumer process on the instance alongside the
   PyWorker (e.g. as a separate supervisor service started by the template).

## Client example

```bash
python -m workers.null.client --endpoint <ENDPOINT_NAME> --duration 300
```

This will POST once to `/reserve`, which causes exactly one worker to be
provisioned (if none is free) and held busy for up to 300 seconds. Killing
the client process (Ctrl-C) drops the connection and releases the worker
early.

## Notes and caveats

- The HTTP connection must stay open for the full reservation. Make sure
  your client and any intermediate proxies allow long-lived requests
  (disable idle timeouts, retries, and connection reuse if necessary).
- If your client retries on timeout, you may end up provisioning duplicate
  workers. Use idempotent semantics in *your* queue, or set `duration` to a
  finite value and accept release-on-elapse as the normal exit.
- There is no streaming / heartbeat in the response; the request returns
  exactly once, when the reservation ends.
