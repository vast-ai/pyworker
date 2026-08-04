# Null PyWorker (proof of concept)

> **PoC — not GA.** It uses only the public session API, but three things are
> intentionally left open (see *Open items*): the reservation lifetime must be sized up
> front, health defaults to a stub, and the customer owns their own credentials. Read the
> design note before taking this to release.

Holds a Vast Serverless reservation open without forwarding any work to a model. Use it
when your real workload — a queue consumer in any language — already exists and you just
want Vast to drive autoscaling: **the client reserves a worker, and releases it when done.**

## Use case

You have a job queue on your own infrastructure (Redis, SQS, NATS, …) and a consumer
(node, go, python, a binary — anything). You want one Vast worker per unit of in-flight
work, scaling elastically from zero, without rewriting your consumer. The null PyWorker is
the autoscaling driver; your consumer does the work.

## How it works

Reservations use the public session API. `endpoint.session(...)` POSTs `/session/create`
to reserve a worker; `Session.close()` (or leaving the `async with` block) POSTs
`/session/end` to release it. `max_sessions=1` means each worker holds exactly one
reservation — the next reservation lands on a free worker or triggers a scale-up.

The PyWorker itself does nothing functional: one trivial `/ping` route to satisfy the
framework's benchmark requirement (its `max_perf` is pinned to 100). **It does not manage
sessions itself** — reserve and release are driven by the client that holds `session_auth`,
off-instance. There is no on-box release endpoint (the SDK has no public worker-side close).

## Release model

The reserving client owns the session and releases it with the public `Session.close()`
(the demo client's `async with session` already does this). Drive it from your queue
dispatcher: reserve when a unit of work starts, close when it finishes — or hold across
consecutive units and close only when the queue drains.

## Credentials

The customer owns credential management; nothing long-lived is placed on the rented box:

1. Your dispatcher holds real creds in **volatile memory** on your own infra.
2. Reserving yields the short-lived channel credential: Vast mints `session_auth` +
   `worker_url` on `endpoint.session()`. No pre-shared secret goes to the box.
3. Mint a **short-lived, single-job scoped token** from your in-memory creds and hand it to
   the box over the session channel; the box uses it for the data plane, then it expires.

Residual risk: an untrusted host could exfiltrate the token during its TTL — keep the scope
tight and the TTL short, and consider verified-hosts-only for sensitive workloads.

## Open items (before GA)

- **Lifetime** renews only on a forwarded request; this worker forwards none, so size
  `lifetime` to your job or use the hybrid dispatch model (work rides `Session.request()`,
  which renews the TTL and lets Vast see real load). *SDK ask: a renew primitive.*
- **Health** defaults to a stub that always returns 200 and hides a wedged consumer. Set
  `BACKEND_HEALTH_URL` to a real readiness endpoint. *Consider making it mandatory for GA.*
- **Perf pin** uses the SDK-internal `.has_benchmark` file. *SDK ask: a public fixed-perf
  config.*

## Endpoint parameters

Tested configuration:

| Parameter | Value | Why |
|---|---|---|
| `target_util` | `1.0` | One session = one worker. Default `0.9` rounds up to an extra worker. |
| `min_load` | `0` | Scale-to-zero floor. |
| `max_queue_time` | `1` | Stop routing to an occupied worker after ~1s of implied queue. |
| `target_queue_time` | `0.5` | Trigger scale-up promptly once anything queues. |
| `inactivity_timeout` | `10` (seconds) | Permit scale-to-zero after 10s idle. |

## Deploying

1. Point `PYWORKER_REPO` at this repo (or your fork); set `BACKEND=null` in the template.
2. Set `BACKEND_HEALTH_URL` to your consumer's readiness endpoint.
3. Run your consumer alongside the PyWorker, and drive reserve/release from your dispatcher
   (see the client demo).

## Client demo

```bash
# Single reservation, hold 180s
python -m workers.null.client --endpoint <NAME>

# Three concurrent reservations, started 30s apart, each held 360s
python -m workers.null.client --endpoint <NAME> --count 3 --hold 360
```

Flags: `--count` (concurrent sessions, default 1), `--hold` (seconds held, default 180),
`--interval` (seconds between starts when `--count > 1`, default 30), `--cost` (cost at
session-create, default 100 = `max_perf`).

## Environment variables

- `BACKEND_HEALTH_URL` — absolute URL the framework healthchecks. Stub used when unset (PoC).
- `NULL_CONTROL_PORT` — stub-health server port. Defaults to `18999`.
