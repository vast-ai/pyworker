#!/bin/bash

set -e -o pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

SERVER_DIR="$WORKSPACE_DIR/vast-pyworker"
ENV_PATH="$WORKSPACE_DIR/worker-env"
DEBUG_LOG="$WORKSPACE_DIR/debug.log"
PYWORKER_LOG="$WORKSPACE_DIR/pyworker.log"

REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"
USE_SSL="${USE_SSL:-true}"
WORKER_PORT="${WORKER_PORT:-3000}"
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# make all output go to $DEBUG_LOG and stdout without having to add `... | tee -a $DEBUG_LOG` to every command
exec &> >(tee -a "$DEBUG_LOG")

function echo_var(){
    echo "$1: ${!1}"
}

[ -z "$BACKEND" ] && echo "BACKEND must be set!" && exit 1
[ -z "$MODEL_LOG" ] && echo "MODEL_LOG must be set!" && exit 1
[ -z "$HF_TOKEN" ] && echo "HF_TOKEN must be set!" && exit 1
[ "$BACKEND" = "comfyui" ] && [ -z "$COMFY_MODEL" ] && echo "For comfyui backends, COMFY_MODEL must be set!" && exit 1


echo "start_server.sh"
date

echo_var BACKEND
echo_var REPORT_ADDR
echo_var WORKER_PORT
echo_var WORKSPACE_DIR
echo_var SERVER_DIR
echo_var ENV_PATH
echo_var DEBUG_LOG
echo_var PYWORKER_LOG
echo_var MODEL_LOG

# if instance is rebooted, we want to clear out the log file so pyworker doesn't read lines
# from the run prior to reboot. past logs are saved in $MODEL_LOG.old for debugging only
if [ -e "$MODEL_LOG" ]; then
    echo "Rotating model log at $MODEL_LOG to $MODEL_LOG.old"
    cat "$MODEL_LOG" >> "$MODEL_LOG.old" 
    : > "$MODEL_LOG"
fi

# Populate /etc/environment with quoted values
if ! grep -q "VAST" /etc/environment; then
    env -0 | grep -zEv "^(HOME=|SHLVL=)|CONDA" | while IFS= read -r -d '' line; do
            name=${line%%=*}
            value=${line#*=}
            printf '%s="%s"\n' "$name" "$value"
        done > /etc/environment
fi

if [ ! -d "$ENV_PATH" ]
then
    echo "setting up venv"
    if ! which uv; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.local/bin/env
    fi

    if [[ ! -d $SERVER_DIR ]]; then
        git clone --depth=1 "${PYWORKER_REPO:-https://github.com/vast-ai/pyworker}" "$SERVER_DIR"
    fi

    if [[ -n ${PYWORKER_REF:-} ]]; then
        (
            cd "$SERVER_DIR"
            git fetch --depth=1 origin "$PYWORKER_REF"
            git checkout "$PYWORKER_REF"
        )
    fi

    uv venv --python-preference only-managed "$ENV_PATH" -p 3.10
    source "$ENV_PATH/bin/activate"

    uv pip install -r "${SERVER_DIR}/requirements.txt"

    touch ~/.no_auto_tmux
else
    [[ -f ~/.local/bin/env ]] && source ~/.local/bin/env
    source "$WORKSPACE_DIR/worker-env/bin/activate"
    echo "environment activated"
    echo "venv: $VIRTUAL_ENV"
fi

[ ! -d "$SERVER_DIR/workers/$BACKEND" ] && echo "$BACKEND not supported!" && exit 1

if [ "$USE_SSL" = true ]; then

    cat << EOF > /etc/openssl-san.cnf
    [req]
    default_bits       = 2048
    distinguished_name = req_distinguished_name
    req_extensions     = v3_req

    [req_distinguished_name]
    countryName         = US
    stateOrProvinceName = CA
    organizationName    = Vast.ai Inc.
    commonName          = vast.ai

    [v3_req]
    basicConstraints = CA:FALSE
    keyUsage         = nonRepudiation, digitalSignature, keyEncipherment
    subjectAltName   = @alt_names

    [alt_names]
    IP.1   = 0.0.0.0
EOF

    openssl req -newkey rsa:2048 -subj "/C=US/ST=CA/CN=pyworker.vast.ai/" \
        -nodes \
        -sha256 \
        -keyout /etc/instance.key \
        -out /etc/instance.csr \
        -config /etc/openssl-san.cnf

    curl --header 'Content-Type: application/octet-stream' \
        --data-binary @//etc/instance.csr \
        -X \
        POST "https://console.vast.ai/api/v0/sign_cert/?instance_id=$CONTAINER_ID" > /etc/instance.crt;
fi




export REPORT_ADDR WORKER_PORT USE_SSL UNSECURED

cd "$SERVER_DIR"

echo "launching PyWorker server"

set +e
python3 -m "workers.$BACKEND.server" |& tee -a "$PYWORKER_LOG"
PY_STATUS=${PIPESTATUS[0]}
set -e

if [ "${PY_STATUS}" -ne 0 ]; then
  echo "PyWorker exited with status ${PY_STATUS}; notifying autoscaler..."
  ERROR_MSG="PyWorker exited: code ${PY_STATUS}"
  MTOKEN="${MASTER_TOKEN:-}"
  VERSION="${PYWORKER_VERSION:-0}"

  IFS=',' read -r -a REPORT_ADDRS <<< "${REPORT_ADDR}"
  for addr in "${REPORT_ADDRS[@]}"; do
    curl -sS -X POST -H 'Content-Type: application/json' \
      -d "$(cat <<JSON
{
  "id": ${CONTAINER_ID:-0},
  "mtoken": "${MTOKEN}",
  "version": "${VERSION}",
  "loadtime": 0,
  "new_load": 0,
  "cur_load": 0,
  "rej_load": 0,
  "max_perf": 0,
  "cur_perf": 0,
  "error_msg": "${ERROR_MSG}",
  "num_requests_working": 0,
  "num_requests_recieved": 0,
  "additional_disk_usage": 0,
  "working_request_idxs": [],
  "cur_capacity": 0,
  "max_capacity": 0,
  "url": "${URL}"
}
JSON
)" "${addr%/}/worker_status/" || true
  done
fi

echo "launching PyWorker server done"