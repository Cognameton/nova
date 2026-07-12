#!/usr/bin/env bash
# Phase 22 Stage 22.2 — daily session-id rotation wrapper for the Nova
# daemon.
#
# NovaDaemon itself runs with one fixed --daemon-session-id for its
# whole process lifetime; it does not rotate sessions internally. This
# script is the systemd ExecStart target: it runs indefinitely,
# restarting the daemon once per calendar day with a fresh session id
# (nova-live-YYYYMMDD), which is what lets cross-session ladder
# evidence (the L1/L2 >= 2/3 distinct sessions requirement) accumulate
# from a live deployment instead of only from operator-run test
# scripts.
#
# Each rotation goes through `--daemon-stop` (the socket-based graceful
# shutdown path -> NovaDaemon._on_sigterm -> GPU release) rather than
# killing the process outright, so the in-flight tick always finishes
# and the model unloads cleanly before the next day's daemon loads it
# again. `--daemon-stop` targets the socket derived from data_dir, not
# from session id, so it always finds the currently running daemon
# regardless of which day's session is active.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${NOVA_LIVE_CONFIG:-configs/nova.qwen3-14b.live.yaml}"
TICK_INTERVAL="${NOVA_LIVE_TICK_INTERVAL:-300}"
PYTHON=.venv/bin/python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

while true; do
    SESSION_ID="nova-live-$(date '+%Y%m%d')"
    log "starting daemon: session=${SESSION_ID} config=${CONFIG} tick_interval=${TICK_INTERVAL}"

    "$PYTHON" -m nova.cli --config "$CONFIG" --daemon \
        --tick-interval "$TICK_INTERVAL" \
        --daemon-session-id "$SESSION_ID" &
    DAEMON_PID=$!

    # Sleep until the next local-midnight rollover, then rotate.
    NOW_EPOCH=$(date +%s)
    MIDNIGHT_EPOCH=$(date -d 'tomorrow 00:00:00' +%s)
    SLEEP_SECONDS=$(( MIDNIGHT_EPOCH - NOW_EPOCH ))
    log "daemon pid=${DAEMON_PID}; sleeping ${SLEEP_SECONDS}s until next day rollover"
    sleep "$SLEEP_SECONDS"

    log "rotating session: stopping ${SESSION_ID}"
    if ! "$PYTHON" -m nova.cli --config "$CONFIG" --daemon-stop; then
        log "daemon-stop failed to connect; sending SIGTERM directly to pid=${DAEMON_PID}"
        kill -TERM "$DAEMON_PID" 2>/dev/null
    fi

    # Bound the wait for clean shutdown before forcing, in case it hangs.
    for _ in $(seq 1 30); do
        kill -0 "$DAEMON_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
        log "daemon did not exit within 30s; sending SIGKILL"
        kill -KILL "$DAEMON_PID" 2>/dev/null
    fi
    wait "$DAEMON_PID" 2>/dev/null
    log "daemon for ${SESSION_ID} stopped; rotating to next day"
done
