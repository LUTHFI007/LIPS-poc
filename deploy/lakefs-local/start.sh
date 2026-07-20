#!/usr/bin/env bash
# Start the local MinIO + lakeFS stack (idempotent: skips already-running
# services). Data persists under data/ across restarts. Stop with ./stop.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/bin"
DATA="$HERE/data"
LOGS="$HERE/logs"
BUCKET="lakefs-data"

# shellcheck disable=SC1091
source "$HERE/local.env"

_wait_for() { # url, name
    for _ in $(seq 1 60); do
        curl -fsS -o /dev/null "$1" 2>/dev/null && return 0
        sleep 0.5
    done
    echo "ERROR: $2 did not become healthy at $1" >&2
    return 1
}

# ── MinIO ─────────────────────────────────────────────────────────────────────
if ! pgrep -f "minio server" > /dev/null; then
    echo "Starting MinIO on :9000 (console :9001)..."
    MINIO_ROOT_USER="$MINIO_ROOT_USER" MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
        nohup "$BIN/minio" server "$DATA/minio" \
        --address 127.0.0.1:9000 --console-address 127.0.0.1:9001 \
        >> "$LOGS/minio.log" 2>&1 &
fi
_wait_for "http://127.0.0.1:9000/minio/health/live" "MinIO"

# Bucket for lakeFS object storage (idempotent).
"$BIN/mc" --quiet alias set lipslocal http://127.0.0.1:9000 \
    "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" > /dev/null
"$BIN/mc" --quiet mb --ignore-existing "lipslocal/$BUCKET" > /dev/null

# ── lakeFS ────────────────────────────────────────────────────────────────────
# The LAKEFS_INSTALLATION_* vars create the initial admin user on the very
# first run; on later runs (already set up) they are ignored.
if ! pgrep -f "lakefs run" > /dev/null; then
    echo "Starting lakeFS on :8000..."
    LAKEFS_INSTALLATION_USER_NAME="admin" \
    LAKEFS_INSTALLATION_ACCESS_KEY_ID="$LAKEFS_ACCESS_KEY_ID" \
    LAKEFS_INSTALLATION_SECRET_ACCESS_KEY="$LAKEFS_SECRET_ACCESS_KEY" \
        nohup "$BIN/lakefs" run --config "$HERE/lakefs.yaml" \
        >> "$LOGS/lakefs.log" 2>&1 &
fi
_wait_for "http://127.0.0.1:8000/healthcheck" "lakeFS"

echo "MinIO:  http://127.0.0.1:9000 (console http://127.0.0.1:9001)"
echo "lakeFS: http://127.0.0.1:8000"
