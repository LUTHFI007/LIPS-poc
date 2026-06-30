#!/usr/bin/env bash
set -e

# The Neon Postgres connection string comes from the Space secret
# MLFLOW_BACKEND_STORE_URI (must start with postgresql://, not postgres://).
if [ -z "${MLFLOW_BACKEND_STORE_URI}" ]; then
  echo "FATAL: MLFLOW_BACKEND_STORE_URI is not set. Add it as a Space secret." >&2
  exit 1
fi

exec mlflow server \
  --host 0.0.0.0 --port 7860 \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --artifacts-destination /tmp/mlartifacts \
  --serve-artifacts
