#!/usr/bin/env bash
# Stop the local MinIO + lakeFS stack. Data under data/ is preserved.
pkill -f "lakefs run" 2>/dev/null && echo "lakeFS stopped." || echo "lakeFS was not running."
pkill -f "minio server" 2>/dev/null && echo "MinIO stopped."  || echo "MinIO was not running."
