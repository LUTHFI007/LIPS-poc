---
title: LIPS MLflow Tracking Server
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Central MLflow tracking server for the LIPS benchmark POC
---

# LIPS MLflow Tracking Server

A central [MLflow](https://mlflow.org) tracking server for the LIPS Power Grid
benchmark POC, so multiple users log and compare evaluation runs against one
shared store.

- **Backend store:** external Neon Postgres — set via the `MLFLOW_BACKEND_STORE_URI`
  Space **secret** (this is what persists all run metadata across restarts).
- **Artifacts:** local `/tmp` (tiny — only JSON; model weights live in HuggingFace).
- **Port:** 7860 (HF Spaces convention).

The app points at this Space by setting `MLFLOW_TRACKING_URI=<this space's URL>`.
