#!/bin/bash
# تشغيل FastAPI server
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
