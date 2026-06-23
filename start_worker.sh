#!/bin/bash
# تشغيل Celery worker
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
celery -A tasks.worker.celery_app worker --loglevel=info --concurrency=2 -Q celery
