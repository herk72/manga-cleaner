#!/bin/bash
# تشغيل IOPaint server (لازم تثبته أول مرة)
# pip install iopaint
# أو: pip install lama-cleaner

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

# تشغيل IOPaint على بورت 8080
iopaint start \
  --model=lama \
  --device=cuda \
  --port=8080 \
  --host=127.0.0.1

# لو مفيش GPU استخدم:
# iopaint start --model=lama --device=cpu --port=8080 --host=127.0.0.1
