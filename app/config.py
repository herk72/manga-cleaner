import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_API_ID = int(os.environ["TELEGRAM_API_ID"])  # Telethon accepts int
TELEGRAM_API_HASH = os.environ["TELEGRAM_API_HASH"]

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "downloads")
TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

for d in [UPLOAD_DIR, DOWNLOAD_DIR, TEMP_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

IOPAINT_URL = os.getenv("IOPAINT_URL", "http://localhost:8080")

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
CELERY_TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "600"))
