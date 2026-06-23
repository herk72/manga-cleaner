"""
Celery worker — handles async page/chapter cleaning tasks.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from celery import Celery
from app.config import REDIS_URL, TEMP_DIR, DOWNLOAD_DIR

celery_app = Celery(
    "manga_cleaner",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=600,
    task_soft_time_limit=540,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    result_expires=3600,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="tasks.clean_chapter")
def clean_chapter_task(self, job_id: str, zip_path: str):
    """Process a zip file of manga pages asynchronously."""
    from utils.pipeline import process_chapter_zip

    output_dir = os.path.join(DOWNLOAD_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    self.update_state(state="PROGRESS", meta={"status": "processing", "job_id": job_id})
    result = process_chapter_zip(zip_path, output_dir)
    result["job_id"] = job_id
    return result


@celery_app.task(bind=True, name="tasks.clean_images")
def clean_images_task(self, job_id: str, image_paths: list):
    """Process a list of individual image files asynchronously."""
    from utils.pipeline import process_image_list

    output_dir = os.path.join(DOWNLOAD_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    self.update_state(state="PROGRESS", meta={"status": "processing", "job_id": job_id})
    result = process_image_list(image_paths, output_dir)
    result["job_id"] = job_id
    return result
