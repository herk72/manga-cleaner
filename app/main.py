"""
FastAPI — المدخل الرئيسي للـ API.
"""

import os
import sys
import uuid
import shutil
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import aiofiles

from app.config import UPLOAD_DIR, DOWNLOAD_DIR, TEMP_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Manga Cleaner API",
    description="تبييض صفحات المانهوا باستخدام AI",
    version="1.0.0",
)


def get_celery():
    from tasks.worker import celery_app
    return celery_app


@app.get("/health")
async def health():
    return {"status": "ok", "service": "manga-cleaner"}


@app.post("/api/clean/chapter")
async def clean_chapter(file: UploadFile = File(...)):
    """
    استقبل ملف ZIP فيه صفحات الفصل وابعته للمعالجة.
    يرجع job_id للتتبع.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="فقط ملفات ZIP مقبولة")

    job_id = str(uuid.uuid4())
    zip_path = os.path.join(UPLOAD_DIR, f"{job_id}.zip")

    async with aiofiles.open(zip_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    from tasks.worker import clean_chapter_task
    task = clean_chapter_task.delay(job_id, zip_path)

    return {"job_id": job_id, "task_id": task.id, "status": "queued"}


@app.post("/api/clean/images")
async def clean_images(files: list[UploadFile] = File(...)):
    """
    استقبل صور منفردة (PNG/JPG) وابعتهم للمعالجة.
    """
    job_id = str(uuid.uuid4())
    saved_paths = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        save_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        async with aiofiles.open(save_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        saved_paths.append(save_path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="لازم ترفع صور بامتداد JPG أو PNG")

    from tasks.worker import clean_images_task
    task = clean_images_task.delay(job_id, saved_paths)

    return {"job_id": job_id, "task_id": task.id, "status": "queued"}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """تتبع حالة المهمة."""
    from celery.result import AsyncResult
    celery_app = get_celery()
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return {"task_id": task_id, "status": "waiting"}
    elif result.state == "PROGRESS":
        return {"task_id": task_id, "status": "processing", "info": result.info}
    elif result.state == "SUCCESS":
        data = result.result or {}
        return {
            "task_id": task_id,
            "status": "done",
            "job_id": data.get("job_id"),
            "total": data.get("total"),
            "succeeded": data.get("succeeded"),
            "failed": data.get("failed"),
        }
    elif result.state == "FAILURE":
        return {"task_id": task_id, "status": "failed", "error": str(result.info)}
    else:
        return {"task_id": task_id, "status": result.state}


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """تحميل ملف ZIP بالصور المبيضة."""
    output_dir = os.path.join(DOWNLOAD_DIR, job_id)

    zip_candidates = list(Path(output_dir).glob("*.zip")) if os.path.exists(output_dir) else []
    if not zip_candidates:
        zip_candidates = list(Path(DOWNLOAD_DIR).glob(f"{job_id}*.zip"))

    if not zip_candidates:
        raise HTTPException(status_code=404, detail="النتيجة مش موجودة أو لسه بتتعالج")

    output_zip = str(zip_candidates[0])
    return FileResponse(
        output_zip,
        media_type="application/zip",
        filename=f"cleaned_{job_id}.zip",
    )


@app.post("/api/clean/sync")
async def clean_sync(file: UploadFile = File(...)):
    """
    معالجة فورية (Synchronous) لصورة واحدة — بدون queue.
    مناسب للاختبار.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="صيغة الملف مش مدعومة")

    job_id = str(uuid.uuid4())
    in_path = os.path.join(TEMP_DIR, f"{job_id}_in{ext}")
    out_path = os.path.join(TEMP_DIR, f"{job_id}_out.png")

    async with aiofiles.open(in_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    from utils.pipeline import process_single_page
    result = process_single_page(in_path, out_path)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return FileResponse(out_path, media_type="image/png", filename="cleaned.png")
