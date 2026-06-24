"""
Telegram Bot — واجهة الفريق لرفع الفصول واستلام النتائج.
يستخدم aiogram 3.x — يشتغل على Bot Token مباشرة.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
import httpx

from app.config import TELEGRAM_BOT_TOKEN, DOWNLOAD_DIR, TEMP_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

API_BASE = os.getenv("MANGA_API_URL", "http://localhost:8000")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

user_jobs: dict[int, dict] = {}

POLL_INTERVAL = 4
MAX_WAIT = 600


async def api_post_file(endpoint: str, field: str, file_path: str, filename: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        with open(file_path, "rb") as f:
            files = {field: (filename, f, "application/octet-stream")}
            resp = await client.post(f"{API_BASE}{endpoint}", files=files)
            resp.raise_for_status()
            return resp.json()


async def api_get(endpoint: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{API_BASE}{endpoint}")
        resp.raise_for_status()
        return resp.json()


async def wait_for_job(task_id: str, progress_msg: Message) -> dict:
    elapsed = 0
    last_edit = 0
    while elapsed < MAX_WAIT:
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        try:
            status = await api_get(f"/api/status/{task_id}")
        except Exception as e:
            logger.warning("Poll error: %s", e)
            continue

        if elapsed - last_edit >= 15:
            last_edit = elapsed
            mins, secs = divmod(int(elapsed), 60)
            try:
                await progress_msg.edit_text(
                    f"🔄 جاري التبييض… ({mins:02d}:{secs:02d} مضت)\n"
                    f"الحالة: {status.get('status', 'processing')}"
                )
            except Exception:
                pass

        state = status.get("status")
        if state in ("done", "failed"):
            return status

    return {"status": "timeout"}


@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer(
        "🎌 *أهلاً بك في Manga Cleaner Bot*\n\n"
        "أنا هساعدك تبيض صفحات المانهوا بالذكاء الاصطناعي\\!\n\n"
        "*طريقة الاستخدام:*\n"
        "• ابعت ملف *ZIP* فيه صفحات الفصل ← هبيضهم كلهم\n"
        "• ابعت *صورة واحدة* \\(PNG/JPG\\) ← هبيضها فوراً\n\n"
        "⚡ الصفحة الواحدة: أجزاء من الثانية على GPU\n"
        "📦 الحد الأقصى: 200 ميجا\n\n"
        "/help \\- المساعدة   /status \\- حالة المهمة",
        parse_mode="MarkdownV2"
    )


@dp.message(Command("help"))
async def help_handler(message: Message):
    await message.answer(
        "📖 *دليل الاستخدام:*\n\n"
        "1️⃣ *فصل كامل \\(ZIP\\):*\n"
        "   ارفع الـ ZIP كـ File\n"
        "   النظام هيبيض كل الصور ويرجعلك ZIP جاهز\n\n"
        "2️⃣ *صورة واحدة:*\n"
        "   ابعت الصورة كـ File \\(مش كصورة\\) للحفاظ على الجودة\n\n"
        "⚡ *الأداء على GPU RTX 4080:*\n"
        "   صفحة واحدة: ~0\\.5 ثانية\n"
        "   فصل 50 صفحة: دقيقة \\- دقيقتين",
        parse_mode="MarkdownV2"
    )


@dp.message(Command("status"))
async def status_handler(message: Message):
    user_id = message.from_user.id
    job = user_jobs.get(user_id)
    if not job:
        await message.answer("⚠️ مفيش مهمة نشطة. ابعت ملف أو صورة لتبدأ.")
        return
    try:
        status = await api_get(f"/api/status/{job['task_id']}")
        state = status.get("status", "unknown")
        text = f"📊 حالة المهمة:\nالحالة: {state}"
        if state == "done":
            text += f"\n✅ {status.get('succeeded', 0)}/{status.get('total', 0)} صفحة"
        await message.answer(text)
    except Exception as e:
        await message.answer(f"❌ خطأ: {e}")


@dp.message(F.document)
async def document_handler(message: Message):
    doc = message.document
    filename = doc.file_name or f"file_{doc.file_id}"
    ext = Path(filename).suffix.lower()
    user_id = message.from_user.id

    supported = {".zip", ".jpg", ".jpeg", ".png", ".webp"}
    if ext not in supported:
        await message.answer(f"⚠️ الامتداد '{ext}' مش مدعوم. ارفع ZIP أو PNG/JPG.")
        return

    # Download the file
    dl_path = os.path.join(TEMP_DIR, f"{user_id}{ext}")
    progress = await message.answer(f"⬇️ جاري تحميل {filename}…")

    file_info = await bot.get_file(doc.file_id)
    await bot.download_file(file_info.file_path, destination=dl_path)

    if ext == ".zip":
        await handle_zip(message, progress, dl_path, filename, user_id)
    else:
        await handle_image(message, progress, dl_path, filename)


async def handle_zip(message: Message, progress: Message, dl_path: str, filename: str, user_id: int):
    await progress.edit_text("⏳ بيترفع الفصل للسيرفر…")
    try:
        result = await api_post_file("/api/clean/chapter", "file", dl_path, filename)
        task_id = result["task_id"]
        job_id = result["job_id"]
        user_jobs[user_id] = {"task_id": task_id, "job_id": job_id}
        await progress.edit_text("🔄 الفصل في طابور المعالجة… بيتبيض صفحة ورا صفحة 🎨")
    except Exception as e:
        await progress.edit_text(f"❌ فشل الرفع: {e}")
        return

    final = await wait_for_job(task_id, progress)

    if final["status"] == "done":
        job_id = user_jobs[user_id]["job_id"]
        output_dir = Path(DOWNLOAD_DIR) / job_id
        zip_files = list(output_dir.glob("*.zip")) if output_dir.exists() else []
        if not zip_files:
            zip_files = list(Path(DOWNLOAD_DIR).rglob(f"*{job_id}*cleaned*.zip"))

        if zip_files:
            await progress.edit_text(
                f"✅ تم التبييض!\n"
                f"📄 {final.get('succeeded', 0)}/{final.get('total', 0)} صفحة\n"
                "⬇️ جاري إرسال الملف…"
            )
            doc_file = FSInputFile(str(zip_files[0]), filename=f"cleaned_{filename}")
            await message.answer_document(
                doc_file,
                caption=f"🎌 الفصل المبيض جاهز!\n✅ {final.get('succeeded', 0)} صفحة"
            )
            await progress.delete()
        else:
            await progress.edit_text("⚠️ تمت المعالجة لكن ملف النتيجة مش لقيه، تواصل مع المشرف.")
    elif final["status"] == "failed":
        await progress.edit_text(f"❌ المعالجة فشلت: {final.get('error', 'خطأ غير معروف')}")
    else:
        await progress.edit_text("⏰ انتهى وقت الانتظار. استخدم /status لمتابعة المهمة.")


async def handle_image(message: Message, progress: Message, dl_path: str, filename: str):
    await progress.edit_text("🔄 جاري تبييض الصورة…")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            with open(dl_path, "rb") as f:
                resp = await client.post(
                    f"{API_BASE}/api/clean/sync",
                    files={"file": (filename, f, "image/png")},
                )
        if resp.status_code == 200:
            out_path = dl_path.replace(Path(dl_path).suffix, "_cleaned.jpg")
            with open(out_path, "wb") as f:
                f.write(resp.content)
            doc_file = FSInputFile(out_path, filename="cleaned.jpg")
            await message.answer_document(doc_file, caption="✅ الصورة اتبيضت!")
            await progress.delete()
        else:
            try:
                detail = resp.json().get("detail", resp.text[:200])
            except Exception:
                detail = resp.text[:200] or f"HTTP {resp.status_code}"
            await progress.edit_text(f"❌ فشل التبييض: {detail}")
    except Exception as e:
        await progress.edit_text(f"❌ خطأ: {e}")


async def main():
    logger.info("🚀 Manga Cleaner Bot (aiogram) is running…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
