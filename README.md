# 🎌 Manga Cleaner — نظام تبييض المانهوا بالذكاء الاصطناعي

## المكونات

| المكوّن | الوظيفة |
|---------|---------|
| **FastAPI** | API الرئيسي يستقبل الصور |
| **comic-text-detector** | يكتشف فقاعات الكلام ويعمل Mask |
| **IOPaint (LaMa)** | يمسح الكلام ويعيد رسم الخلفية |
| **Celery + Redis** | طابور المهام للفصول الكبيرة |
| **Telegram Bot** | واجهة الفريق (Telethon) |

## متطلبات التثبيت

### 1. Python packages
```bash
pip install -r requirements.txt
pip install iopaint  # أو: pip install lama-cleaner
```

### 2. Redis
```bash
# Linux/Mac
sudo apt install redis-server
redis-server

# أو باستخدام Docker
docker run -d -p 6379:6379 redis:alpine
```

### 3. متغيرات البيئة
```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
REDIS_URL=redis://localhost:6379/0
IOPAINT_URL=http://localhost:8080
```

## تشغيل النظام

افتح **4 terminals** وشغّل كل واحد:

### Terminal 1 — IOPaint (LaMa)
```bash
bash start_iopaint.sh
```

### Terminal 2 — FastAPI
```bash
bash start_api.sh
```

### Terminal 3 — Celery Worker
```bash
bash start_worker.sh
```

### Terminal 4 — Telegram Bot
```bash
bash start_bot.sh
```

## استخدام الـ API مباشرة

### تبييض فصل كامل (ZIP)
```bash
curl -X POST http://localhost:8000/api/clean/chapter \
  -F "file=@chapter01.zip" \
  | jq .
# → {"job_id": "...", "task_id": "...", "status": "queued"}
```

### تتبع الحالة
```bash
curl http://localhost:8000/api/status/<task_id>
# → {"status": "done", "succeeded": 45, "total": 47}
```

### تحميل النتيجة
```bash
curl -o cleaned.zip http://localhost:8000/api/download/<job_id>
```

### تبييض صورة واحدة (فوري)
```bash
curl -X POST http://localhost:8000/api/clean/sync \
  -F "file=@page_001.png" \
  -o cleaned_page.png
```

## استخدام البوت على تليجرام

1. افتح البوت وابعت `/start`
2. ارفع ملف **ZIP** فيه صفحات الفصل
3. انتظر رسالة التأكيد
4. استلم الـ ZIP المبيض جاهز للـ Typesetting

## الأداء المتوقع

| الجهاز | زمن الصفحة الواحدة | فصل 50 صفحة |
|--------|---------------------|-------------|
| RTX 4080 (GPU) | ~0.3-0.8 ثانية | ~1-2 دقيقة |
| CPU فقط | ~5-15 ثانية | ~8-12 دقيقة |

## هيكل الملفات

```
manga-cleaner/
├── app/
│   ├── config.py          # إعدادات النظام
│   └── main.py            # FastAPI endpoints
├── models/
│   ├── detector.py        # comic-text-detector (ONNX)
│   └── inpainter.py       # IOPaint client
├── tasks/
│   └── worker.py          # Celery tasks
├── utils/
│   └── pipeline.py        # البايبلاين الرئيسي
├── bot/
│   └── telegram_bot.py    # Telethon bot
├── uploads/               # الصور المرفوعة
├── downloads/             # الصور المبيضة
├── temp/                  # ملفات مؤقتة
└── models/                # أوزان الموديلات (تحمّل تلقائياً)
```
