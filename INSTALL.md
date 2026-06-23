# 🚀 دليل التثبيت على جهازك الخاص

## المتطلبات

| الأداة | الإصدار |
|--------|---------|
| Python | 3.10 أو أحدث |
| Redis | 7.x |
| CUDA | 11.8+ (لو عندك GPU) |
| Git | أي إصدار |

---

## الخطوة 1 — تحميل المشروع

```bash
git clone https://github.com/herk72/manga-cleaner.git
cd manga-cleaner
```

---

## الخطوة 2 — تثبيت المكتبات

```bash
pip install -r requirements.txt
pip install iopaint          # نظام الـ Inpainting
```

---

## الخطوة 3 — إعداد متغيرات البيئة

انسخ الملف وعدّله:
```bash
cp .env.example .env
```

افتح `.env` وحط فيه:
```env
TELEGRAM_BOT_TOKEN=8748117053:AAGteTFNxo0QGBGRJoUEG6JNm97j3PTpfR8
TELEGRAM_API_ID=25243065
TELEGRAM_API_HASH=2a8217328eb97dcabb733ebf24ce6de6
REDIS_URL=redis://localhost:6379/0
IOPAINT_URL=http://localhost:8080
MANGA_API_URL=http://localhost:8000
```

---

## الخطوة 4 — تشغيل النظام

افتح **4 terminals** وشغّل كل واحد:

### Terminal 1 — Redis
```bash
redis-server
```

### Terminal 2 — IOPaint (LaMa)
```bash
# على GPU (RTX 4080) ← الأسرع
iopaint start --model=lama --device=cuda --port=8080 --host=127.0.0.1

# على CPU فقط
iopaint start --model=lama --device=cpu --port=8080 --host=127.0.0.1
```

### Terminal 3 — FastAPI
```bash
cd manga-cleaner
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 4 — Celery Worker
```bash
cd manga-cleaner
PYTHONPATH=. celery -A tasks.worker.celery_app worker --loglevel=info --concurrency=2
```

### Terminal 5 — Telegram Bot
```bash
cd manga-cleaner
PYTHONPATH=. python3 bot/telegram_bot.py
```

---

## الخطوة 5 — اختبار النظام

### اختبار الـ API
```bash
# Health check
curl http://localhost:8000/health

# تبييض صورة واحدة فوراً
curl -X POST http://localhost:8000/api/clean/sync \
  -F "file=@test_page.png" \
  -o cleaned.png
```

### اختبار البوت
1. افتح التليجرام
2. افتح البوت وابعت `/start`
3. ارفع ملف ZIP أو صورة PNG/JPG

---

## الأداء المتوقع على RTX 4080

| العملية | الوقت |
|---------|-------|
| صفحة واحدة | ~0.3 – 0.8 ثانية |
| فصل 50 صفحة | ~1 – 2 دقيقة |
| فصل 100 صفحة | ~2 – 4 دقائق |

---

## استكشاف الأخطاء

### Redis مش شغال
```bash
sudo systemctl start redis
# أو
redis-server --daemonize yes
```

### IOPaint بطيء على الـ CPU
دي طبيعي — استخدم `--device=cuda` لو عندك GPU

### البوت مبيرد
- تأكد إن الـ FastAPI شغال على port 8000
- تأكد إن `MANGA_API_URL=http://localhost:8000` في `.env`

---

## هيكل المشروع

```
manga-cleaner/
├── app/
│   ├── config.py          # إعدادات النظام
│   └── main.py            # FastAPI endpoints
├── models/
│   ├── detector.py        # comic-text-detector (ONNX)
│   └── inpainter.py       # IOPaint client + OpenCV fallback
├── tasks/
│   └── worker.py          # Celery tasks
├── utils/
│   └── pipeline.py        # البايبلاين الرئيسي
├── bot/
│   └── telegram_bot.py    # Telegram Bot (aiogram)
├── .env.example           # مثال على متغيرات البيئة
├── requirements.txt       # Python dependencies
└── INSTALL.md             # هذا الملف
```
