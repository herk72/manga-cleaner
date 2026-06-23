@echo off
chcp 65001 > nul
echo.
echo  ==========================================
echo   Manga Cleaner - Windows Setup
echo   Quadro P1000 / CUDA Edition
echo  ==========================================
echo.

:: ── 1. إنشاء ملف .env ─────────────────────────────────────
if exist .env (
    echo [OK] ملف .env موجود.
) else (
    copy .env.example .env > nul
    echo [تنبيه] تم إنشاء ملف .env — افتحه وحط الـ tokens
    echo.
    notepad .env
    echo بعد ما تحفظ .env اضغط أي زر للمتابعة...
    pause > nul
)

:: ── 2. تحقق من Docker ─────────────────────────────────────
echo [*] بيتحقق من Docker...
docker info > nul 2>&1
if errorlevel 1 (
    echo.
    echo [!] Docker Desktop مش شغال!
    echo     1. افتح Docker Desktop
    echo     2. انتظر ما تظهر رسالة "Docker Desktop is running"
    echo     3. اضغط أي زر هنا للمتابعة
    echo.
    pause > nul
    docker info > nul 2>&1
    if errorlevel 1 (
        echo [X] Docker لسه مش شغال. أعد تشغيل السكريبت.
        pause
        exit /b 1
    )
)
echo [OK] Docker شغال!

:: ── 3. تحقق من NVIDIA GPU ─────────────────────────────────
echo [*] بيتحقق من Quadro P1000...
nvidia-smi > nul 2>&1
if errorlevel 1 (
    echo [!] nvidia-smi مش شغال — هيشتغل على CPU بدل GPU
) else (
    echo [OK] Quadro P1000 متعرف عليه!
)

:: ── 4. تشغيل النظام ───────────────────────────────────────
echo.
echo [*] بيبني وبيشغّل كل الـ services...
docker-compose up -d --build

if errorlevel 1 (
    echo.
    echo [X] في مشكلة — شوف الـ logs:
    echo     docker-compose logs
    pause
    exit /b 1
)

echo.
echo  ==========================================
echo   النظام شغال بنجاح!
echo  ==========================================
echo   API Docs : http://localhost:8000/docs
echo   Health   : http://localhost:8000/health
echo  ==========================================
echo.
echo  الأوامر المفيدة:
echo    docker-compose logs -f        -- شوف الـ logs
echo    docker-compose down           -- وقّف النظام
echo    docker-compose restart bot    -- أعد تشغيل البوت
echo.
pause
