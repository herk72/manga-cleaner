#!/bin/bash
# ==========================================================
# سكريبت رفع manga-cleaner على GitHub
# شغّله مرة واحدة بعد ما تحمّل المشروع
# ==========================================================

set -e

GITHUB_USER="herk72"
REPO_NAME="manga-cleaner"
GITHUB_TOKEN="${GITHUB_PERSONAL_ACCESS_TOKEN:-}"

# ── 1. تحقق من وجود الـ token ──────────────────────────────
if [ -z "$GITHUB_TOKEN" ]; then
  echo "⚠️  مفيش GITHUB_PERSONAL_ACCESS_TOKEN في البيئة."
  echo "    شغّل الأمر ده أول:"
  echo "    export GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here"
  exit 1
fi

# ── 2. إنشاء الـ repo على GitHub ──────────────────────────
echo "📦 بيعمل repo على GitHub..."
HTTP_CODE=$(curl -s -o /tmp/gh_response.json -w "%{http_code}" \
  -X POST \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"$REPO_NAME\",\"description\":\"نظام تبييض المانهوا بالذكاء الاصطناعي — comic-text-detector + IOPaint + Telegram Bot\",\"private\":false,\"auto_init\":false}")

if [ "$HTTP_CODE" = "201" ]; then
  echo "✅ تم إنشاء الـ repo!"
elif [ "$HTTP_CODE" = "422" ]; then
  echo "ℹ️  الـ repo موجود بالفعل، هكمل..."
else
  echo "❌ فشل إنشاء الـ repo (HTTP $HTTP_CODE)"
  cat /tmp/gh_response.json
  echo ""
  echo "💡 امل يدوياً: https://github.com/new → اسم الـ repo: $REPO_NAME"
  echo "   وبعدين ارجع شغّل السكريبت تاني"
  exit 1
fi

# ── 3. تهيئة git ───────────────────────────────────────────
echo "🔧 بيهيئ git..."
git init
git config user.email "$GITHUB_USER@users.noreply.github.com"
git config user.name "$GITHUB_USER"

# ── 4. إضافة الملفات والـ commit ──────────────────────────
echo "📝 بيعمل commit..."
git add -A
git commit -m "feat: initial manga cleaner system

Components:
- FastAPI backend (sync + async endpoints)
- comic-text-detector ONNX model for speech bubble detection
- IOPaint/LaMa client for AI inpainting (with OpenCV fallback)
- Celery + Redis task queue for batch chapter processing
- Telegram Bot (aiogram) for team workflow

Pipeline:
ZIP upload → detect text regions → create mask → AI inpaint → ZIP download"

# ── 5. رفع على GitHub ──────────────────────────────────────
echo "🚀 بيرفع على GitHub..."
git remote remove origin 2>/dev/null || true
git remote add origin "https://$GITHUB_USER:$GITHUB_TOKEN@github.com/$GITHUB_USER/$REPO_NAME.git"
git branch -M main
git push -u origin main

echo ""
echo "🎉 تم الرفع بنجاح!"
echo "🔗 الرابط: https://github.com/$GITHUB_USER/$REPO_NAME"
