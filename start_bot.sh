#!/bin/bash
# تشغيل Telegram Bot
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
python3 bot/telegram_bot.py
