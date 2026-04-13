#!/usr/bin/env python3
"""
Скрипт для запуска RAG-сервиса.
Используется: python serve.py
"""

import os

import uvicorn
from dotenv import load_dotenv

# Загружаем переменные из .env файла (если есть)
load_dotenv()

def main():
    # Настройки сервера из окружения или значения по умолчанию
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "False").lower() == "true"
    app_import = "api.main:app"

    uvicorn.run(
        app_import,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
