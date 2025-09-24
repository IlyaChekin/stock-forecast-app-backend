# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

# Системные утилиты (минимум)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Установим Poetry
RUN pip install "poetry==2.0.1"

# Рабочая папка
WORKDIR /app

# Сначала только зависимости (слой кэшируется)
COPY pyproject.toml poetry.lock* ./ 
RUN poetry install --no-interaction --no-ansi --no-root --only main

# Затем код
COPY app ./app
COPY scripts ./scripts

# Папки для данных/моделей (будем монтировать тома)
RUN mkdir -p /app/data /app/dataset/train /app/dataset/test /app/models/lstm/latest

# Непривилегированный пользователь
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
