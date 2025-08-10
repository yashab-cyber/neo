FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .

COPY src ./src

EXPOSE 8000
ENV NEO_ENVIRONMENT=prod
CMD ["uvicorn", "neo.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
