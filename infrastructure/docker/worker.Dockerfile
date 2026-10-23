
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SIMULATION_ONLY=true \
    HARDWARE_ACTUATION_ENABLED=false

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY ml/ ./ml/

CMD ["celery", "-A", "backend.workers.celery_app", "worker", "--loglevel=info"]
