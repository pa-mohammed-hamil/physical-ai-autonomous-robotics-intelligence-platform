
# ============================================================
# PHYSICAL AI & AUTONOMOUS ROBOTICS INTELLIGENCE PLATFORM
# Backend Dockerfile
# Python 3.11 | FastAPI | AI/ML | Robotics Simulation
# ============================================================

FROM python:3.11-slim

# ------------------------------------------------------------
# METADATA
# ------------------------------------------------------------
LABEL org.opencontainers.image.title="Physical AI Backend"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="FastAPI backend for AI robotics intelligence"

# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=development \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    ROBOT_MODE=simulation \
    SIMULATION_ONLY=true \
    PHYSICAL_ROBOT_CONTROL_ENABLED=false \
    HARDWARE_ACTUATION_ENABLED=false \
    ROBOT_DRIVER_ENABLED=false \
    ACTUATOR_COMMANDS_ENABLED=false \
    REMOTE_MOTION_COMMANDS_ENABLED=false \
    AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED=false \
    EMERGENCY_STOP_RESET_ENABLED=false

WORKDIR /app

# ------------------------------------------------------------
# SYSTEM DEPENDENCIES
# ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gcc \
        g++ \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# PYTHON DEPENDENCIES
# Build context must be the project root.
# ------------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# ------------------------------------------------------------
# APPLICATION SOURCE
# ------------------------------------------------------------
COPY backend/ /app/backend/
COPY ml/ /app/ml/
COPY simulation/ /app/simulation/
COPY robot_configs/ /app/robot_configs/
COPY scripts/ /app/scripts/

# ------------------------------------------------------------
# RUNTIME DIRECTORIES
# ------------------------------------------------------------
RUN mkdir -p \
    /app/data \
    /app/storage \
    /app/reports \
    /app/logs \
    /app/ml/artifacts

# ------------------------------------------------------------
# NON-ROOT USER
# ------------------------------------------------------------
RUN groupadd --system appgroup && \
    useradd --system \
        --gid appgroup \
        --home-dir /app \
        appuser && \
    chown -R appuser:appgroup /app

USER appuser

# ------------------------------------------------------------
# NETWORK
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# HEALTH CHECK
# Requires GET /health in backend/main.py
# ------------------------------------------------------------
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=60s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" \
    || exit 1

# ------------------------------------------------------------
# START FASTAPI
# ------------------------------------------------------------
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
