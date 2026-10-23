
# ============================================================
# PHYSICAL AI & AUTONOMOUS ROBOTICS INTELLIGENCE PLATFORM
# Main FastAPI Dockerfile
# ============================================================

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Physical AI Robotics"
LABEL org.opencontainers.image.description="AI Robotics Intelligence Platform"
LABEL org.opencontainers.image.version="1.0.0"

# Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    ROBOT_MODE=simulation \
    SIMULATION_ONLY=true \
    PHYSICAL_ROBOT_CONTROL_ENABLED=false \
    HARDWARE_ACTUATION_ENABLED=false \
    ACTUATOR_COMMANDS_ENABLED=false \
    REMOTE_MOTION_COMMANDS_ENABLED=false \
    AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED=false \
    EMERGENCY_STOP_RESET_ENABLED=false

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    curl \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project code
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY simulation/ ./simulation/
COPY robot_configs/ ./robot_configs/
COPY scripts/ ./scripts/

# Create runtime directories
RUN mkdir -p \
    /app/data \
    /app/storage \
    /app/reports \
    /app/logs \
    /app/ml/artifacts

# Non-root application user
RUN groupadd --system appgroup \
    && useradd --system \
       --gid appgroup \
       --home-dir /app \
       appuser \
    && chown -R appuser:appgroup /app

USER appuser

# API port
EXPOSE 8000

# Container health check
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" \
    || exit 1

# Start FastAPI
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
