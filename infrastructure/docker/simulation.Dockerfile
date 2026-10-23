
# ============================================================
# PHYSICAL AI & AUTONOMOUS ROBOTICS INTELLIGENCE PLATFORM
# SIMULATION SERVICE DOCKERFILE
# Python 3.11 | MuJoCo | PyBullet | Gymnasium
# ============================================================

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="Physical AI Simulation"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="Simulation-only robotics environment"

# ------------------------------------------------------------
# PYTHON & SIMULATION CONFIGURATION
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=development \
    PYTHONPATH=/app \
    ROBOT_MODE=simulation \
    SIMULATION_ENABLED=true \
    SIMULATION_ONLY=true \
    SIMULATION_ENGINE=mujoco \
    SIMULATION_TIMESTEP=0.002 \
    SIMULATION_SEED=42 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    OMP_NUM_THREADS=2 \
    PHYSICAL_ROBOT_CONTROL_ENABLED=false \
    HARDWARE_ACTUATION_ENABLED=false \
    ROBOT_DRIVER_ENABLED=false \
    ACTUATOR_COMMANDS_ENABLED=false \
    REMOTE_MOTION_COMMANDS_ENABLED=false \
    AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED=false \
    EMERGENCY_STOP_RESET_ENABLED=false \
    SAFETY_INTERLOCKS_REQUIRED=true

WORKDIR /app

# ------------------------------------------------------------
# SYSTEM LIBRARIES FOR HEADLESS ROBOTICS SIMULATION
# ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libegl1 \
        libgl1 \
        libgles2 \
        libglib2.0-0 \
        libglfw3 \
        libosmesa6 \
        libgomp1 \
        libstdc++6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        libjpeg62-turbo \
        libpng16-16 \
        libopenblas0 \
        liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# INSTALL SIMULATION DEPENDENCIES
# Uses a separate requirements file to avoid installing the
# entire backend and ML training dependency stack.
# ------------------------------------------------------------
COPY simulation/requirements.txt /app/simulation/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        -r /app/simulation/requirements.txt

# ------------------------------------------------------------
# COPY SIMULATION SOURCE AND CONFIGURATION
# ------------------------------------------------------------
COPY simulation/ /app/simulation/

# ------------------------------------------------------------
# CREATE RUNTIME DIRECTORIES
# ------------------------------------------------------------
RUN mkdir -p \
    /app/data/simulation \
    /app/data/telemetry \
    /app/reports/simulation \
    /app/logs \
    /app/storage/simulation

# ------------------------------------------------------------
# CREATE NON-ROOT USER
# ------------------------------------------------------------
RUN groupadd --system simgroup && \
    useradd --system \
        --gid simgroup \
        --home-dir /app \
        simuser && \
    chown -R simuser:simgroup /app

USER simuser

# ------------------------------------------------------------
# SIMULATION PROCESS
# ------------------------------------------------------------
STOPSIGNAL SIGTERM

# Launch the simulation manager.
# Requires simulation/main.py with a runnable main entry point.
CMD ["python", "-m", "simulation.main"]
