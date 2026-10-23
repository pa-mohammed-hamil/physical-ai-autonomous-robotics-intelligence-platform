
#!/usr/bin/env bash
# ============================================================
# Physical AI & Autonomous Robotics Intelligence Platform
# File: scripts/setup_dev.sh
#
# Development environment setup.
#
# Features:
#   - Linux / macOS / WSL2 support
#   - Python virtual environment
#   - Python dependency installation
#   - Backend / ML / RAG / agent dependency groups
#   - Optional Node.js frontend dependencies
#   - Optional Docker PostgreSQL and Redis
#   - Optional database migrations
#   - Optional demo data seeding
#   - Optional pytest execution
#   - Optional pre-commit hooks
#   - ROS 2 and GPU capability detection
#   - Safe simulation-only environment
#   - Idempotent project directory setup
#   - Setup logging
#
# Usage:
#   bash scripts/setup_dev.sh
#   bash scripts/setup_dev.sh --all
#   bash scripts/setup_dev.sh --with-ml --with-rag
#   bash scripts/setup_dev.sh --with-docker
#   bash scripts/setup_dev.sh --with-frontend
#   bash scripts/setup_dev.sh --seed
#   bash scripts/setup_dev.sh --test
#   bash scripts/setup_dev.sh --skip-install
#   bash scripts/setup_dev.sh --help
#
# Safety:
#   - Does not control physical robots.
#   - Does not start robot drivers.
#   - Does not reset emergency-stop devices.
#   - Does not enable hardware actuation.
# ============================================================

set -Eeuo pipefail

IFS=$'\n\t'

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" \
        && pwd
)"

PROJECT_ROOT="$(
    cd -- "${SCRIPT_DIR}/.." \
        && pwd
)"

VENV_DIR="${PROJECT_ROOT}/.venv"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_EXAMPLE="${PROJECT_ROOT}/.env.example"

LOG_DIR="${PROJECT_ROOT}/logs"
RUN_DIR="${PROJECT_ROOT}/.run"

SETUP_LOG="${LOG_DIR}/setup_dev.log"

# ============================================================
# Options
# ============================================================

WITH_ML=false
WITH_RAG=false
WITH_AGENTS=false
WITH_FRONTEND=false
WITH_DOCKER=false
WITH_TESTS=false
WITH_SEED=false
WITH_PRECOMMIT=false
WITH_MIGRATIONS=false

SKIP_INSTALL=false
FORCE_VENV=false
QUIET=false

PYTHON_BIN=""
PIP_BIN=""

# ============================================================
# Terminal Colors
# ============================================================

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'
    CYAN=$'\033[0;36m'
    RESET=$'\033[0m'
else
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    CYAN=""
    RESET=""
fi

# ============================================================
# Logging
# ============================================================

mkdir -p "${LOG_DIR}"

log() {
    local level="$1"
    shift

    local message="$*"
    local timestamp

    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

    printf '[%s] [%s] %s\n' \
        "${timestamp}" \
        "${level}" \
        "${message}" >> "${SETUP_LOG}"

    if [[ "${QUIET}" != true || "${level}" == "ERROR" ]]; then
        case "${level}" in
            INFO)
                printf '%s[INFO]%s %s\n' \
                    "${BLUE}" "${RESET}" "${message}"
                ;;
            OK)
                printf '%s[ OK ]%s %s\n' \
                    "${GREEN}" "${RESET}" "${message}"
                ;;
            WARN)
                printf '%s[WARN]%s %s\n' \
                    "${YELLOW}" "${RESET}" "${message}"
                ;;
            ERROR)
                printf '%s[ERROR]%s %s\n' \
                    "${RED}" "${RESET}" "${message}" >&2
                ;;
            *)
                printf '[%s] %s\n' "${level}" "${message}"
                ;;
        esac
    fi
}

die() {
    log ERROR "$*"
    exit 1
}

on_error() {
    local exit_code="$1"
    local line_number="$2"

    log ERROR \
        "Setup failed at line ${line_number} (exit ${exit_code})"

    exit "${exit_code}"
}

trap 'on_error "$?" "$LINENO"' ERR

# ============================================================
# Help
# ============================================================

show_help() {
    cat <<'EOF'
Physical AI Development Environment Setup

Usage:
    bash scripts/setup_dev.sh [OPTIONS]

Options:
    --all                Install optional development groups
                         and configure local services
    --with-ml            Install ML/DL dependencies
    --with-rag           Install RAG dependencies
    --with-agents        Install agent dependencies
    --with-frontend      Install frontend dependencies
    --with-docker        Start PostgreSQL and Redis containers
    --migrate            Run Alembic migrations
    --seed               Seed synthetic demonstration data
    --test               Run pytest
    --precommit          Install pre-commit hooks
    --skip-install       Skip Python dependency installation
    --force-venv         Recreate Python virtual environment
    --quiet              Reduce console output
    --help               Display this help

Examples:
    bash scripts/setup_dev.sh

    bash scripts/setup_dev.sh --with-ml --with-rag

    bash scripts/setup_dev.sh --with-docker --migrate

    bash scripts/setup_dev.sh --seed --test

    bash scripts/setup_dev.sh --all

Default:
    Creates .venv and installs the project's base
    requirements when available.

Safety:
    All setup operations remain simulation-only.
EOF
}

# ============================================================
# Argument Parsing
# ============================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)
                WITH_ML=true
                WITH_RAG=true
                WITH_AGENTS=true
                WITH_FRONTEND=true
                WITH_DOCKER=true
                WITH_TESTS=true
                WITH_PRECOMMIT=true
                ;;
            --with-ml)
                WITH_ML=true
                ;;
            --with-rag)
                WITH_RAG=true
                ;;
            --with-agents)
                WITH_AGENTS=true
                ;;
            --with-frontend)
                WITH_FRONTEND=true
                ;;
            --with-docker)
                WITH_DOCKER=true
                ;;
            --migrate)
                WITH_MIGRATIONS=true
                ;;
            --seed)
                WITH_SEED=true
                ;;
            --test)
                WITH_TESTS=true
                ;;
            --precommit)
                WITH_PRECOMMIT=true
                ;;
            --skip-install)
                SKIP_INSTALL=true
                ;;
            --force-venv)
                FORCE_VENV=true
                ;;
            --quiet)
                QUIET=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac

        shift
    done
}

# ============================================================
# System Detection
# ============================================================

detect_os() {
    local os_name

    os_name="$(uname -s)"

    case "${os_name}" in
        Linux)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                log INFO "Detected Windows Subsystem for Linux"
            else
                log INFO "Detected Linux"
            fi
            ;;
        Darwin)
            log INFO "Detected macOS"
            ;;
        *)
            log WARN "Untested operating system: ${os_name}"
            ;;
    esac
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================
# Python Detection
# ============================================================

detect_python() {
    local candidate
    local version_ok

    for candidate in python3 python; do
        if ! check_command "${candidate}"; then
            continue
        fi

        version_ok="$(
            "${candidate}" -c '
import sys
print(int(sys.version_info >= (3, 10)))
' 2>/dev/null || printf '0'
        )"

        if [[ "${version_ok}" == "1" ]]; then
            PYTHON_BIN="$(
                command -v "${candidate}"
            )"

            log OK "Python: $("${PYTHON_BIN}" --version)"
            return 0
        fi
    done

    die "Python 3.10 or newer is required"
}

# ============================================================
# Directory Creation
# ============================================================

create_directories() {
    log INFO "Creating development directories"

    local directories=(
        "logs"
        ".run"
        "storage"
        "storage/uploads"
        "storage/cache"
        "storage/models"
        "storage/vector_store"
        "storage/checkpoints"
        "data"
        "data/raw"
        "data/processed"
        "data/demo"
        "data/telemetry"
        "data/robot_parts"
        "data/maintenance"
        "data/prices"
        "reports"
        "reports/evaluation"
        "reports/safety"
        "reports/maintenance"
        "reports/analytics"
        "artifacts"
        "artifacts/models"
        "artifacts/experiments"
        "artifacts/metrics"
    )

    local directory

    for directory in "${directories[@]}"; do
        mkdir -p "${PROJECT_ROOT}/${directory}"
    done

    log OK "Development directories ready"
}

# ============================================================
# Environment Configuration
# ============================================================

create_environment_file() {
    if [[ -f "${ENV_FILE}" ]]; then
        log INFO "Existing .env preserved"
    elif [[ -f "${ENV_EXAMPLE}" ]]; then
        cp "${ENV_EXAMPLE}" "${ENV_FILE}"
        chmod 600 "${ENV_FILE}"
        log OK ".env created from .env.example"
    else
        cat > "${ENV_FILE}" <<'EOF'
# Physical AI Development Configuration

APP_NAME=Physical AI Robotics Platform
APP_ENV=development
DEBUG=true

BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

ML_HOST=127.0.0.1
ML_PORT=8001

FRONTEND_HOST=127.0.0.1
FRONTEND_PORT=3000

DATABASE_URL=sqlite:///storage/robotics_demo.db

REDIS_URL=redis://127.0.0.1:6379/0

LOG_LEVEL=INFO

# Simulation Safety
ROBOT_MODE=simulation
SIMULATION_ONLY=true
PHYSICAL_ROBOT_CONTROL_ENABLED=false
HARDWARE_ACTUATION_ENABLED=false
ACTUATOR_COMMANDS_ENABLED=false
REMOTE_MOTION_COMMANDS_ENABLED=false
AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED=false
EMERGENCY_STOP_RESET_ENABLED=false
ROBOT_DRIVER_ENABLED=false
SAFETY_INTERLOCKS_REQUIRED=true
HUMAN_APPROVAL_REQUIRED=true

# Models
MODEL_DEVICE=auto
MODEL_CACHE_DIR=storage/models

# RAG
RAG_INDEX_DIR=storage/vector_store

# Telemetry
TELEMETRY_MODE=synthetic
EOF

        chmod 600 "${ENV_FILE}"
        log OK "Safe development .env created"
    fi

    # Do not source .env as shell code.
    # Reject explicitly unsafe settings rather than silently
    # overwriting user configuration.
    local key
    local value
    local unsafe_keys=(
        "PHYSICAL_ROBOT_CONTROL_ENABLED"
        "HARDWARE_ACTUATION_ENABLED"
        "ACTUATOR_COMMANDS_ENABLED"
        "REMOTE_MOTION_COMMANDS_ENABLED"
        "AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED"
        "EMERGENCY_STOP_RESET_ENABLED"
        "ROBOT_DRIVER_ENABLED"
    )

    for key in "${unsafe_keys[@]}"; do
        value="$(
            awk -F= -v key="${key}" '
                $1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
                    value=$2
                    sub(/[[:space:]]*#.*/, "", value)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
                    gsub(/^["\047]|["\047]$/, "", value)
                    print tolower(value)
                }
            ' "${ENV_FILE}" | tail -n 1
        )"

        if [[ "${value}" == "true" || "${value}" == "1" || "${value}" == "yes" ]]; then
            die "Unsafe .env setting: ${key}=${value}"
        fi
    done

    log OK "Simulation safety configuration checked"
}

# ============================================================
# Virtual Environment
# ============================================================

create_virtual_environment() {
    if [[ "${FORCE_VENV}" == true && -d "${VENV_DIR}" ]]; then
        log WARN "Removing existing virtual environment"
        rm -rf -- "${VENV_DIR}"
    fi

    if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
        log INFO "Creating Python virtual environment"

        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    else
        log INFO "Using existing virtual environment"
    fi

    PYTHON_BIN="${VENV_DIR}/bin/python"
    PIP_BIN="${VENV_DIR}/bin/pip"

    [[ -x "${PYTHON_BIN}" ]] || die "Virtual environment creation failed"
    [[ -x "${PIP_BIN}" ]] || die "pip is unavailable in virtual environment"

    log OK "Virtual environment ready"
}

# ============================================================
# Python Dependency Installation
# ============================================================

install_requirements_file() {
    local requirements_file="$1"

    if [[ ! -f "${requirements_file}" ]]; then
        log WARN "Optional requirements file not found: ${requirements_file}"
        return 0
    fi

    log INFO "Installing $(basename "${requirements_file}")"

    "${PIP_BIN}" install \
        -r "${requirements_file}"

    log OK "Installed $(basename "${requirements_file}")"
}

install_python_dependencies() {
    if [[ "${SKIP_INSTALL}" == true ]]; then
        log INFO "Skipping Python package installation"
        return 0
    fi

    log INFO "Updating Python packaging tools"

    "${PYTHON_BIN}" -m pip install \
        --upgrade pip setuptools wheel

    if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
        install_requirements_file \
            "${PROJECT_ROOT}/requirements.txt"
    elif [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
        log INFO "Installing project from pyproject.toml"

        "${PIP_BIN}" install -e "${PROJECT_ROOT}"
    else
        log WARN "No requirements.txt or pyproject.toml found"
    fi

    if [[ "${WITH_ML}" == true ]]; then
        install_requirements_file \
            "${PROJECT_ROOT}/requirements-ml.txt"
    fi

    if [[ "${WITH_RAG}" == true ]]; then
        install_requirements_file \
            "${PROJECT_ROOT}/requirements-rag.txt"
    fi

    if [[ "${WITH_AGENTS}" == true ]]; then
        install_requirements_file \
            "${PROJECT_ROOT}/requirements-agents.txt"
    fi

    if [[ "${WITH_TESTS}" == true ]]; then
        install_requirements_file \
            "${PROJECT_ROOT}/requirements-dev.txt"
    fi

    log OK "Python dependency installation finished"
}

# ============================================================
# Frontend Setup
# ============================================================

setup_frontend() {
    if [[ "${WITH_FRONTEND}" != true ]]; then
        return 0
    fi

    log INFO "Checking frontend environment"

    if [[ ! -d "${PROJECT_ROOT}/frontend" ]]; then
        log WARN "frontend/ directory not found"
        return 0
    fi

    if [[ ! -f "${PROJECT_ROOT}/frontend/package.json" ]]; then
        log INFO "Static HTML/CSS/JavaScript frontend detected"
        log INFO "No Node.js installation required"
        return 0
    fi

    if ! check_command node; then
        log WARN "Node.js not installed"
        return 0
    fi

    if ! check_command npm; then
        log WARN "npm not installed"
        return 0
    fi

    log INFO "Node.js: $(node --version)"
    log INFO "npm: $(npm --version)"

    (
        cd "${PROJECT_ROOT}/frontend"

        if [[ -f "package-lock.json" ]]; then
            npm ci
        else
            npm install
        fi
    )

    log OK "Frontend dependencies installed"
}

# ============================================================
# Docker Setup
# ============================================================

docker_compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    elif check_command docker-compose; then
        docker-compose "$@"
    else
        return 1
    fi
}

setup_docker() {
    if [[ "${WITH_DOCKER}" != true ]]; then
        return 0
    fi

    if ! check_command docker; then
        log WARN "Docker is not installed"
        return 0
    fi

    if ! docker info >/dev/null 2>&1; then
        log WARN "Docker daemon is not running"
        return 0
    fi

    local compose_file=""

    if [[ -f "${PROJECT_ROOT}/docker-compose.yml" ]]; then
        compose_file="${PROJECT_ROOT}/docker-compose.yml"
    elif [[ -f "${PROJECT_ROOT}/compose.yaml" ]]; then
        compose_file="${PROJECT_ROOT}/compose.yaml"
    fi

    if [[ -z "${compose_file}" ]]; then
        log WARN "Docker Compose configuration not found"
        return 0
    fi

    log INFO "Starting development PostgreSQL and Redis"

    # Explicit service selection prevents accidental launch
    # of hardware drivers or other Compose services.
    (
        cd "${PROJECT_ROOT}"
        docker_compose -f "${compose_file}" \
            up -d postgres redis
    )

    log OK "Development database services requested"
}

# ============================================================
# Database Migrations
# ============================================================

run_migrations() {
    if [[ "${WITH_MIGRATIONS}" != true ]]; then
        return 0
    fi

    if [[ ! -f "${PROJECT_ROOT}/alembic.ini" ]]; then
        log WARN "alembic.ini not found"
        return 0
    fi

    if ! "${PYTHON_BIN}" -m alembic --help >/dev/null 2>&1; then
        log WARN "Alembic is not installed"
        return 0
    fi

    log INFO "Running database migrations"

    (
        cd "${PROJECT_ROOT}"
        "${PYTHON_BIN}" -m alembic upgrade head
    )

    log OK "Database migrations completed"
}

# ============================================================
# Synthetic Database Seeding
# ============================================================

run_seed_script() {
    local script_name="$1"
    local script_path="${SCRIPT_DIR}/${script_name}"

    if [[ ! -f "${script_path}" ]]; then
        log WARN "Seed script not found: ${script_name}"
        return 0
    fi

    log INFO "Running ${script_name}"

    (
        cd "${PROJECT_ROOT}"
        "${PYTHON_BIN}" "${script_path}"
    )

    log OK "${script_name} completed"
}

seed_demo_data() {
    if [[ "${WITH_SEED}" != true ]]; then
        return 0
    fi

    log INFO "Seeding synthetic robotics demonstration data"

    # Existing robot IDs from seed_database.py and
    # seed_robots.py must use the same ID namespace.
    # Do not run both robot seeders automatically until
    # their identifiers are aligned.
    run_seed_script "seed_database.py"

    run_seed_script "seed_maintenance.py"
    run_seed_script "seed_parts.py"
    run_seed_script "seed_prices.py"

    log OK "Synthetic demo data seeding completed"
}

# ============================================================
# Pre-commit
# ============================================================

setup_precommit() {
    if [[ "${WITH_PRECOMMIT}" != true ]]; then
        return 0
    fi

    if [[ ! -f "${PROJECT_ROOT}/.pre-commit-config.yaml" ]]; then
        log WARN "Pre-commit configuration not found"
        return 0
    fi

    if ! "${PYTHON_BIN}" -m pre_commit --version >/dev/null 2>&1; then
        log WARN "pre-commit package not installed"
        return 0
    fi

    (
        cd "${PROJECT_ROOT}"
        "${PYTHON_BIN}" -m pre_commit install
    )

    log OK "Pre-commit hooks installed"
}

# ============================================================
# Tests
# ============================================================

run_tests() {
    if [[ "${WITH_TESTS}" != true ]]; then
        return 0
    fi

    if ! "${PYTHON_BIN}" -m pytest --version >/dev/null 2>&1; then
        log WARN "pytest not installed"
        return 0
    fi

    if [[ ! -d "${PROJECT_ROOT}/tests" ]]; then
        log WARN "tests/ directory not found"
        return 0
    fi

    log INFO "Running project tests"

    (
        cd "${PROJECT_ROOT}"
        "${PYTHON_BIN}" -m pytest tests/ -q
    )

    log OK "Tests completed"
}

# ============================================================
# ROS 2 Detection
# ============================================================

detect_ros2() {
    if check_command ros2; then
        log OK "ROS 2 CLI detected"

        local ros_distro="${ROS_DISTRO:-unknown}"
        log INFO "ROS distribution: ${ros_distro}"
    else
        log INFO "ROS 2 not detected (optional)"
    fi

    # No ROS launch or robot driver startup.
}

# ============================================================
# GPU Detection
# ============================================================

detect_gpu() {
    if check_command nvidia-smi; then
        log INFO "NVIDIA GPU detected"

        nvidia-smi \
            --query-gpu=name,memory.total \
            --format=csv,noheader \
            2>/dev/null || true
    else
        log INFO "NVIDIA GPU not detected"
    fi

    if "${PYTHON_BIN}" -c \
        'import torch' >/dev/null 2>&1; then

        "${PYTHON_BIN}" -c '
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
'
    else
        log INFO "PyTorch not installed"
    fi
}

# ============================================================
# Health Check
# ============================================================

run_setup_health_check() {
    log INFO "Checking development environment"

    "${PYTHON_BIN}" --version

    if "${PYTHON_BIN}" -c \
        'import fastapi' >/dev/null 2>&1; then
        log OK "FastAPI import successful"
    else
        log WARN "FastAPI unavailable"
    fi

    if "${PYTHON_BIN}" -c \
        'import sqlalchemy' >/dev/null 2>&1; then
        log OK "SQLAlchemy import successful"
    else
        log WARN "SQLAlchemy unavailable"
    fi

    if "${PYTHON_BIN}" -c \
        'import pydantic' >/dev/null 2>&1; then
        log OK "Pydantic import successful"
    else
        log WARN "Pydantic unavailable"
    fi

    if "${PYTHON_BIN}" -c \
        'import uvicorn' >/dev/null 2>&1; then
        log OK "Uvicorn import successful"
    else
        log WARN "Uvicorn unavailable"
    fi

    if [[ -f "${SCRIPT_DIR}/health_check.py" ]]; then
        log INFO "Project health checker available"
    fi
}

# ============================================================
# Setup Summary
# ============================================================

show_summary() {
    cat <<EOF

${GREEN}============================================================${RESET}
${GREEN} Physical AI Development Setup Finished${RESET}
${GREEN}============================================================${RESET}

Project:
    ${PROJECT_ROOT}

Virtual environment:
    ${VENV_DIR}

Environment:
    ${ENV_FILE}

Setup log:
    ${SETUP_LOG}

Safety:
    Simulation-only development configuration checked
    Physical actuation disabled by default
    Hardware drivers not started

Activate environment:
    source .venv/bin/activate

Run backend:
    python -m uvicorn backend.main:app --reload

Run local launcher:
    bash scripts/run_local.sh

Check health:
    python scripts/health_check.py --help

Seed synthetic data:
    python scripts/seed_database.py
    python scripts/seed_maintenance.py
    python scripts/seed_parts.py
    python scripts/seed_prices.py

${CYAN}============================================================${RESET}

EOF
}

# ============================================================
# Main
# ============================================================

main() {
    parse_args "$@"

    log INFO "Starting Physical AI development setup"
    log INFO "Project root: ${PROJECT_ROOT}"

    detect_os
    detect_python

    create_directories
    create_environment_file

    create_virtual_environment

    install_python_dependencies
    setup_frontend
    setup_docker
    run_migrations
    seed_demo_data
    setup_precommit
    run_tests

    detect_ros2
    detect_gpu

    run_setup_health_check

    log OK "Development setup completed"

    show_summary
}

main "$@"
```
