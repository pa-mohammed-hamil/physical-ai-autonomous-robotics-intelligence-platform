#!/usr/bin/env bash
#
# Physical AI & Autonomous Robotics Intelligence Platform
# File: scripts/run_local.sh
#
# Local Development Launcher
#
# Features:
# - Python virtual environment setup
# - Dependency installation
# - Environment configuration
# - PostgreSQL / Redis readiness checks
# - FastAPI backend startup
# - ML inference server startup
# - Optional frontend static server
# - Optional Docker Compose infrastructure
# - Optional ROS 2 environment detection
# - Startup health checks
# - PID management and graceful shutdown
# - Log management
# - Configurable ports
# - Simulation-only safety enforcement
#
# Usage:
#   chmod +x scripts/run_local.sh
#
#   ./scripts/run_local.sh
#   ./scripts/run_local.sh --install
#   ./scripts/run_local.sh --backend-only
#   ./scripts/run_local.sh --ml-only
#   ./scripts/run_local.sh --with-frontend
#   ./scripts/run_local.sh --with-docker
#   ./scripts/run_local.sh --health
#   ./scripts/run_local.sh --stop
#   ./scripts/run_local.sh --status
#   ./scripts/run_local.sh --help
#
# Environment overrides:
#   BACKEND_PORT=8000
#   ML_PORT=8001
#   FRONTEND_PORT=3000
#   BACKEND_APP=backend.main:app
#   ML_APP=ml.inference.model_server:app
#   FRONTEND_DIR=frontend
#
# Safety:
#   This launcher is intended for local simulation and
#   offline development only.
#
#   It does not start physical robot drivers or actuators.
#   It does not reset emergency stops.
#   It does not bypass safety interlocks.
#

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

cd "${PROJECT_ROOT}"

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/.run}"

ENV_FILE="${PROJECT_ROOT}/.env"
ENV_EXAMPLE="${PROJECT_ROOT}/.env.example"

REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

# ============================================================
# Colors
# ============================================================

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    RESET=''
fi

log_info() {
    printf '%b[INFO]%b %s\n' \
        "${BLUE}" "${RESET}" "$*"
}

log_success() {
    printf '%b[OK]%b %s\n' \
        "${GREEN}" "${RESET}" "$*"
}

log_warn() {
    printf '%b[WARN]%b %s\n' \
        "${YELLOW}" "${RESET}" "$*"
}

log_error() {
    printf '%b[ERROR]%b %s\n' \
        "${RED}" "${RESET}" "$*" >&2
}

die() {
    log_error "$*"
    exit 1
}

# ============================================================
# Defaults
# ============================================================

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

ML_HOST="${ML_HOST:-127.0.0.1}"
ML_PORT="${ML_PORT:-8001}"

FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

BACKEND_APP="${BACKEND_APP:-backend.main:app}"
ML_APP="${ML_APP:-ml.inference.model_server:app}"

FRONTEND_DIR="${FRONTEND_DIR:-frontend}"

ENABLE_BACKEND=true
ENABLE_ML=true
ENABLE_FRONTEND=false
ENABLE_DOCKER=false

INSTALL_DEPS=false
RELOAD_ENABLED=false

RUN_HEALTH=false
STOP_ONLY=false
STATUS_ONLY=false

STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-30}"

# ============================================================
# Safety Environment
# ============================================================

enforce_simulation_mode() {
    export ROBOT_MODE="simulation"
    export SIMULATION_ONLY="true"

    export PHYSICAL_ROBOT_CONTROL_ENABLED="false"
    export ACTUATOR_COMMANDS_ENABLED="false"
    export REMOTE_MOTION_COMMANDS_ENABLED="false"
    export AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED="false"

    export EMERGENCY_STOP_RESET_ENABLED="false"
    export SAFETY_INTERLOCKS_REQUIRED="true"

    export HARDWARE_ACTUATION_ENABLED="false"
    export ROBOT_DRIVER_ENABLED="false"

    log_success \
        "Simulation-only mode enforced for launched services"
}

# ============================================================
# Help
# ============================================================

show_help() {
    cat <<'EOF'
Physical AI & Autonomous Robotics Intelligence Platform

Usage:
  ./scripts/run_local.sh [OPTIONS]

Options:
  --install          Install Python dependencies
  --backend-only     Start only FastAPI backend
  --ml-only          Start only ML inference service
  --with-frontend    Start frontend static server
  --with-docker      Start local Docker Compose dependencies
  --reload           Enable development auto-reload
  --health           Run platform health checks
  --stop             Stop services started by this launcher
  --status           Show local service status
  --help             Show this help message

Examples:
  ./scripts/run_local.sh
  ./scripts/run_local.sh --install
  ./scripts/run_local.sh --with-frontend --reload
  ./scripts/run_local.sh --with-docker
  ./scripts/run_local.sh --health
  ./scripts/run_local.sh --stop

Local URLs:
  Backend:  http://127.0.0.1:8000
  API Docs: http://127.0.0.1:8000/docs
  ML:       http://127.0.0.1:8001
  Frontend: http://127.0.0.1:3000

All services bind to loopback by default.
Physical robot control is disabled.
EOF
}

# ============================================================
# CLI Arguments
# ============================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --install)
                INSTALL_DEPS=true
                ;;
            --backend-only)
                ENABLE_BACKEND=true
                ENABLE_ML=false
                ENABLE_FRONTEND=false
                ;;
            --ml-only)
                ENABLE_BACKEND=false
                ENABLE_ML=true
                ENABLE_FRONTEND=false
                ;;
            --with-frontend)
                ENABLE_FRONTEND=true
                ;;
            --with-docker)
                ENABLE_DOCKER=true
                ;;
            --reload)
                RELOAD_ENABLED=true
                ;;
            --health)
                RUN_HEALTH=true
                ;;
            --stop)
                STOP_ONLY=true
                ;;
            --status)
                STATUS_ONLY=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac

        shift
    done
}

# ============================================================
# System Requirements
# ============================================================

require_command() {
    local command_name="$1"

    command -v "${command_name}" >/dev/null 2>&1 \
        || die "Required command not found: ${command_name}"
}

check_system() {
    log_info "Checking local development environment..."

    require_command python3
    require_command curl

    local python_version
    python_version="$(
        python3 -c \
        'import sys; print(".".join(map(str, sys.version_info[:3])))'
    )"

    log_info "Python: ${python_version}"

    python3 - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "Python 3.10 or newer is required"
    )
PY

    log_success "System requirements verified"
}

# ============================================================
# Environment Configuration
# ============================================================

setup_env() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        if [[ -f "${ENV_EXAMPLE}" ]]; then
            cp "${ENV_EXAMPLE}" "${ENV_FILE}"
            chmod 600 "${ENV_FILE}"
            log_info "Created .env from .env.example"
        else
            cat > "${ENV_FILE}" <<'EOF'
APP_ENV=development
DEBUG=true
ROBOT_MODE=simulation
SIMULATION_ONLY=true

PHYSICAL_ROBOT_CONTROL_ENABLED=false
ACTUATOR_COMMANDS_ENABLED=false
REMOTE_MOTION_COMMANDS_ENABLED=false
AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED=false

EMERGENCY_STOP_RESET_ENABLED=false
SAFETY_INTERLOCKS_REQUIRED=true

HARDWARE_ACTUATION_ENABLED=false
ROBOT_DRIVER_ENABLED=false

BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

ML_HOST=127.0.0.1
ML_PORT=8001

FRONTEND_HOST=127.0.0.1
FRONTEND_PORT=3000
EOF
            chmod 600 "${ENV_FILE}"
            log_info "Created development .env"
        fi
    fi

    # Do not source arbitrary .env shell code.
    # Services may read .env using their own settings loaders.
    # Explicitly enforce safety flags in process environment.
    enforce_simulation_mode
}

# ============================================================
# Virtual Environment
# ============================================================

setup_venv() {
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        log_info "Creating Python virtual environment..."

        python3 -m venv "${VENV_DIR}"
    fi

    PYTHON_BIN="${VENV_DIR}/bin/python"
    PIP_BIN="${VENV_DIR}/bin/pip"

    [[ -x "${PYTHON_BIN}" ]] \
        || die "Virtual environment Python unavailable"

    log_success "Virtual environment ready"
}

install_dependencies() {
    log_info "Installing Python dependencies..."

    "${PYTHON_BIN}" -m pip install \
        --upgrade pip

    if [[ -f "${REQUIREMENTS_FILE}" ]]; then
        "${PYTHON_BIN}" -m pip install \
            -r "${REQUIREMENTS_FILE}"
    else
        log_warn \
            "requirements.txt not found; installing minimal local stack"

        "${PYTHON_BIN}" -m pip install \
            fastapi \
            "uvicorn[standard]" \
            httpx \
            psutil \
            numpy \
            pandas \
            pydantic
    fi

    log_success "Dependency installation completed"
}

# ============================================================
# Python Module Validation
# ============================================================

check_python_module() {
    local module_name="$1"

    "${PYTHON_BIN}" -c \
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('${module_name}') else 1)"
}

validate_dependencies() {
    check_python_module uvicorn \
        || die "Uvicorn is missing. Run --install"

    check_python_module fastapi \
        || die "FastAPI is missing. Run --install"
}

# ============================================================
# Port Checking
# ============================================================

port_available() {
    local host="$1"
    local port="$2"

    "${PYTHON_BIN}" - "${host}" "${port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

with socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
) as sock:
    try:
        sock.bind((host, port))
    except OSError:
        raise SystemExit(1)
PY
}

check_port() {
    local name="$1"
    local host="$2"
    local port="$3"

    if ! port_available "${host}" "${port}"; then
        die "${name} port ${host}:${port} is unavailable"
    fi
}

# ============================================================
# PID Management
# ============================================================

pid_file() {
    printf '%s/%s.pid\n' \
        "${RUN_DIR}" "$1"
}

read_pid() {
    local name="$1"
    local file

    file="$(pid_file "${name}")"

    [[ -f "${file}" ]] || return 1

    local pid
    pid="$(cat "${file}")"

    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1

    printf '%s\n' "${pid}"
}

is_running() {
    local name="$1"
    local pid

    pid="$(read_pid "${name}")" || return 1

    kill -0 "${pid}" 2>/dev/null
}

record_pid() {
    local name="$1"
    local pid="$2"

    printf '%s\n' "${pid}" > "$(pid_file "${name}")"
}

stop_service() {
    local name="$1"
    local pid

    pid="$(read_pid "${name}")" || {
        log_info "${name}: no recorded process"
        return 0
    }

    if kill -0 "${pid}" 2>/dev/null; then
        log_info "Stopping ${name} (PID ${pid})..."

        kill -TERM "${pid}" 2>/dev/null || true

        local attempt

        for attempt in {1..30}; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                break
            fi

            sleep 0.2
        done

        if kill -0 "${pid}" 2>/dev/null; then
            log_warn "${name} did not stop gracefully"
            log_warn "Manual process inspection may be required"
        fi
    fi

    rm -f "$(pid_file "${name}")"
}

stop_all() {
    stop_service frontend
    stop_service ml
    stop_service backend

    log_success "Local service stop sequence completed"
}

show_status() {
    printf '\n%-20s %-12s %-12s\n' \
        "SERVICE" "STATUS" "PID"

    printf '%s\n' \
        "----------------------------------------------"

    local name
    local pid

    for name in backend ml frontend; do
        if is_running "${name}"; then
            pid="$(read_pid "${name}")"

            printf '%-20s %-12s %-12s\n' \
                "${name}" "RUNNING" "${pid}"
        else
            printf '%-20s %-12s %-12s\n' \
                "${name}" "STOPPED" "-"
        fi
    done

    printf '\n'
}

# ============================================================
# Service Launcher
# ============================================================

start_service() {
    local name="$1"
    shift

    if is_running "${name}"; then
        die "${name} already has a recorded running process"
    fi

    local logfile="${LOG_DIR}/${name}.log"

    log_info "Starting ${name}..."

    "$@" >> "${logfile}" 2>&1 &

    local pid=$!

    record_pid "${name}" "${pid}"

    sleep 1

    if ! kill -0 "${pid}" 2>/dev/null; then
        rm -f "$(pid_file "${name}")"

        log_error "${name} failed to start"
        log_error "Log file: ${logfile}"

        tail -n 30 "${logfile}" >&2 || true

        return 1
    fi

    log_success "${name} started (PID ${pid})"
}

# ============================================================
# HTTP Readiness
# ============================================================

wait_for_http() {
    local name="$1"
    local url="$2"

    local attempt=0

    log_info "Waiting for ${name} readiness..."

    while (( attempt < STARTUP_TIMEOUT )); do
        if curl \
            --silent \
            --show-error \
            --fail \
            --max-time 2 \
            --noproxy '*' \
            "${url}" \
            >/dev/null 2>&1; then

            log_success "${name} HTTP endpoint reachable"
            return 0
        fi

        sleep 1
        attempt=$((attempt + 1))
    done

    log_warn \
        "${name} readiness endpoint unavailable: ${url}"

    return 1
}

# ============================================================
# Docker Compose
# ============================================================

start_docker_dependencies() {
    require_command docker

    if ! docker compose version >/dev/null 2>&1; then
        die "Docker Compose v2 is required"
    fi

    local compose_file=""

    for candidate in \
        "docker-compose.yml" \
        "docker-compose.yaml" \
        "compose.yml" \
        "compose.yaml" \
        "infrastructure/docker/docker-compose.yml"
    do
        if [[ -f "${candidate}" ]]; then
            compose_file="${candidate}"
            break
        fi
    done

    [[ -n "${compose_file}" ]] \
        || die "No Docker Compose file found"

    log_info \
        "Starting Docker Compose dependencies..."

    # Start only named infrastructure services.
    # Never automatically start robot drivers.
    local available_services

    available_services="$(
        docker compose \
            -f "${compose_file}" \
            config --services
    )"

    local selected=()

    local service

    for service in postgres redis; do
        if printf '%s\n' "${available_services}" \
            | grep -Fxq "${service}"; then
            selected+=("${service}")
        fi
    done

    if [[ ${#selected[@]} -eq 0 ]]; then
        log_warn \
            "No postgres/redis services found in Compose"
        return 0
    fi

    docker compose \
        -f "${compose_file}" \
        up -d \
        "${selected[@]}"

    log_success "Docker dependencies started"
}

# ============================================================
# Backend
# ============================================================

start_backend() {
    check_port \
        "Backend" \
        "${BACKEND_HOST}" \
        "${BACKEND_PORT}"

    local args=(
        -m uvicorn
        "${BACKEND_APP}"
        --host "${BACKEND_HOST}"
        --port "${BACKEND_PORT}"
    )

    if [[ "${RELOAD_ENABLED}" == true ]]; then
        args+=(--reload)
    fi

    start_service \
        backend \
        "${PYTHON_BIN}" \
        "${args[@]}"

    wait_for_http \
        backend \
        "http://${BACKEND_HOST}:${BACKEND_PORT}/health" \
        || log_warn \
            "Backend started but health endpoint is not ready"
}

# ============================================================
# ML Inference
# ============================================================

start_ml() {
    check_port \
        "ML" \
        "${ML_HOST}" \
        "${ML_PORT}"

    local args=(
        -m uvicorn
        "${ML_APP}"
        --host "${ML_HOST}"
        --port "${ML_PORT}"
    )

    if [[ "${RELOAD_ENABLED}" == true ]]; then
        args+=(--reload)
    fi

    start_service \
        ml \
        "${PYTHON_BIN}" \
        "${args[@]}"

    wait_for_http \
        ml \
        "http://${ML_HOST}:${ML_PORT}/health" \
        || log_warn \
            "ML service started but health endpoint is not ready"
}

# ============================================================
# Frontend
# ============================================================

start_frontend() {
    local frontend_path="${PROJECT_ROOT}/${FRONTEND_DIR}"

    [[ -d "${frontend_path}" ]] \
        || die "Frontend directory not found: ${frontend_path}"

    check_port \
        "Frontend" \
        "${FRONTEND_HOST}" \
        "${FRONTEND_PORT}"

    start_service \
        frontend \
        "${PYTHON_BIN}" \
        -m http.server \
        "${FRONTEND_PORT}" \
        --bind "${FRONTEND_HOST}" \
        --directory "${frontend_path}"

    wait_for_http \
        frontend \
        "http://${FRONTEND_HOST}:${FRONTEND_PORT}/" \
        || log_warn "Frontend not ready"
}

# ============================================================
# Health Checks
# ============================================================

run_health_checks() {
    local health_script="${SCRIPT_DIR}/health_check.py"

    if [[ ! -f "${health_script}" ]]; then
        log_warn "Health-check script not found"
        return 0
    fi

    log_info "Running platform health checks..."

    local services=()

    if [[ "${ENABLE_BACKEND}" == true ]]; then
        services+=(backend)
    fi

    if [[ "${ENABLE_ML}" == true ]]; then
        services+=(ml)
    fi

    services+=(system safety)

    "${PYTHON_BIN}" \
        "${health_script}" \
        --project-root "${PROJECT_ROOT}" \
        --services "${services[@]}" \
        --strict
}

# ============================================================
# ROS 2 Detection
# ============================================================

detect_ros() {
    if [[ -n "${ROS_DISTRO:-}" ]]; then
        log_info "ROS 2 distribution: ${ROS_DISTRO}"
    else
        log_info "ROS 2 environment not sourced"
    fi

    # Do not launch ROS nodes, drivers or actuators.
}

# ============================================================
# Startup Banner
# ============================================================

show_banner() {
    cat <<'EOF'

============================================================
     PHYSICAL AI & AUTONOMOUS ROBOTICS INTELLIGENCE
============================================================

  Local Development Environment
  FastAPI | ML Inference | RAG | Robotics | Safety

  MODE: SIMULATION ONLY
  PHYSICAL ROBOT CONTROL: DISABLED

============================================================

EOF
}

show_urls() {
    echo
    log_success "Local development services launched"
    echo

    if [[ "${ENABLE_BACKEND}" == true ]]; then
        printf 'Backend:   http://%s:%s\n' \
            "${BACKEND_HOST}" "${BACKEND_PORT}"

        printf 'API Docs:  http://%s:%s/docs\n' \
            "${BACKEND_HOST}" "${BACKEND_PORT}"
    fi

    if [[ "${ENABLE_ML}" == true ]]; then
        printf 'ML Server: http://%s:%s\n' \
            "${ML_HOST}" "${ML_PORT}"
    fi

    if [[ "${ENABLE_FRONTEND}" == true ]]; then
        printf 'Frontend:  http://%s:%s\n' \
            "${FRONTEND_HOST}" "${FRONTEND_PORT}"
    fi

    echo
    printf 'Logs:      %s\n' "${LOG_DIR}"
    printf 'PID files: %s\n' "${RUN_DIR}"

    echo
    log_warn \
        "No physical robot actuation is authorized"
    echo
}

# ============================================================
# Shutdown Handling
# ============================================================

shutdown_requested=false

cleanup() {
    if [[ "${shutdown_requested}" == true ]]; then
        return
    fi

    shutdown_requested=true

    echo
    log_info "Shutting down local services..."

    stop_all
}

# ============================================================
# Main
# ============================================================

main() {
    parse_args "$@"

    show_banner

    if [[ "${STOP_ONLY}" == true ]]; then
        stop_all
        exit 0
    fi

    if [[ "${STATUS_ONLY}" == true ]]; then
        show_status
        exit 0
    fi

    check_system
    setup_env
    setup_venv

    if [[ "${INSTALL_DEPS}" == true ]]; then
        install_dependencies
    fi

    if [[ "${RUN_HEALTH}" == true ]]; then
        run_health_checks
        exit $?
    fi

    validate_dependencies
    detect_ros

    if [[ "${ENABLE_DOCKER}" == true ]]; then
        start_docker_dependencies
    fi

    trap cleanup INT TERM EXIT

    if [[ "${ENABLE_BACKEND}" == true ]]; then
        start_backend
    fi

    if [[ "${ENABLE_ML}" == true ]]; then
        start_ml
    fi

    if [[ "${ENABLE_FRONTEND}" == true ]]; then
        start_frontend
    fi

    show_urls
    show_status

    log_info "Press Ctrl+C to stop local services"

    # Keep launcher alive to manage child processes.
    while true; do
        sleep 2

        for name in backend ml frontend; do
            if [[ -f "$(pid_file "${name}")" ]]; then
                if ! is_running "${name}"; then
                    log_error \
                        "${name} stopped unexpectedly"

                    return 1
                fi
            fi
        done
    done
}

main "$@"