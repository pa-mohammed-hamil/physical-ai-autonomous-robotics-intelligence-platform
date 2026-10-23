#!/usr/bin/env bash
# ============================================================
# Physical AI & Autonomous Robotics Intelligence Platform
# File: scripts/bootstrap.sh
#
# Purpose:
#   Initialize a local development environment for the
#   robotics, machine learning, agentic AI, and simulation
#   platform.
#
# Features:
#   - Linux/macOS/WSL support
#   - Python virtual environment
#   - Python dependency installation
#   - Optional ML, ROS 2, and development dependencies
#   - Project directory initialization
#   - Environment configuration
#   - Local storage initialization
#   - Basic hardware detection
#   - Dependency verification
#   - Simulation-only safety defaults
#   - Idempotent execution
#
# Usage:
#   chmod +x scripts/bootstrap.sh
#   ./scripts/bootstrap.sh
#
#   ./scripts/bootstrap.sh --with-ml
#   ./scripts/bootstrap.sh --with-dev
#   ./scripts/bootstrap.sh --with-ros
#   ./scripts/bootstrap.sh --all
#
#   ./scripts/bootstrap.sh --skip-install
#   ./scripts/bootstrap.sh --check
#
# Safety:
#   Does not start physical robots, actuators, controllers,
#   or emergency-stop reset operations.
# ============================================================

set -Eeuo pipefail
IFS=$'\n\t'

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

readonly SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
    pwd
)"

readonly PROJECT_ROOT="$(
    cd -- "${SCRIPT_DIR}/.." &&
    pwd
)"

readonly MIN_PYTHON_MAJOR=3
readonly MIN_PYTHON_MINOR=10

readonly DEFAULT_VENV="${PROJECT_ROOT}/.venv"
readonly DEFAULT_ENV_FILE="${PROJECT_ROOT}/.env"

VENV_DIR="${VENV_DIR:-$DEFAULT_VENV}"
ENV_FILE="${ENV_FILE:-$DEFAULT_ENV_FILE}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

WITH_ML=false
WITH_DEV=false
WITH_ROS=false
SKIP_INSTALL=false
CHECK_ONLY=false
FORCE_ENV=false
OFFLINE=false

BOOTSTRAP_STARTED_AT="$(date +%s)"

# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[0;33m'
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

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

log_info() {
    printf '%s[INFO]%s %s\n' \
        "$BLUE" "$RESET" "$*"
}

log_success() {
    printf '%s[OK]%s %s\n' \
        "$GREEN" "$RESET" "$*"
}

log_warning() {
    printf '%s[WARN]%s %s\n' \
        "$YELLOW" "$RESET" "$*" >&2
}

log_error() {
    printf '%s[ERROR]%s %s\n' \
        "$RED" "$RESET" "$*" >&2
}

log_section() {
    printf '\n%s========== %s ==========%s\n' \
        "$CYAN" "$*" "$RESET"
}

die() {
    log_error "$*"
    exit 1
}

on_error() {
    local exit_code=$?

    log_error \
        "Bootstrap failed at line ${BASH_LINENO[0]} (exit ${exit_code})"

    exit "$exit_code"
}

trap on_error ERR

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------

show_help() {
    cat <<'HELP'
Physical AI & Autonomous Robotics Platform Bootstrap

Usage:
  ./scripts/bootstrap.sh [OPTIONS]

Options:
  --with-ml        Install machine learning dependencies
  --with-dev       Install development/testing dependencies
  --with-ros       Verify ROS 2 and install optional ROS Python dependencies
  --all            Enable ML, development, and ROS options
  --skip-install   Initialize without installing Python packages
  --offline        Avoid network package installation
  --check          Run environment checks only
  --force-env      Regenerate .env (backs up existing file)
  --python PATH    Choose Python executable
  --venv PATH      Choose virtual environment directory
  --help           Show this help

Examples:
  ./scripts/bootstrap.sh
  ./scripts/bootstrap.sh --with-ml --with-dev
  ./scripts/bootstrap.sh --all
  ./scripts/bootstrap.sh --offline
  ./scripts/bootstrap.sh --check

Notes:
  - Run from a trusted project checkout.
  - ROS 2 installation is not performed automatically.
  - Physical robot execution remains disabled.
HELP
}

# ------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------

parse_arguments() {
    while (($# > 0)); do
        case "$1" in
            --with-ml)
                WITH_ML=true
                ;;

            --with-dev)
                WITH_DEV=true
                ;;

            --with-ros)
                WITH_ROS=true
                ;;

            --all)
                WITH_ML=true
                WITH_DEV=true
                WITH_ROS=true
                ;;

            --skip-install)
                SKIP_INSTALL=true
                ;;

            --offline)
                OFFLINE=true
                SKIP_INSTALL=true
                ;;

            --check)
                CHECK_ONLY=true
                ;;

            --force-env)
                FORCE_ENV=true
                ;;

            --python)
                (($# >= 2)) || die \
                    "--python requires a value"

                PYTHON_BIN="$2"
                shift
                ;;

            --venv)
                (($# >= 2)) || die \
                    "--venv requires a value"

                VENV_DIR="$2"
                shift
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

# ------------------------------------------------------------
# System Detection
# ------------------------------------------------------------

detect_operating_system() {
    log_section "Operating System"

    local system
    system="$(uname -s)"

    case "$system" in
        Linux)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                log_info "Environment: Windows Subsystem for Linux"
            else
                log_info "Environment: Linux"
            fi
            ;;

        Darwin)
            log_info "Environment: macOS"
            ;;

        *)
            log_warning \
                "Untested operating system: ${system}"
            ;;
    esac

    log_info "Architecture: $(uname -m)"
    log_info "Project root: ${PROJECT_ROOT}"
}

# ------------------------------------------------------------
# Required Commands
# ------------------------------------------------------------

require_command() {
    local command_name="$1"

    command -v "$command_name" >/dev/null 2>&1 ||
        die "Required command not found: ${command_name}"
}

check_required_commands() {
    log_section "Required Commands"

    require_command bash
    require_command git
    require_command "$PYTHON_BIN"

    log_success "Required commands are available"

    if command -v curl >/dev/null 2>&1; then
        log_info "curl available"
    fi

    if command -v docker >/dev/null 2>&1; then
        log_info "Docker available"
    else
        log_warning \
            "Docker unavailable; container workflows disabled"
    fi
}

# ------------------------------------------------------------
# Python Version
# ------------------------------------------------------------

check_python_version() {
    log_section "Python Version"

    local version

    version="$(
        "$PYTHON_BIN" -c \
            'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )"

    log_info "Detected Python: ${version}"

    "$PYTHON_BIN" -c "
import sys
required = (
    ${MIN_PYTHON_MAJOR},
    ${MIN_PYTHON_MINOR},
)
if sys.version_info[:2] < required:
    raise SystemExit(
        'Python 3.10 or newer is required'
    )
" || die "Unsupported Python version"

    log_success "Python version supported"
}

# ------------------------------------------------------------
# Hardware Information
# ------------------------------------------------------------

detect_hardware() {
    log_section "Hardware"

    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "NVIDIA GPU detected"

        nvidia-smi \
            --query-gpu=name,memory.total \
            --format=csv,noheader \
            2>/dev/null || true
    else
        log_info "No NVIDIA GPU detected by nvidia-smi"
    fi

    if command -v free >/dev/null 2>&1; then
        free -h || true
    fi

    if command -v df >/dev/null 2>&1; then
        df -h "$PROJECT_ROOT" || true
    fi

    if command -v nproc >/dev/null 2>&1; then
        log_info "Logical CPU cores: $(nproc)"
    fi
}

# ------------------------------------------------------------
# Python Virtual Environment
# ------------------------------------------------------------

create_virtual_environment() {
    log_section "Python Virtual Environment"

    if [[ -f "${VENV_DIR}/bin/python" ]]; then
        log_success \
            "Existing virtual environment: ${VENV_DIR}"
        return
    fi

    log_info "Creating virtual environment"

    "$PYTHON_BIN" -m venv "$VENV_DIR" ||
        die "Unable to create Python virtual environment"

    log_success \
        "Virtual environment created: ${VENV_DIR}"
}

activate_virtual_environment() {
    [[ -f "${VENV_DIR}/bin/activate" ]] ||
        die "Virtual environment activation file missing"

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    log_info "Active Python: $(command -v python)"
}

# ------------------------------------------------------------
# Dependency Installation
# ------------------------------------------------------------

install_requirements_file() {
    local requirements_file="$1"

    if [[ ! -f "$requirements_file" ]]; then
        log_warning \
            "Requirements file not found: ${requirements_file}"
        return
    fi

    log_info \
        "Installing $(basename "$requirements_file")"

    python -m pip install \
        -r "$requirements_file"
}

install_base_dependencies() {
    log_section "Base Dependencies"

    if [[ "$SKIP_INSTALL" == true ]]; then
        log_warning "Dependency installation skipped"
        return
    fi

    local base_requirements=""

    for candidate in \
        "${PROJECT_ROOT}/requirements.txt" \
        "${PROJECT_ROOT}/backend/requirements.txt"
    do
        if [[ -f "$candidate" ]]; then
            base_requirements="$candidate"
            break
        fi
    done

    if [[ -n "$base_requirements" ]]; then
        install_requirements_file "$base_requirements"
    elif [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
        log_info "Installing local Python project"

        python -m pip install \
            -e "$PROJECT_ROOT"
    else
        log_warning \
            "No base requirements.txt or pyproject.toml found"
        log_info \
            "Add project dependencies before running services"
    fi
}

install_ml_dependencies() {
    if [[ "$WITH_ML" != true ]]; then
        return
    fi

    log_section "Machine Learning Dependencies"

    if [[ "$SKIP_INSTALL" == true ]]; then
        log_warning "ML dependency installation skipped"
        return
    fi

    local candidates=(
        "${PROJECT_ROOT}/requirements-ml.txt"
        "${PROJECT_ROOT}/ml/requirements.txt"
    )

    local requirements_file=""

    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            requirements_file="$candidate"
            break
        fi
    done

    if [[ -n "$requirements_file" ]]; then
        install_requirements_file "$requirements_file"
    else
        log_warning \
            "No ML requirements file found"

        log_info \
            "Expected: requirements-ml.txt or ml/requirements.txt"
    fi
}

install_dev_dependencies() {
    if [[ "$WITH_DEV" != true ]]; then
        return
    fi

    log_section "Development Dependencies"

    if [[ "$SKIP_INSTALL" == true ]]; then
        log_warning "Development installation skipped"
        return
    fi

    local requirements_file="${PROJECT_ROOT}/requirements-dev.txt"

    if [[ -f "$requirements_file" ]]; then
        install_requirements_file "$requirements_file"
    else
        log_warning \
            "No requirements-dev.txt found"
    fi
}

# ------------------------------------------------------------
# ROS 2 Integration
# ------------------------------------------------------------

check_ros_environment() {
    if [[ "$WITH_ROS" != true ]]; then
        return
    fi

    log_section "ROS 2 Environment"

    if [[ -n "${ROS_DISTRO:-}" ]]; then
        log_success \
            "ROS 2 distribution: ${ROS_DISTRO}"
    else
        log_warning \
            "ROS_DISTRO is not set"

        log_info \
            "Source an installed ROS 2 environment before use"
    fi

    if command -v ros2 >/dev/null 2>&1; then
        log_success "ROS 2 CLI detected"
    else
        log_warning "ROS 2 CLI not detected"
    fi

    # ROS Python modules are commonly supplied by the
    # system's ROS distribution, not by generic pip packages.
    # Do not automatically pip-install rclpy or ROS packages.

    log_info \
        "ROS 2 installation is managed separately"

    log_info \
        "No ROS nodes or robot controllers are started"
}

# ------------------------------------------------------------
# Directory Initialization
# ------------------------------------------------------------

create_project_directories() {
    log_section "Project Directories"

    local directories=(
        "data"
        "data/raw"
        "data/processed"
        "data/simulation"
        "data/telemetry"
        "data/robot_logs"
        "data/world_model"
        "data/vla"
        "data/perception"
        "data/anomaly"
        "data/failure"
        "data/rul"
        "data/part_detection"

        "artifacts"
        "artifacts/models"
        "artifacts/checkpoints"
        "artifacts/world_model"
        "artifacts/vla"
        "artifacts/perception"
        "artifacts/anomaly"
        "artifacts/failure"
        "artifacts/rul"
        "artifacts/part_detector"

        "logs"
        "logs/backend"
        "logs/ml"
        "logs/agents"
        "logs/simulation"
        "logs/security"

        "storage"
        "storage/model_registry"
        "storage/vector_store"
        "storage/cache"
        "storage/reports"
        "storage/uploads"
        "storage/audit"

        "tmp"
        "tmp/checkpoints"
        "tmp/uploads"
        "tmp/processing"

        "reports"
        "reports/training"
        "reports/evaluation"
        "reports/safety"

        "simulation/results"
        "simulation/recordings"

        "monitoring/exports"
    )

    local directory

    for directory in "${directories[@]}"; do
        mkdir -p \
            "${PROJECT_ROOT}/${directory}"
    done

    chmod 700 \
        "${PROJECT_ROOT}/storage/audit" \
        "${PROJECT_ROOT}/storage/uploads" \
        "${PROJECT_ROOT}/tmp/uploads" \
        2>/dev/null || true

    log_success \
        "Project runtime directories initialized"
}

# ------------------------------------------------------------
# Environment Configuration
# ------------------------------------------------------------

create_environment_file() {
    log_section "Environment Configuration"

    if [[ -f "$ENV_FILE" && "$FORCE_ENV" != true ]]; then
        log_success \
            "Existing .env preserved"
        return
    fi

    if [[ -f "$ENV_FILE" && "$FORCE_ENV" == true ]]; then
        local backup_file

        backup_file="${ENV_FILE}.backup.$(date +%Y%m%d%H%M%S)"

        cp -p \
            "$ENV_FILE" \
            "$backup_file"

        chmod 600 "$backup_file"

        log_warning \
            "Previous environment backed up: ${backup_file}"
    fi

    local original_umask
    original_umask="$(umask)"
    umask 077

    cat > "$ENV_FILE" <<'ENV'
# ============================================================
# Physical AI & Autonomous Robotics Intelligence Platform
# Local Development Environment
# ============================================================

APP_NAME=physical-ai-robotics-platform
APP_ENV=development
DEBUG=false

# ------------------------------------------------------------
# API
# ------------------------------------------------------------

API_HOST=127.0.0.1
API_PORT=8000

# ------------------------------------------------------------
# Database
# ------------------------------------------------------------

DATABASE_URL=sqlite:///./storage/platform.db

# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------

REDIS_URL=redis://127.0.0.1:6379/0

# ------------------------------------------------------------
# Machine Learning
# ------------------------------------------------------------

ML_DEVICE=auto
MODEL_REGISTRY_PATH=storage/model_registry
MODEL_ARTIFACT_PATH=artifacts/models

MODEL_INFERENCE_TIMEOUT_SECONDS=30
MODEL_MAX_BATCH_SIZE=32

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

SIMULATION_ENABLED=true
SIMULATION_BACKEND=none

# ------------------------------------------------------------
# Robotics Safety
# ------------------------------------------------------------

ROBOT_EXECUTION_MODE=simulation
PHYSICAL_ROBOT_CONTROL_ENABLED=false
ACTUATOR_COMMANDS_ENABLED=false
REMOTE_MOTION_COMMANDS_ENABLED=false
AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED=false

HUMAN_APPROVAL_REQUIRED=true
SAFETY_INTERLOCKS_REQUIRED=true
EMERGENCY_STOP_RESET_ENABLED=false

# ------------------------------------------------------------
# Agents
# ------------------------------------------------------------

AGENT_EXECUTION_ENABLED=true
AGENT_PHYSICAL_ACTIONS_ENABLED=false
AGENT_HUMAN_APPROVAL_REQUIRED=true

# ------------------------------------------------------------
# Security
# ------------------------------------------------------------

AUTH_ENABLED=true
RBAC_ENABLED=true
AUDIT_LOGGING_ENABLED=true

ALLOW_ANONYMOUS_ROBOT_CONTROL=false
ALLOW_UNVERIFIED_MODEL_DEPLOYMENT=false

# ------------------------------------------------------------
# External AI Providers
# ------------------------------------------------------------

# Add credentials manually using a secret manager or
# local environment variables. Never commit secrets.

OPENAI_API_KEY=
GOOGLE_API_KEY=

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

LOG_LEVEL=INFO
LOG_FORMAT=json
ENV

    umask "$original_umask"

    chmod 600 "$ENV_FILE"

    log_success \
        "Environment file created: ${ENV_FILE}"
}

# ------------------------------------------------------------
# Git Ignore
# ------------------------------------------------------------

update_gitignore() {
    log_section "Git Ignore"

    local gitignore="${PROJECT_ROOT}/.gitignore"

    touch "$gitignore"

    local entries=(
        ".env"
        ".env.*"
        "!.env.example"
        ".venv/"
        "__pycache__/"
        "*.py[cod]"
        ".pytest_cache/"
        ".mypy_cache/"
        ".ruff_cache/"
        ".coverage"
        "htmlcov/"
        "logs/"
        "tmp/"
        "storage/"
        "artifacts/"
        "data/raw/"
        "data/processed/"
        "simulation/results/"
        "simulation/recordings/"
        "*.pt"
        "*.pth"
        "*.ckpt"
    )

    local entry

    for entry in "${entries[@]}"; do
        if ! grep -Fxq \
            "$entry" \
            "$gitignore"
        then
            printf '%s\n' "$entry" >> "$gitignore"
        fi
    done

    log_success ".gitignore updated"
}

# ------------------------------------------------------------
# Optional Docker Check
# ------------------------------------------------------------

check_docker() {
    log_section "Docker"

    if ! command -v docker >/dev/null 2>&1; then
        log_warning "Docker is not installed"
        return
    fi

    docker --version

    if docker info >/dev/null 2>&1; then
        log_success "Docker daemon accessible"
    else
        log_warning \
            "Docker installed but daemon unavailable"
    fi

    # Never automatically start containers here.
}

# ------------------------------------------------------------
# Python Import Verification
# ------------------------------------------------------------

verify_python_modules() {
    log_section "Python Module Verification"

    python - <<'PY'
import importlib.util

modules = [
    "numpy",
    "pandas",
    "torch",
    "torchvision",
    "sklearn",
    "fastapi",
    "pydantic",
    "uvicorn",
]

print("Python dependency status:")

for name in modules:
    available = (
        importlib.util.find_spec(name) is not None
    )

    status = "AVAILABLE" if available else "MISSING"

    print(f"  {name:<16} {status}")
PY
}

# ------------------------------------------------------------
# Project Validation
# ------------------------------------------------------------

verify_project_structure() {
    log_section "Project Structure"

    local expected_directories=(
        "backend"
        "ml"
        "simulation"
        "robot_configs"
        "infrastructure"
        "scripts"
    )

    local directory

    for directory in "${expected_directories[@]}"; do
        if [[ -d "${PROJECT_ROOT}/${directory}" ]]; then
            log_success "${directory}/"
        else
            log_warning \
                "Missing optional directory: ${directory}/"
        fi
    done

    local expected_modules=(
        "ml/training/train_vla.py"
        "ml/training/train_world_model.py"
        "ml/training/train_anomaly.py"
        "ml/training/train_failure.py"
        "ml/training/train_rul.py"
        "ml/training/train_part_detector.py"
        "ml/inference/model_server.py"
        "ml/registry/model_registry.py"
    )

    local module

    for module in "${expected_modules[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${module}" ]]; then
            log_success "$module"
        else
            log_warning \
                "Module not found: ${module}"
        fi
    done
}

# ------------------------------------------------------------
# Safety Verification
# ------------------------------------------------------------

verify_safety_configuration() {
    log_section "Robotics Safety"

    if [[ ! -f "$ENV_FILE" ]]; then
        log_warning \
            "No .env file found for safety validation"
        return
    fi

    # Parse only known literal key/value pairs.
    # Do not source arbitrary .env files as shell scripts.

    local required_settings=(
        "PHYSICAL_ROBOT_CONTROL_ENABLED=false"
        "ACTUATOR_COMMANDS_ENABLED=false"
        "REMOTE_MOTION_COMMANDS_ENABLED=false"
        "AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED=false"
        "EMERGENCY_STOP_RESET_ENABLED=false"
        "SAFETY_INTERLOCKS_REQUIRED=true"
    )

    local setting

    for setting in "${required_settings[@]}"; do
        if grep -Fxq "$setting" "$ENV_FILE"; then
            log_success "$setting"
        else
            log_warning \
                "Missing simulation safety default: ${setting}"
        fi
    done

    log_info \
        "Bootstrap never sends robot control commands"

    log_info \
        "Safety requires enforcement in the runtime, not just .env"
}

# ------------------------------------------------------------
# Check-Only Mode
# ------------------------------------------------------------

run_checks() {
    log_section "Environment Checks"

    detect_operating_system
    check_required_commands
    check_python_version
    detect_hardware
    check_docker
    verify_project_structure

    if [[ -f "${VENV_DIR}/bin/python" ]]; then
        activate_virtual_environment
        verify_python_modules
    else
        log_warning \
            "Virtual environment not initialized"
    fi

    verify_safety_configuration

    log_success "Environment checks completed"
}

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print_summary() {
    local completed_at
    completed_at="$(date +%s)"

    local duration
    duration="$((completed_at - BOOTSTRAP_STARTED_AT))"

    log_section "Bootstrap Summary"

    printf 'Project root:        %s\n' "$PROJECT_ROOT"
    printf 'Virtual environment: %s\n' "$VENV_DIR"
    printf 'Environment file:    %s\n' "$ENV_FILE"
    printf 'ML dependencies:     %s\n' "$WITH_ML"
    printf 'Dev dependencies:    %s\n' "$WITH_DEV"
    printf 'ROS verification:    %s\n' "$WITH_ROS"
    printf 'Offline mode:        %s\n' "$OFFLINE"
    printf 'Duration:            %ss\n' "$duration"

    printf '\n'

    log_success "Bootstrap completed successfully"

    printf '\nActivate environment:\n'
    printf '  source "%s/bin/activate"\n' "$VENV_DIR"

    printf '\nExample ML training:\n'
    printf '  python -m ml.training.train_world_model --help\n'
    printf '  python -m ml.training.train_vla --help\n'

    printf '\nSafety mode:\n'
    printf '  Simulation only. Physical control disabled.\n'
}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

main() {
    parse_arguments "$@"

    if [[ "$CHECK_ONLY" == true ]]; then
        run_checks
        return
    fi

    log_section \
        "Physical AI & Autonomous Robotics Bootstrap"

    detect_operating_system
    check_required_commands
    check_python_version
    detect_hardware

    create_virtual_environment
    activate_virtual_environment

    install_base_dependencies
    install_ml_dependencies
    install_dev_dependencies

    check_ros_environment

    create_project_directories
    create_environment_file
    update_gitignore

    check_docker
    verify_python_modules
    verify_project_structure
    verify_safety_configuration

    print_summary
}

main "$@"