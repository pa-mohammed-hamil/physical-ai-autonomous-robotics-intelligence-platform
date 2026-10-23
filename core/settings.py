"""Settings module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Application Settings

Centralized configuration for:
- FastAPI application
- PostgreSQL / SQLAlchemy
- Redis and caching
- Authentication and JWT
- CORS and trusted hosts
- Rate limiting
- Logging and observability
- Prometheus and OpenTelemetry
- AI/LLM providers
- RAG and vector databases
- Robotics, ROS 2 and fleet management
- Sensors, telemetry and IoT
- Navigation, manipulation and VLA
- World models and simulation
- Predictive maintenance and repair intelligence
- Safety and emergency response
- File uploads and object storage
- MLOps and model registry

Dependencies:
    pip install pydantic-settings pydantic

Environment:
    Copy .env.example to .env and configure secrets.

Security:
    - Never hardcode production credentials.
    - Production JWT secrets must be explicitly configured.
    - Robot actuation is disabled by default.
    - AI-generated actions require independent authorization
      and physical safety enforcement.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BACKEND_ROOT = PROJECT_ROOT / "backend"

DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = DATA_DIR / "models"
ARTIFACT_DIR = DATA_DIR / "artifacts"

ENV_FILE = PROJECT_ROOT / ".env"


# ============================================================
# Enums
# ============================================================


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


class RateLimitBackend(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"


class CacheBackend(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"


class VectorDatabase(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    PGVECTOR = "pgvector"


class ObjectStorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    MINIO = "minio"


class SimulationEngine(str, Enum):
    MOCK = "mock"
    GAZEBO = "gazebo"
    ISAAC_SIM = "isaac_sim"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"
    WEBOTS = "webots"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


# ============================================================
# Validation Helpers
# ============================================================


def validate_positive(value: int) -> int:
    if value <= 0:
        raise ValueError("Value must be positive")
    return value


PositiveInt = Annotated[
    int,
    AfterValidator(validate_positive),
]


def parse_string_list(value: Any) -> list[str]:
    """
    Accept:
        ["a", "b"]
        '["a", "b"]'
        "a,b"
    """

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        if value.startswith("["):
            parsed = json.loads(value)

            if not isinstance(parsed, list):
                raise ValueError(
                    "Expected a JSON list"
                )

            return parse_string_list(parsed)

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    raise ValueError(
        "Expected a list or comma-separated string"
    )


# ============================================================
# Main Application Settings
# ============================================================


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    APP_NAME: str = (
        "Physical AI & Autonomous Robotics "
        "Intelligence Platform"
    )

    APP_SLUG: str = (
        "physical-ai-autonomous-robotics"
    )

    APP_VERSION: str = "1.0.0"

    APP_DESCRIPTION: str = (
        "AI-powered robotics intelligence, fleet "
        "management, predictive maintenance, "
        "simulation, perception and safety platform."
    )

    ENVIRONMENT: Environment = Environment.DEVELOPMENT

    DEBUG: bool = False

    API_V1_PREFIX: str = "/api/v1"

    HOST: str = "127.0.0.1"
    PORT: int = Field(default=8000, ge=1, le=65535)

    WORKERS: PositiveInt = 1

    RELOAD: bool = False

    ROOT_PATH: str = ""

    # --------------------------------------------------------
    # Security / Authentication
    # --------------------------------------------------------

    JWT_SECRET_KEY: SecretStr | None = None

    JWT_ALGORITHM: Literal["HS256"] = "HS256"

    JWT_ISSUER: str = "physical-ai-platform"

    JWT_AUDIENCE: str = "physical-ai-api"

    ACCESS_TOKEN_EXPIRE_MINUTES: PositiveInt = 30

    REFRESH_TOKEN_EXPIRE_DAYS: PositiveInt = 7

    PASSWORD_MIN_LENGTH: int = Field(
        default=12,
        ge=8,
        le=128,
    )

    REQUIRE_EMAIL_VERIFICATION: bool = False

    ENABLE_API_KEYS: bool = True

    ENABLE_RBAC: bool = True

    ENABLE_TENANT_ISOLATION: bool = True

    ENABLE_SECURITY_AUDIT: bool = True

    # --------------------------------------------------------
    # CORS / Trusted Hosts
    # --------------------------------------------------------

    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    )

    CORS_ALLOW_CREDENTIALS: bool = False

    CORS_METHODS: list[str] = Field(
        default_factory=lambda: [
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
        ]
    )

    CORS_HEADERS: list[str] = Field(
        default_factory=lambda: [
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-Correlation-ID",
        ]
    )

    TRUSTED_HOSTS: list[str] = Field(
        default_factory=lambda: [
            "localhost",
            "127.0.0.1",
            "testserver",
        ]
    )

    ENABLE_HSTS: bool = False

    HSTS_MAX_AGE_SECONDS: int = Field(
        default=31_536_000,
        ge=0,
    )

    # --------------------------------------------------------
    # PostgreSQL
    # --------------------------------------------------------

    DATABASE_URL: str = (
        "postgresql+asyncpg://"
        "postgres:postgres@localhost:5432/physical_ai"
    )

    DATABASE_ECHO: bool = False

    DATABASE_POOL_SIZE: PositiveInt = 10

    DATABASE_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
    )

    DATABASE_POOL_TIMEOUT: PositiveInt = 30

    DATABASE_POOL_RECYCLE: PositiveInt = 1800

    DATABASE_CONNECT_TIMEOUT: PositiveInt = 10

    DATABASE_SSL_REQUIRED: bool = False

    # --------------------------------------------------------
    # Redis
    # --------------------------------------------------------

    REDIS_URL: str = "redis://localhost:6379/0"

    REDIS_MAX_CONNECTIONS: PositiveInt = 50

    REDIS_SOCKET_TIMEOUT_SECONDS: float = Field(
        default=5.0,
        gt=0,
    )

    REDIS_CONNECT_TIMEOUT_SECONDS: float = Field(
        default=5.0,
        gt=0,
    )

    # --------------------------------------------------------
    # Caching
    # --------------------------------------------------------

    CACHE_ENABLED: bool = True

    CACHE_BACKEND: CacheBackend = CacheBackend.REDIS

    CACHE_DEFAULT_TTL_SECONDS: PositiveInt = 300

    CACHE_KEY_PREFIX: str = "physical_ai:cache"

    CACHE_MEMORY_MAX_ENTRIES: PositiveInt = 10_000

    # --------------------------------------------------------
    # Rate Limiting
    # --------------------------------------------------------

    RATE_LIMIT_ENABLED: bool = True

    RATE_LIMIT_BACKEND: RateLimitBackend = (
        RateLimitBackend.REDIS
    )

    RATE_LIMIT_REQUESTS: PositiveInt = 100

    RATE_LIMIT_WINDOW_SECONDS: PositiveInt = 60

    RATE_LIMIT_FAIL_CLOSED: bool = True

    # --------------------------------------------------------
    # HTTP / Middleware
    # --------------------------------------------------------

    MAX_REQUEST_BODY_BYTES: PositiveInt = (
        20 * 1024 * 1024
    )

    REQUEST_TIMEOUT_SECONDS: float = Field(
        default=60.0,
        gt=0,
    )

    ENABLE_REQUEST_LOGGING: bool = True

    ENABLE_REQUEST_ID: bool = True

    ENABLE_SECURITY_HEADERS: bool = True

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    LOG_LEVEL: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    LOG_FORMAT: LogFormat = LogFormat.JSON

    LOG_TO_FILE: bool = False

    LOG_FILE_PATH: Path = LOG_DIR / "app.log"

    LOG_MAX_BYTES: PositiveInt = 10 * 1024 * 1024

    LOG_BACKUP_COUNT: int = Field(
        default=5,
        ge=0,
    )

    LOG_REDACT_SECRETS: bool = True

    # --------------------------------------------------------
    # Metrics / Observability
    # --------------------------------------------------------

    METRICS_ENABLED: bool = True

    METRICS_ENDPOINT: str = "/metrics"

    OTEL_ENABLED: bool = True

    OTEL_SERVICE_NAME: str = (
        "physical-ai-autonomous-robotics"
    )

    OTEL_EXPORTER_OTLP_ENDPOINT: str = (
        "http://localhost:4318/v1/traces"
    )

    OTEL_EXPORT_ENABLED: bool = False

    OTEL_TRACE_SAMPLE_RATE: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
    )

    HEALTH_CHECK_TIMEOUT_SECONDS: float = Field(
        default=5.0,
        gt=0,
    )

    # --------------------------------------------------------
    # Event Bus / Streaming
    # --------------------------------------------------------

    EVENT_BUS_ENABLED: bool = True

    EVENT_BUS_BACKEND: Literal[
        "memory",
        "redis",
        "kafka",
    ] = "redis"

    EVENT_BUS_STREAM_PREFIX: str = (
        "physical_ai:events"
    )

    EVENT_BUS_MAX_QUEUE_SIZE: PositiveInt = 10_000

    KAFKA_ENABLED: bool = False

    KAFKA_BOOTSTRAP_SERVERS: str = (
        "localhost:9092"
    )

    KAFKA_CONSUMER_GROUP: str = (
        "physical-ai-backend"
    )

    MQTT_ENABLED: bool = False

    MQTT_BROKER_HOST: str = "localhost"

    MQTT_BROKER_PORT: int = Field(
        default=1883,
        ge=1,
        le=65535,
    )

    MQTT_USERNAME: str | None = None

    MQTT_PASSWORD: SecretStr | None = None

    MQTT_TLS_ENABLED: bool = False

    # --------------------------------------------------------
    # Robotics
    # --------------------------------------------------------

    ROBOTICS_ENABLED: bool = True

    ROBOT_COMMANDS_ENABLED: bool = False

    ROBOT_HARDWARE_CONTROL_ENABLED: bool = False

    ROBOT_SIMULATION_ONLY: bool = True

    ROBOT_HEARTBEAT_INTERVAL_SECONDS: float = Field(
        default=5.0,
        gt=0,
    )

    ROBOT_OFFLINE_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        gt=0,
    )

    ROBOT_COMMAND_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        gt=0,
    )

    ROBOT_MAX_CONCURRENT_TASKS: PositiveInt = 5

    ROBOT_REQUIRE_AUTHORIZATION: bool = True

    ROBOT_REQUIRE_SAFETY_VALIDATION: bool = True

    ROBOT_REQUIRE_HUMAN_APPROVAL: bool = True

    # --------------------------------------------------------
    # ROS 2
    # --------------------------------------------------------

    ROS2_ENABLED: bool = False

    ROS_DOMAIN_ID: int = Field(
        default=0,
        ge=0,
        le=232,
    )

    ROS_NAMESPACE: str = "/physical_ai"

    ROS_BRIDGE_ENABLED: bool = False

    ROS_BRIDGE_HOST: str = "localhost"

    ROS_BRIDGE_PORT: int = Field(
        default=9090,
        ge=1,
        le=65535,
    )

    # --------------------------------------------------------
    # Sensors / Telemetry
    # --------------------------------------------------------

    TELEMETRY_ENABLED: bool = True

    TELEMETRY_BATCH_SIZE: PositiveInt = 100

    TELEMETRY_FLUSH_INTERVAL_SECONDS: float = Field(
        default=2.0,
        gt=0,
    )

    TELEMETRY_RETENTION_DAYS: PositiveInt = 90

    SENSOR_HEALTH_MONITORING_ENABLED: bool = True

    SENSOR_ANOMALY_DETECTION_ENABLED: bool = True

    CAMERA_STREAMING_ENABLED: bool = False

    CAMERA_MAX_FPS: PositiveInt = 30

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    NAVIGATION_ENABLED: bool = True

    NAVIGATION_PLANNING_ENABLED: bool = True

    NAVIGATION_EXECUTION_ENABLED: bool = False

    NAVIGATION_OBSTACLE_AVOIDANCE_ENABLED: bool = True

    NAVIGATION_MAX_SPEED_MPS: float = Field(
        default=0.5,
        gt=0,
    )

    NAVIGATION_GOAL_TOLERANCE_METERS: float = Field(
        default=0.15,
        gt=0,
    )

    # --------------------------------------------------------
    # Manipulation
    # --------------------------------------------------------

    MANIPULATION_ENABLED: bool = True

    MANIPULATION_PLANNING_ENABLED: bool = True

    MANIPULATION_EXECUTION_ENABLED: bool = False

    MANIPULATION_FORCE_LIMIT_NEWTONS: float = Field(
        default=20.0,
        gt=0,
    )

    MANIPULATION_REQUIRE_COLLISION_CHECK: bool = True

    # --------------------------------------------------------
    # VLA
    # --------------------------------------------------------

    VLA_ENABLED: bool = True

    VLA_MODEL_NAME: str = "configured-vla-model"

    VLA_INFERENCE_ENABLED: bool = True

    VLA_ACTION_EXECUTION_ENABLED: bool = False

    VLA_MIN_CONFIDENCE: float = Field(
        default=0.9,
        ge=0,
        le=1,
    )

    VLA_REQUIRE_HUMAN_REVIEW: bool = True

    # --------------------------------------------------------
    # World Models
    # --------------------------------------------------------

    WORLD_MODEL_ENABLED: bool = True

    WORLD_MODEL_PREDICTION_ENABLED: bool = True

    WORLD_MODEL_TRAINING_ENABLED: bool = False

    WORLD_MODEL_PREDICTION_HORIZON: PositiveInt = 20

    WORLD_MODEL_UNCERTAINTY_THRESHOLD: float = Field(
        default=0.3,
        ge=0,
        le=1,
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    SIMULATION_ENABLED: bool = True

    SIMULATION_ENGINE: SimulationEngine = (
        SimulationEngine.MOCK
    )

    SIMULATION_HEADLESS: bool = True

    SIMULATION_MAX_DURATION_SECONDS: PositiveInt = 3600

    SIMULATION_MAX_CONCURRENT_RUNS: PositiveInt = 3

    SIMULATION_RECORD_RESULTS: bool = True

    # --------------------------------------------------------
    # AI / LLM
    # --------------------------------------------------------

    AI_ENABLED: bool = True

    LLM_PROVIDER: LLMProvider = LLMProvider.LOCAL

    LLM_MODEL_NAME: str = "configured-model"

    LLM_API_BASE_URL: str | None = None

    LLM_API_KEY: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None

    ANTHROPIC_API_KEY: SecretStr | None = None

    GOOGLE_API_KEY: SecretStr | None = None

    HUGGINGFACE_TOKEN: SecretStr | None = None

    OLLAMA_BASE_URL: str = (
        "http://localhost:11434"
    )

    LLM_TEMPERATURE: float = Field(
        default=0.2,
        ge=0,
        le=2,
    )

    LLM_MAX_OUTPUT_TOKENS: PositiveInt = 2048

    LLM_TIMEOUT_SECONDS: float = Field(
        default=60.0,
        gt=0,
    )

    LLM_MAX_RETRIES: int = Field(
        default=2,
        ge=0,
        le=10,
    )

    # --------------------------------------------------------
    # Agentic AI
    # --------------------------------------------------------

    AGENTS_ENABLED: bool = True

    AGENT_MAX_ITERATIONS: PositiveInt = 10

    AGENT_MAX_TOOL_CALLS: PositiveInt = 20

    AGENT_TIMEOUT_SECONDS: float = Field(
        default=120.0,
        gt=0,
    )

    AGENT_HUMAN_APPROVAL_ENABLED: bool = True

    AGENT_PHYSICAL_ACTUATION_ENABLED: bool = False

    AGENT_TOOL_ALLOWLIST_ENABLED: bool = True

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    RAG_ENABLED: bool = True

    RAG_VECTOR_DATABASE: VectorDatabase = (
        VectorDatabase.QDRANT
    )

    RAG_EMBEDDING_MODEL: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    RAG_EMBEDDING_DIMENSION: PositiveInt = 384

    RAG_CHUNK_SIZE: PositiveInt = 800

    RAG_CHUNK_OVERLAP: int = Field(
        default=120,
        ge=0,
    )

    RAG_TOP_K: PositiveInt = 8

    RAG_RERANK_ENABLED: bool = True

    RAG_HYBRID_SEARCH_ENABLED: bool = True

    RAG_CITATIONS_REQUIRED: bool = True

    RAG_GROUNDEDNESS_CHECK_ENABLED: bool = True

    QDRANT_URL: str = "http://localhost:6333"

    QDRANT_API_KEY: SecretStr | None = None

    QDRANT_COLLECTION_PREFIX: str = (
        "physical_ai"
    )

    # --------------------------------------------------------
    # Machine Learning / MLOps
    # --------------------------------------------------------

    ML_ENABLED: bool = True

    MODEL_REGISTRY_ENABLED: bool = True

    MODEL_REGISTRY_DIR: Path = MODEL_DIR

    MLFLOW_ENABLED: bool = False

    MLFLOW_TRACKING_URI: str = (
        "http://localhost:5000"
    )

    MODEL_EVALUATION_ENABLED: bool = True

    MODEL_DRIFT_MONITORING_ENABLED: bool = True

    MODEL_AUTOMATIC_DEPLOYMENT_ENABLED: bool = False

    # --------------------------------------------------------
    # Predictive Maintenance
    # --------------------------------------------------------

    PREDICTIVE_MAINTENANCE_ENABLED: bool = True

    RUL_PREDICTION_ENABLED: bool = True

    ANOMALY_DETECTION_ENABLED: bool = True

    VIBRATION_ANALYSIS_ENABLED: bool = True

    THERMAL_MONITORING_ENABLED: bool = True

    SENSOR_HEALTH_PREDICTION_ENABLED: bool = True

    MAINTENANCE_RISK_THRESHOLD: float = Field(
        default=0.75,
        ge=0,
        le=1,
    )

    RUL_WARNING_THRESHOLD_HOURS: float = Field(
        default=168.0,
        gt=0,
    )

    # --------------------------------------------------------
    # Repair / Replacement Intelligence
    # --------------------------------------------------------

    REPAIR_INTELLIGENCE_ENABLED: bool = True

    REPAIR_COST_ESTIMATION_ENABLED: bool = True

    REPAIR_REPLACE_RECOMMENDATION_ENABLED: bool = True

    PART_IMAGE_INSPECTION_ENABLED: bool = True

    PART_PRICE_LOOKUP_ENABLED: bool = False

    DEFAULT_CURRENCY: str = "INR"

    REPAIR_COST_CONFIDENCE_THRESHOLD: float = Field(
        default=0.75,
        ge=0,
        le=1,
    )

    # --------------------------------------------------------
    # Safety / Emergency Response
    # --------------------------------------------------------

    SAFETY_MONITORING_ENABLED: bool = True

    SAFETY_POLICY_ENFORCEMENT_ENABLED: bool = True

    SAFETY_HUMAN_REVIEW_REQUIRED: bool = True

    SAFETY_INCIDENT_LOGGING_ENABLED: bool = True

    SAFETY_ALERTS_ENABLED: bool = True

    SAFETY_EMERGENCY_STOP_MONITORING_ENABLED: bool = True

    SAFETY_REMOTE_RESET_ENABLED: bool = False

    SAFETY_AUTONOMOUS_RECOVERY_ENABLED: bool = False

    SAFETY_FAIL_CLOSED: bool = True

    SAFETY_MAX_TELEMETRY_AGE_SECONDS: float = Field(
        default=2.0,
        gt=0,
    )

    SAFETY_MAX_COMMAND_AGE_SECONDS: float = Field(
        default=1.0,
        gt=0,
    )

    # --------------------------------------------------------
    # Files / Uploads
    # --------------------------------------------------------

    FILE_UPLOAD_ENABLED: bool = True

    MAX_UPLOAD_SIZE_BYTES: PositiveInt = (
        25 * 1024 * 1024
    )

    MAX_FILES_PER_REQUEST: PositiveInt = 10

    UPLOAD_DIRECTORY: Path = UPLOAD_DIR

    ALLOWED_UPLOAD_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".txt",
            ".csv",
            ".json",
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".tiff",
            ".docx",
            ".xlsx",
        ]
    )

    FILE_MALWARE_SCAN_ENABLED: bool = False

    FILE_CONTENT_VALIDATION_ENABLED: bool = True

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    STORAGE_BACKEND: ObjectStorageBackend = (
        ObjectStorageBackend.LOCAL
    )

    AWS_REGION: str = "ap-south-1"

    AWS_ACCESS_KEY_ID: SecretStr | None = None

    AWS_SECRET_ACCESS_KEY: SecretStr | None = None

    S3_BUCKET_NAME: str | None = None

    S3_ENDPOINT_URL: str | None = None

    STORAGE_ENCRYPTION_ENABLED: bool = True

    # --------------------------------------------------------
    # Notifications
    # --------------------------------------------------------

    NOTIFICATIONS_ENABLED: bool = True

    EMAIL_NOTIFICATIONS_ENABLED: bool = False

    SMS_NOTIFICATIONS_ENABLED: bool = False

    WEBHOOK_NOTIFICATIONS_ENABLED: bool = False

    SMTP_HOST: str | None = None

    SMTP_PORT: int = Field(
        default=587,
        ge=1,
        le=65535,
    )

    SMTP_USERNAME: str | None = None

    SMTP_PASSWORD: SecretStr | None = None

    NOTIFICATION_WEBHOOK_URL: str | None = None

    # --------------------------------------------------------
    # Background Tasks
    # --------------------------------------------------------

    BACKGROUND_TASKS_ENABLED: bool = True

    BACKGROUND_MAX_CONCURRENT_TASKS: PositiveInt = 20

    BACKGROUND_TASK_TIMEOUT_SECONDS: float = Field(
        default=300.0,
        gt=0,
    )

    BACKGROUND_TASK_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=20,
    )

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    TESTING: bool = False

    USE_MOCK_ROBOTS: bool = True

    USE_MOCK_SENSORS: bool = True

    USE_MOCK_AI_MODELS: bool = True

    # ========================================================
    # Field Validators
    # ========================================================

    @field_validator(
        "CORS_ORIGINS",
        "CORS_METHODS",
        "CORS_HEADERS",
        "TRUSTED_HOSTS",
        "ALLOWED_UPLOAD_EXTENSIONS",
        mode="before",
    )
    @classmethod
    def validate_string_lists(
        cls,
        value: Any,
    ) -> list[str]:
        return parse_string_list(value)

    @field_validator("DEFAULT_CURRENCY")
    @classmethod
    def validate_currency(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip().upper()

        if (
            len(normalized) != 3
            or not normalized.isalpha()
        ):
            raise ValueError(
                "Currency must be a three-letter ISO code"
            )

        return normalized

    @field_validator("API_V1_PREFIX")
    @classmethod
    def validate_api_prefix(
        cls,
        value: str,
    ) -> str:
        if not value.startswith("/"):
            raise ValueError(
                "API prefix must start with '/'"
            )

        return value.rstrip("/")

    # ========================================================
    # Cross-Field Validation
    # ========================================================

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        if self.RAG_CHUNK_OVERLAP >= self.RAG_CHUNK_SIZE:
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be smaller "
                "than RAG_CHUNK_SIZE"
            )

        if (
            self.ROBOT_OFFLINE_TIMEOUT_SECONDS
            <= self.ROBOT_HEARTBEAT_INTERVAL_SECONDS
        ):
            raise ValueError(
                "Robot offline timeout must exceed "
                "heartbeat interval"
            )

        if (
            self.ROBOT_HARDWARE_CONTROL_ENABLED
            and self.ROBOT_SIMULATION_ONLY
        ):
            raise ValueError(
                "Hardware control cannot be enabled "
                "while simulation-only mode is active"
            )

        if self.ROBOT_HARDWARE_CONTROL_ENABLED:
            if not all([
                self.ROBOT_COMMANDS_ENABLED,
                self.ROBOT_REQUIRE_AUTHORIZATION,
                self.ROBOT_REQUIRE_SAFETY_VALIDATION,
                self.ROBOT_REQUIRE_HUMAN_APPROVAL,
                self.SAFETY_POLICY_ENFORCEMENT_ENABLED,
                self.SAFETY_FAIL_CLOSED,
            ]):
                raise ValueError(
                    "Hardware control requires authorization, "
                    "safety validation, human approval and "
                    "fail-closed safety policies"
                )

        if self.AGENT_PHYSICAL_ACTUATION_ENABLED:
            raise ValueError(
                "Direct AI agent physical actuation is "
                "not supported by core configuration"
            )

        if self.VLA_ACTION_EXECUTION_ENABLED:
            raise ValueError(
                "Direct VLA action execution requires a "
                "separate safety-certified control gateway"
            )

        if self.ENVIRONMENT == Environment.PRODUCTION:
            if self.DEBUG:
                raise ValueError(
                    "DEBUG must be disabled in production"
                )

            if self.RELOAD:
                raise ValueError(
                    "RELOAD must be disabled in production"
                )

            if self.JWT_SECRET_KEY is None:
                raise ValueError(
                    "JWT_SECRET_KEY is required "
                    "in production"
                )

            secret = (
                self.JWT_SECRET_KEY.get_secret_value()
            )

            if len(secret) < 32:
                raise ValueError(
                    "Production JWT secret must contain "
                    "at least 32 characters"
                )

            if "*" in self.TRUSTED_HOSTS:
                raise ValueError(
                    "Wildcard trusted hosts are not "
                    "allowed in production"
                )

            if "*" in self.CORS_ORIGINS:
                raise ValueError(
                    "Wildcard CORS origins are not "
                    "allowed in production"
                )

            if not self.ENABLE_TENANT_ISOLATION:
                raise ValueError(
                    "Tenant isolation is required "
                    "in production"
                )

            if not self.ENABLE_RBAC:
                raise ValueError(
                    "RBAC is required in production"
                )

            if not self.SAFETY_FAIL_CLOSED:
                raise ValueError(
                    "Fail-closed safety enforcement "
                    "is required in production"
                )

            if (
                self.RATE_LIMIT_ENABLED
                and self.RATE_LIMIT_BACKEND
                != RateLimitBackend.REDIS
            ):
                raise ValueError(
                    "Distributed Redis rate limiting "
                    "is required in production"
                )

            if self.DATABASE_URL.startswith(
                "postgresql+asyncpg://postgres:postgres@"
            ):
                raise ValueError(
                    "Default database credentials "
                    "cannot be used in production"
                )

        return self

    # ========================================================
    # Convenience Properties
    # ========================================================

    @property
    def is_development(self) -> bool:
        return (
            self.ENVIRONMENT
            == Environment.DEVELOPMENT
        )

    @property
    def is_testing(self) -> bool:
        return (
            self.TESTING
            or self.ENVIRONMENT
            == Environment.TESTING
        )

    @property
    def is_production(self) -> bool:
        return (
            self.ENVIRONMENT
            == Environment.PRODUCTION
        )

    @property
    def database_url(self) -> str:
        return self.DATABASE_URL

    @property
    def redis_url(self) -> str:
        return self.REDIS_URL

    @property
    def jwt_secret(self) -> str:
        if self.JWT_SECRET_KEY is None:
            raise RuntimeError(
                "JWT_SECRET_KEY is not configured"
            )

        return self.JWT_SECRET_KEY.get_secret_value()

    @property
    def robot_control_available(self) -> bool:
        """
        Configuration eligibility only.

        Does not establish that robot control is safe
        or authorized at runtime.
        """

        return all([
            self.ROBOT_COMMANDS_ENABLED,
            self.ROBOT_HARDWARE_CONTROL_ENABLED,
            not self.ROBOT_SIMULATION_ONLY,
            self.ROBOT_REQUIRE_AUTHORIZATION,
            self.ROBOT_REQUIRE_SAFETY_VALIDATION,
            self.SAFETY_POLICY_ENFORCEMENT_ENABLED,
            self.SAFETY_FAIL_CLOSED,
        ])

    def get_public_config(self) -> dict[str, Any]:
        """
        Safe subset for status endpoints.

        Does not expose credentials or internal URLs.
        """

        return {
            "app_name": self.APP_NAME,
            "version": self.APP_VERSION,
            "environment": self.ENVIRONMENT.value,
            "api_prefix": self.API_V1_PREFIX,
            "features": {
                "robotics": self.ROBOTICS_ENABLED,
                "simulation": self.SIMULATION_ENABLED,
                "navigation": self.NAVIGATION_ENABLED,
                "manipulation": self.MANIPULATION_ENABLED,
                "vla": self.VLA_ENABLED,
                "world_model": self.WORLD_MODEL_ENABLED,
                "rag": self.RAG_ENABLED,
                "agents": self.AGENTS_ENABLED,
                "predictive_maintenance": (
                    self.PREDICTIVE_MAINTENANCE_ENABLED
                ),
                "repair_intelligence": (
                    self.REPAIR_INTELLIGENCE_ENABLED
                ),
                "safety_monitoring": (
                    self.SAFETY_MONITORING_ENABLED
                ),
            },
        }

    def ensure_local_directories(self) -> None:
        """
        Create local runtime directories.

        Call explicitly during startup.
        """

        for directory in (
            DATA_DIR,
            LOG_DIR,
            self.UPLOAD_DIRECTORY,
            self.MODEL_REGISTRY_DIR,
            ARTIFACT_DIR,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================
# Settings Factory
# ============================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached application settings.

    Example:
        settings = get_settings()
        print(settings.APP_NAME)
    """

    return Settings()


def reload_settings() -> Settings:
    """
    Clear the settings cache and reload environment values.

    Intended for tests and controlled configuration reloads.
    """

    get_settings.cache_clear()
    return get_settings()


# ============================================================
# Module-Level Lazy Access
# ============================================================


def __getattr__(name: str) -> Any:
    """
    Supports:
        from app.core.settings import settings

    without constructing Settings during module import.
    """

    if name == "settings":
        return get_settings()

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "Settings",
    "Environment",
    "LogFormat",
    "RateLimitBackend",
    "CacheBackend",
    "VectorDatabase",
    "ObjectStorageBackend",
    "SimulationEngine",
    "LLMProvider",
    "PROJECT_ROOT",
    "BACKEND_ROOT",
    "DATA_DIR",
    "LOG_DIR",
    "UPLOAD_DIR",
    "MODEL_DIR",
    "ARTIFACT_DIR",
    "ENV_FILE",
    "get_settings",
    "reload_settings",
    "settings",
]