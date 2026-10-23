"""Config module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Application Configuration
=========================

Physical AI & Autonomous Robotics Intelligence Platform

Centralized configuration for:
    - Application and API
    - Environment and deployment
    - PostgreSQL
    - Redis
    - Authentication and security
    - CORS
    - Robot telemetry
    - ROS 2 integration
    - MQTT and Kafka
    - AI / ML / VLM models
    - Agentic AI and RAG
    - Navigation and simulation
    - Robot health and maintenance
    - Safety analytics
    - Cost and performance analytics
    - File uploads
    - Cloud storage
    - Monitoring and logging

Environment variables can be loaded from .env.

Sensitive credentials must be supplied through
environment variables or a secrets manager.

Safety-related settings below are application policy
parameters, NOT certified hardware safety controls.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


BASE_DIR = Path(__file__).resolve().parents[2]


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Global application settings.

    Environment variable examples:
        APP_NAME=Physical AI Platform
        ENVIRONMENT=development
        DATABASE_URL=postgresql+asyncpg://...
        REDIS_URL=redis://localhost:6379/0
        JWT_SECRET_KEY=...
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # --------------------------------------------------
    # APPLICATION
    # --------------------------------------------------

    APP_NAME: str = (
        "Physical AI & Autonomous Robotics Intelligence"
    )

    APP_VERSION: str = "1.0.0"

    APP_DESCRIPTION: str = (
        "Autonomous robotics intelligence, monitoring, "
        "perception, planning, maintenance and analytics."
    )

    ENVIRONMENT: Environment = Environment.DEVELOPMENT

    DEBUG: bool = False

    API_V1_PREFIX: str = "/api/v1"

    HOST: str = "127.0.0.1"

    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    WORKERS: int = Field(
        default=1,
        ge=1,
        le=64,
    )

    PUBLIC_BASE_URL: str = "http://localhost:8000"

    FRONTEND_URL: str = "http://localhost:3000"

    TIMEZONE: str = "UTC"

    # --------------------------------------------------
    # CORS
    # --------------------------------------------------

    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
        ]
    )

    CORS_ALLOW_CREDENTIALS: bool = False

    CORS_ALLOW_METHODS: list[str] = Field(
        default_factory=lambda: [
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        ]
    )

    CORS_ALLOW_HEADERS: list[str] = Field(
        default_factory=lambda: [
            "Authorization",
            "Content-Type",
            "X-Request-ID",
        ]
    )

    # --------------------------------------------------
    # DATABASE
    # --------------------------------------------------

    DATABASE_URL: SecretStr = SecretStr(
        "postgresql+asyncpg://robotics:robotics"
        "@localhost:5432/robotics_ai"
    )

    DB_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    DB_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        le=100,
    )

    DB_POOL_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
    )

    DB_POOL_RECYCLE_SECONDS: int = Field(
        default=1800,
        ge=30,
    )

    DB_ECHO: bool = False

    # --------------------------------------------------
    # REDIS
    # --------------------------------------------------

    REDIS_ENABLED: bool = False

    REDIS_URL: SecretStr = SecretStr(
        "redis://localhost:6379/0"
    )

    REDIS_CACHE_TTL_SECONDS: int = Field(
        default=300,
        ge=1,
    )

    REDIS_SOCKET_TIMEOUT_SECONDS: int = Field(
        default=5,
        ge=1,
    )

    # --------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------

    JWT_SECRET_KEY: SecretStr = SecretStr(
        "development-only-change-this-secret"
    )

    JWT_ALGORITHM: Literal["HS256", "HS384", "HS512"] = (
        "HS256"
    )

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
    )

    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
    )

    PASSWORD_MIN_LENGTH: int = Field(
        default=12,
        ge=8,
        le=128,
    )

    AUTH_RATE_LIMIT_PER_MINUTE: int = Field(
        default=10,
        ge=1,
    )

    API_RATE_LIMIT_PER_MINUTE: int = Field(
        default=120,
        ge=1,
    )

    ENABLE_RBAC: bool = True

    ENABLE_AUDIT_LOGS: bool = True

    # --------------------------------------------------
    # ROBOT PLATFORM
    # --------------------------------------------------

    ROBOT_PLATFORM_ENABLED: bool = True

    MAX_REGISTERED_ROBOTS: int = Field(
        default=1000,
        ge=1,
    )

    ROBOT_HEARTBEAT_INTERVAL_SECONDS: float = Field(
        default=5.0,
        gt=0,
    )

    ROBOT_HEARTBEAT_TIMEOUT_SECONDS: float = Field(
        default=15.0,
        gt=0,
    )

    ROBOT_TELEMETRY_INTERVAL_SECONDS: float = Field(
        default=1.0,
        gt=0,
    )

    ROBOT_OFFLINE_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        gt=0,
    )

    ROBOT_DEFAULT_FRAME_ID: str = "base_link"

    ROBOT_DEFAULT_MAP_FRAME: str = "map"

    ROBOT_DEFAULT_ODOM_FRAME: str = "odom"

    # --------------------------------------------------
    # ROBOT COMMAND AUTHORIZATION
    # --------------------------------------------------

    ROBOT_COMMANDS_ENABLED: bool = False

    REQUIRE_ROBOT_OPERATOR_APPROVAL: bool = True

    REQUIRE_COMMAND_AUDIT: bool = True

    REQUIRE_COMMAND_EXPIRATION: bool = True

    ROBOT_COMMAND_TTL_SECONDS: int = Field(
        default=10,
        ge=1,
        le=300,
    )

    REQUIRE_SIMULATION_VALIDATION: bool = True

    # --------------------------------------------------
    # ROS 2
    # --------------------------------------------------

    ROS2_ENABLED: bool = False

    ROS_DOMAIN_ID: int = Field(
        default=0,
        ge=0,
        le=232,
    )

    ROS_NAMESPACE: str = "/robotics"

    ROS_USE_SIM_TIME: bool = False

    ROS_DISCOVERY_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        gt=0,
    )

    # --------------------------------------------------
    # MQTT
    # --------------------------------------------------

    MQTT_ENABLED: bool = False

    MQTT_HOST: str = "localhost"

    MQTT_PORT: int = Field(
        default=1883,
        ge=1,
        le=65535,
    )

    MQTT_USERNAME: str | None = None

    MQTT_PASSWORD: SecretStr | None = None

    MQTT_USE_TLS: bool = False

    MQTT_TOPIC_PREFIX: str = "robotics"

    MQTT_KEEPALIVE_SECONDS: int = Field(
        default=60,
        ge=5,
    )

    # --------------------------------------------------
    # KAFKA
    # --------------------------------------------------

    KAFKA_ENABLED: bool = False

    KAFKA_BOOTSTRAP_SERVERS: str = (
        "localhost:9092"
    )

    KAFKA_TELEMETRY_TOPIC: str = (
        "robotics.telemetry"
    )

    KAFKA_EVENTS_TOPIC: str = (
        "robotics.events"
    )

    KAFKA_SAFETY_TOPIC: str = (
        "robotics.safety"
    )

    KAFKA_CONSUMER_GROUP: str = (
        "robotics-platform"
    )

    # --------------------------------------------------
    # AI MODEL PROVIDERS
    # --------------------------------------------------

    AI_ENABLED: bool = True

    DEFAULT_LLM_PROVIDER: Literal[
        "openai",
        "anthropic",
        "google",
        "huggingface",
        "local",
    ] = "local"

    DEFAULT_LLM_MODEL: str = "local-model"

    OPENAI_API_KEY: SecretStr | None = None

    ANTHROPIC_API_KEY: SecretStr | None = None

    GOOGLE_API_KEY: SecretStr | None = None

    HUGGINGFACE_TOKEN: SecretStr | None = None

    LOCAL_LLM_BASE_URL: str = (
        "http://localhost:11434"
    )

    LLM_TEMPERATURE: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
    )

    LLM_MAX_OUTPUT_TOKENS: int = Field(
        default=4096,
        ge=1,
    )

    LLM_TIMEOUT_SECONDS: int = Field(
        default=120,
        ge=1,
    )

    LLM_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
    )

    # --------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------

    ML_ENABLED: bool = True

    ML_MODEL_DIR: Path = (
        BASE_DIR / "models" / "ml"
    )

    ML_DEVICE: Literal[
        "auto",
        "cpu",
        "cuda",
        "mps",
    ] = "auto"

    ML_BATCH_SIZE: int = Field(
        default=16,
        ge=1,
    )

    ML_INFERENCE_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        gt=0,
    )

    MODEL_CACHE_ENABLED: bool = True

    # --------------------------------------------------
    # COMPUTER VISION
    # --------------------------------------------------

    CV_ENABLED: bool = True

    CV_MODEL_DIR: Path = (
        BASE_DIR / "models" / "cv"
    )

    CV_CONFIDENCE_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    CV_IOU_THRESHOLD: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
    )

    CV_MAX_IMAGE_SIZE_MB: int = Field(
        default=20,
        ge=1,
    )

    CV_MAX_VIDEO_SIZE_MB: int = Field(
        default=500,
        ge=1,
    )

    CV_ENABLE_OBJECT_DETECTION: bool = True

    CV_ENABLE_SEGMENTATION: bool = True

    CV_ENABLE_DEFECT_DETECTION: bool = True

    CV_ENABLE_PART_RECOGNITION: bool = True

    CV_ENABLE_HUMAN_DETECTION: bool = True

    # --------------------------------------------------
    # VISION LANGUAGE MODELS
    # --------------------------------------------------

    VLM_ENABLED: bool = True

    VLM_PROVIDER: str = "local"

    VLM_MODEL: str = "local-vision-model"

    VLM_MAX_IMAGES_PER_REQUEST: int = Field(
        default=10,
        ge=1,
    )

    VLM_TIMEOUT_SECONDS: int = Field(
        default=90,
        ge=1,
    )

    # --------------------------------------------------
    # AGENTIC AI
    # --------------------------------------------------

    AGENTS_ENABLED: bool = True

    AGENT_MAX_STEPS: int = Field(
        default=20,
        ge=1,
        le=200,
    )

    AGENT_TIMEOUT_SECONDS: int = Field(
        default=180,
        ge=1,
    )

    AGENT_MAX_PARALLEL_TASKS: int = Field(
        default=5,
        ge=1,
    )

    AGENT_REQUIRE_TOOL_APPROVAL: bool = True

    AGENT_ENABLE_MEMORY: bool = True

    AGENT_ENABLE_PLANNING: bool = True

    AGENT_ENABLE_VERIFICATION: bool = True

    AGENT_ALLOW_DIRECT_HARDWARE_CONTROL: bool = False

    # --------------------------------------------------
    # RAG
    # --------------------------------------------------

    RAG_ENABLED: bool = True

    RAG_EMBEDDING_MODEL: str = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    RAG_VECTOR_STORE: Literal[
        "faiss",
        "qdrant",
        "chroma",
        "pgvector",
    ] = "faiss"

    RAG_TOP_K: int = Field(
        default=8,
        ge=1,
        le=100,
    )

    RAG_CHUNK_SIZE: int = Field(
        default=1000,
        ge=100,
    )

    RAG_CHUNK_OVERLAP: int = Field(
        default=150,
        ge=0,
    )

    RAG_ENABLE_HYBRID_SEARCH: bool = True

    RAG_ENABLE_RERANKING: bool = True

    RAG_REQUIRE_CITATIONS: bool = True

    RAG_ENABLE_QUERY_REWRITING: bool = True

    # --------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------

    NAVIGATION_ENABLED: bool = True

    NAVIGATION_PLANNER: str = "astar"

    NAVIGATION_REPLAN_INTERVAL_SECONDS: float = Field(
        default=2.0,
        gt=0,
    )

    NAVIGATION_GOAL_TOLERANCE_M: float = Field(
        default=0.1,
        gt=0,
    )

    NAVIGATION_OBSTACLE_INFLATION_M: float = Field(
        default=0.3,
        ge=0,
    )

    # --------------------------------------------------
    # ROBOT SIMULATION
    # --------------------------------------------------

    SIMULATION_ENABLED: bool = True

    SIMULATION_BACKEND: Literal[
        "gazebo",
        "isaac_sim",
        "pybullet",
        "mujoco",
        "custom",
    ] = "gazebo"

    SIMULATION_HEADLESS: bool = True

    SIMULATION_REAL_TIME_FACTOR: float = Field(
        default=1.0,
        gt=0,
    )

    SIMULATION_MAX_EPISODE_SECONDS: int = Field(
        default=3600,
        ge=1,
    )

    # --------------------------------------------------
    # ROBOT HEALTH
    # --------------------------------------------------

    ROBOT_HEALTH_ENABLED: bool = True

    HEALTH_MONITOR_INTERVAL_SECONDS: int = Field(
        default=10,
        ge=1,
    )

    HEALTH_WARNING_SCORE: float = Field(
        default=70.0,
        ge=0,
        le=100,
    )

    HEALTH_CRITICAL_SCORE: float = Field(
        default=40.0,
        ge=0,
        le=100,
    )

    ENABLE_ANOMALY_DETECTION: bool = True

    ENABLE_PREDICTIVE_MAINTENANCE: bool = True

    ENABLE_RUL_PREDICTION: bool = True

    ENABLE_COMPONENT_HEALTH_MONITORING: bool = True

    # --------------------------------------------------
    # MAINTENANCE
    # --------------------------------------------------

    MAINTENANCE_ENABLED: bool = True

    MAINTENANCE_OVERDUE_ALERTS: bool = True

    MAINTENANCE_COST_TRACKING: bool = True

    MAINTENANCE_REPAIR_REPLACE_ANALYSIS: bool = True

    MAINTENANCE_DEFAULT_LABOR_RATE: float = Field(
        default=500.0,
        ge=0,
    )

    # --------------------------------------------------
    # ROBOT SAFETY POLICY
    # --------------------------------------------------

    SAFETY_ANALYTICS_ENABLED: bool = True

    SAFETY_INCIDENT_RATE_WARNING: float = Field(
        default=5.0,
        ge=0,
    )

    SAFETY_INCIDENT_RATE_CRITICAL: float = Field(
        default=15.0,
        ge=0,
    )

    SAFETY_INCIDENT_RATE_BASE_HOURS: float = Field(
        default=1000.0,
        gt=0,
    )

    SAFETY_REQUIRE_HUMAN_REVIEW: bool = True

    SAFETY_LOG_ALL_INCIDENTS: bool = True

    SAFETY_ENABLE_RISK_ASSESSMENTS: bool = True

    # --------------------------------------------------
    # UTILIZATION ANALYTICS
    # --------------------------------------------------

    UTILIZATION_ENABLED: bool = True

    UTILIZATION_LOW_THRESHOLD: float = Field(
        default=40.0,
        ge=0,
        le=100,
    )

    UTILIZATION_HIGH_THRESHOLD: float = Field(
        default=90.0,
        ge=0,
        le=100,
    )

    # --------------------------------------------------
    # TASK PERFORMANCE ANALYTICS
    # --------------------------------------------------

    TASK_ANALYTICS_ENABLED: bool = True

    TASK_SUCCESS_WARNING_PERCENTAGE: float = Field(
        default=85.0,
        ge=0,
        le=100,
    )

    TASK_RETRY_WARNING_PERCENTAGE: float = Field(
        default=30.0,
        ge=0,
        le=100,
    )

    TASK_SLA_WARNING_PERCENTAGE: float = Field(
        default=90.0,
        ge=0,
        le=100,
    )

    PERFORMANCE_LATENCY_WARNING_MS: float = Field(
        default=200.0,
        ge=0,
    )

    # --------------------------------------------------
    # COST ANALYTICS
    # --------------------------------------------------

    COST_ANALYTICS_ENABLED: bool = True

    DEFAULT_CURRENCY: str = "INR"

    ENERGY_PRICE_PER_KWH: float = Field(
        default=8.0,
        ge=0,
    )

    COST_FORECAST_MONTHS: int = Field(
        default=12,
        ge=1,
        le=120,
    )

    # --------------------------------------------------
    # REPORTING
    # --------------------------------------------------

    REPORTING_ENABLED: bool = True

    REPORT_OUTPUT_DIR: Path = (
        BASE_DIR / "reports"
    )

    REPORT_MAX_HISTORY: int = Field(
        default=100,
        ge=1,
    )

    REPORT_INCLUDE_ALERTS: bool = True

    REPORT_INCLUDE_RECOMMENDATIONS: bool = True

    REPORT_MAX_EXPORT_MB: int = Field(
        default=20,
        ge=1,
    )

    # --------------------------------------------------
    # FILE UPLOADS
    # --------------------------------------------------

    UPLOAD_DIR: Path = (
        BASE_DIR / "uploads"
    )

    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50,
        ge=1,
    )

    ALLOWED_IMAGE_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".tiff",
        ]
    )

    ALLOWED_DOCUMENT_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".docx",
            ".txt",
            ".csv",
            ".xlsx",
        ]
    )

    ENABLE_FILE_MALWARE_SCANNING: bool = True

    ENABLE_UPLOAD_CONTENT_VALIDATION: bool = True

    # --------------------------------------------------
    # STORAGE
    # --------------------------------------------------

    STORAGE_BACKEND: Literal[
        "local",
        "s3",
        "azure_blob",
        "gcs",
    ] = "local"

    AWS_REGION: str = "ap-south-1"

    AWS_S3_BUCKET: str | None = None

    AWS_ACCESS_KEY_ID: SecretStr | None = None

    AWS_SECRET_ACCESS_KEY: SecretStr | None = None

    AZURE_STORAGE_CONNECTION_STRING: SecretStr | None = None

    GCS_BUCKET: str | None = None

    # --------------------------------------------------
    # MONITORING
    # --------------------------------------------------

    METRICS_ENABLED: bool = True

    PROMETHEUS_ENABLED: bool = True

    TRACING_ENABLED: bool = False

    OTEL_SERVICE_NAME: str = (
        "physical-ai-robotics-platform"
    )

    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None

    LOG_LEVEL: LogLevel = LogLevel.INFO

    LOG_FORMAT: Literal[
        "json",
        "text",
    ] = "json"

    LOG_DIR: Path = (
        BASE_DIR / "logs"
    )

    LOG_REQUEST_BODIES: bool = False

    LOG_SENSITIVE_DATA: bool = False

    # --------------------------------------------------
    # BACKGROUND WORKERS
    # --------------------------------------------------

    BACKGROUND_WORKERS_ENABLED: bool = True

    MAX_BACKGROUND_TASKS: int = Field(
        default=20,
        ge=1,
    )

    BACKGROUND_TASK_TIMEOUT_SECONDS: int = Field(
        default=300,
        ge=1,
    )

    # --------------------------------------------------
    # VALIDATORS
    # --------------------------------------------------

    @field_validator(
        "API_V1_PREFIX",
        mode="after",
    )
    @classmethod
    def validate_api_prefix(
        cls,
        value: str,
    ) -> str:
        if not value.startswith("/"):
            raise ValueError(
                "API_V1_PREFIX must start with /"
            )

        return value.rstrip("/") or "/"

    @field_validator(
        "DEFAULT_CURRENCY",
        mode="after",
    )
    @classmethod
    def validate_currency(
        cls,
        value: str,
    ) -> str:
        value = value.strip().upper()

        if len(value) != 3 or not value.isalpha():
            raise ValueError(
                "DEFAULT_CURRENCY must be a "
                "three-letter currency code"
            )

        return value

    @model_validator(mode="after")
    def validate_cross_field_settings(
        self,
    ) -> "Settings":
        if (
            self.ROBOT_HEARTBEAT_TIMEOUT_SECONDS
            <= self.ROBOT_HEARTBEAT_INTERVAL_SECONDS
        ):
            raise ValueError(
                "Robot heartbeat timeout must exceed "
                "heartbeat interval"
            )

        if (
            self.ROBOT_OFFLINE_TIMEOUT_SECONDS
            <= self.ROBOT_HEARTBEAT_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "Robot offline timeout must exceed "
                "heartbeat timeout"
            )

        if (
            self.HEALTH_CRITICAL_SCORE
            >= self.HEALTH_WARNING_SCORE
        ):
            raise ValueError(
                "Critical health score must be below "
                "warning health score"
            )

        if (
            self.SAFETY_INCIDENT_RATE_WARNING
            >= self.SAFETY_INCIDENT_RATE_CRITICAL
        ):
            raise ValueError(
                "Safety warning rate must be below "
                "critical rate"
            )

        if (
            self.UTILIZATION_LOW_THRESHOLD
            >= self.UTILIZATION_HIGH_THRESHOLD
        ):
            raise ValueError(
                "Low utilization threshold must be "
                "below high utilization threshold"
            )

        if (
            self.RAG_CHUNK_OVERLAP
            >= self.RAG_CHUNK_SIZE
        ):
            raise ValueError(
                "RAG overlap must be smaller than "
                "chunk size"
            )

        if self.ENVIRONMENT == Environment.PRODUCTION:
            if self.DEBUG:
                raise ValueError(
                    "DEBUG must be disabled in production"
                )

            secret = (
                self.JWT_SECRET_KEY.get_secret_value()
            )

            if (
                secret
                == "development-only-change-this-secret"
                or len(secret) < 32
            ):
                raise ValueError(
                    "Production requires a strong "
                    "JWT_SECRET_KEY of at least 32 characters"
                )

            database_url = (
                self.DATABASE_URL.get_secret_value()
            )

            if (
                "robotics:robotics@localhost"
                in database_url
            ):
                raise ValueError(
                    "Production cannot use the default "
                    "development database credentials"
                )

            if self.ROBOT_COMMANDS_ENABLED:
                if not all((
                    self.REQUIRE_ROBOT_OPERATOR_APPROVAL,
                    self.REQUIRE_COMMAND_AUDIT,
                    self.REQUIRE_COMMAND_EXPIRATION,
                    self.REQUIRE_SIMULATION_VALIDATION,
                )):
                    raise ValueError(
                        "Production robot commands require "
                        "approval, auditing, expiration "
                        "and simulation validation"
                    )

            if self.AGENT_ALLOW_DIRECT_HARDWARE_CONTROL:
                raise ValueError(
                    "Direct agent hardware control "
                    "is prohibited in production"
                )

            if "*" in self.CORS_ORIGINS:
                raise ValueError(
                    "Wildcard CORS is not permitted "
                    "in production"
                )

        return self

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    @property
    def is_production(self) -> bool:
        return (
            self.ENVIRONMENT
            == Environment.PRODUCTION
        )

    @property
    def is_development(self) -> bool:
        return (
            self.ENVIRONMENT
            == Environment.DEVELOPMENT
        )

    @property
    def database_dsn(self) -> str:
        """
        Return database URL for internal DB setup.

        Never include this value in API responses or logs.
        """
        return self.DATABASE_URL.get_secret_value()

    @property
    def redis_dsn(self) -> str:
        return self.REDIS_URL.get_secret_value()

    def ensure_directories(self) -> None:
        """
        Create local runtime directories.

        Call explicitly during application startup.
        """
        directories = [
            self.UPLOAD_DIR,
            self.REPORT_OUTPUT_DIR,
            self.LOG_DIR,
            self.ML_MODEL_DIR,
            self.CV_MODEL_DIR,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def public_settings(self) -> dict[str, object]:
        """
        Return a safe subset for frontend configuration.

        Do not expose internal service endpoints,
        credentials, or deployment details.
        """
        return {
            "app_name": self.APP_NAME,
            "app_version": self.APP_VERSION,
            "environment": self.ENVIRONMENT.value,
            "api_prefix": self.API_V1_PREFIX,
            "features": {
                "robotics": self.ROBOT_PLATFORM_ENABLED,
                "simulation": self.SIMULATION_ENABLED,
                "ai": self.AI_ENABLED,
                "computer_vision": self.CV_ENABLED,
                "agents": self.AGENTS_ENABLED,
                "rag": self.RAG_ENABLED,
                "analytics": self.REPORTING_ENABLED,
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


def reload_settings() -> Settings:
    """
    Clear settings cache and reload environment values.

    Intended for testing or controlled reconfiguration.
    """
    get_settings.cache_clear()
    return get_settings()


settings = get_settings()


__all__ = [
    "BASE_DIR",
    "Environment",
    "LogLevel",
    "Settings",
    "get_settings",
    "reload_settings",
    "settings",
]
