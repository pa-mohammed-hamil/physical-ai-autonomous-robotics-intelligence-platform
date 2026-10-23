"""Constants module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Application Constants

Centralized constants for:
- Application metadata
- API configuration
- Authentication and authorization
- Robotics operations
- Robot health and telemetry
- Sensors and perception
- Navigation and manipulation
- Physical AI and world models
- Simulation and evaluation
- Predictive maintenance
- Repair and replacement
- Fleet management
- AI safety and security
- Caching and infrastructure

Important:
These are software configuration defaults, not certified
hardware safety limits. Robot-specific operating limits
must come from validated hardware specifications and
independently enforced safety controllers.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


# ============================================================
# Application Metadata
# ============================================================

APP_NAME: Final[str] = (
    "Physical AI & Autonomous Robotics Intelligence Platform"
)

APP_SHORT_NAME: Final[str] = "Physical AI"

APP_SLUG: Final[str] = "physical-ai-autonomous-robotics"

APP_VERSION: Final[str] = "1.0.0"

API_VERSION: Final[str] = "v1"

API_PREFIX: Final[str] = "/api/v1"

APP_DESCRIPTION: Final[str] = (
    "Enterprise-grade Physical AI, autonomous robotics, "
    "robot intelligence, predictive maintenance, "
    "simulation, fleet monitoring, and safety analytics."
)

DEFAULT_LANGUAGE: Final[str] = "en"

DEFAULT_TIMEZONE: Final[str] = "UTC"

DEFAULT_ENCODING: Final[str] = "utf-8"

DEFAULT_CONTENT_TYPE: Final[str] = "application/json"


# ============================================================
# Environment
# ============================================================


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


DEFAULT_ENVIRONMENT: Final[str] = Environment.DEVELOPMENT.value

DEBUG_DEFAULT: Final[bool] = False

DOCS_URL: Final[str] = "/docs"

REDOC_URL: Final[str] = "/redoc"

OPENAPI_URL: Final[str] = "/openapi.json"

HEALTH_CHECK_PATH: Final[str] = "/health"

READINESS_CHECK_PATH: Final[str] = "/ready"

LIVENESS_CHECK_PATH: Final[str] = "/live"


# ============================================================
# API Configuration
# ============================================================

DEFAULT_PAGE: Final[int] = 1

DEFAULT_PAGE_SIZE: Final[int] = 20

MAX_PAGE_SIZE: Final[int] = 100

DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[int] = 30

DEFAULT_AI_REQUEST_TIMEOUT_SECONDS: Final[int] = 120

DEFAULT_UPLOAD_TIMEOUT_SECONDS: Final[int] = 300

MAX_REQUEST_BODY_BYTES: Final[int] = 10 * 1024 * 1024

MAX_UPLOAD_SIZE_BYTES: Final[int] = 100 * 1024 * 1024

MAX_BATCH_SIZE: Final[int] = 1000

DEFAULT_RETRY_ATTEMPTS: Final[int] = 3

DEFAULT_RETRY_DELAY_SECONDS: Final[float] = 1.0

MAX_RETRY_DELAY_SECONDS: Final[float] = 30.0


# ============================================================
# Authentication
# ============================================================

ACCESS_TOKEN_EXPIRE_MINUTES: Final[int] = 30

REFRESH_TOKEN_EXPIRE_DAYS: Final[int] = 7

PASSWORD_MIN_LENGTH: Final[int] = 12

PASSWORD_MAX_LENGTH: Final[int] = 128

MAX_FAILED_LOGIN_ATTEMPTS: Final[int] = 5

ACCOUNT_LOCKOUT_MINUTES: Final[int] = 15

SESSION_IDLE_TIMEOUT_MINUTES: Final[int] = 30

SESSION_ABSOLUTE_TIMEOUT_HOURS: Final[int] = 24

MFA_CODE_LENGTH: Final[int] = 6

MFA_CODE_EXPIRE_SECONDS: Final[int] = 300

PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: Final[int] = 15

EMAIL_VERIFICATION_EXPIRE_HOURS: Final[int] = 24


# ============================================================
# RBAC
# ============================================================


class SystemRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    ORGANIZATION_ADMIN = "organization_admin"
    FLEET_MANAGER = "fleet_manager"
    ROBOT_OPERATOR = "robot_operator"
    ROBOT_ENGINEER = "robot_engineer"
    MAINTENANCE_ENGINEER = "maintenance_engineer"
    SAFETY_OFFICER = "safety_officer"
    AI_ENGINEER = "ai_engineer"
    ML_ENGINEER = "ml_engineer"
    DATA_SCIENTIST = "data_scientist"
    RESEARCHER = "researcher"
    TECHNICIAN = "technician"
    AUDITOR = "auditor"
    VIEWER = "viewer"
    SERVICE_ACCOUNT = "service_account"


class Permission(str, Enum):
    ROBOT_READ = "robot:read"
    ROBOT_CREATE = "robot:create"
    ROBOT_UPDATE = "robot:update"
    ROBOT_DELETE = "robot:delete"
    ROBOT_CONTROL_REQUEST = "robot:control_request"
    ROBOT_CONTROL_APPROVE = "robot:control_approve"

    FLEET_READ = "fleet:read"
    FLEET_MANAGE = "fleet:manage"

    TELEMETRY_READ = "telemetry:read"
    TELEMETRY_INGEST = "telemetry:ingest"

    SENSOR_READ = "sensor:read"
    SENSOR_MANAGE = "sensor:manage"

    NAVIGATION_READ = "navigation:read"
    NAVIGATION_PLAN = "navigation:plan"

    MANIPULATION_READ = "manipulation:read"
    MANIPULATION_PLAN = "manipulation:plan"

    SIMULATION_READ = "simulation:read"
    SIMULATION_RUN = "simulation:run"

    MODEL_READ = "model:read"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"

    EVALUATION_READ = "evaluation:read"
    EVALUATION_RUN = "evaluation:run"

    MAINTENANCE_READ = "maintenance:read"
    MAINTENANCE_MANAGE = "maintenance:manage"

    REPAIR_READ = "repair:read"
    REPAIR_MANAGE = "repair:manage"

    PARTS_READ = "parts:read"
    PARTS_MANAGE = "parts:manage"

    SAFETY_READ = "safety:read"
    SAFETY_REVIEW = "safety:review"

    INCIDENT_READ = "incident:read"
    INCIDENT_MANAGE = "incident:manage"

    USER_READ = "user:read"
    USER_MANAGE = "user:manage"

    AUDIT_READ = "audit:read"
    SETTINGS_MANAGE = "settings:manage"


DEFAULT_USER_ROLE: Final[str] = SystemRole.VIEWER.value

DEFAULT_USER_PERMISSIONS: Final[frozenset[str]] = frozenset({
    Permission.ROBOT_READ.value,
    Permission.FLEET_READ.value,
    Permission.TELEMETRY_READ.value,
})


# ============================================================
# Robotics
# ============================================================


class RobotCategory(str, Enum):
    MOBILE = "mobile"
    HUMANOID = "humanoid"
    INDUSTRIAL_ARM = "industrial_arm"
    COLLABORATIVE = "collaborative"
    WAREHOUSE = "warehouse"
    DELIVERY = "delivery"
    SERVICE = "service"
    MEDICAL = "medical"
    AGRICULTURAL = "agricultural"
    CONSTRUCTION = "construction"
    INSPECTION = "inspection"
    SECURITY = "security"
    DRONE = "drone"
    AUTONOMOUS_VEHICLE = "autonomous_vehicle"
    UNDERWATER = "underwater"
    QUADRUPED = "quadruped"
    MANIPULATOR = "manipulator"
    RESEARCH = "research"
    SIMULATED = "simulated"
    CUSTOM = "custom"
    OTHER = "other"


class RobotOperatingMode(str, Enum):
    MANUAL = "manual"
    TELEOPERATION = "teleoperation"
    ASSISTED = "assisted"
    SEMI_AUTONOMOUS = "semi_autonomous"
    AUTONOMOUS = "autonomous"
    SUPERVISED_AUTONOMOUS = "supervised_autonomous"
    SIMULATION = "simulation"
    MAINTENANCE = "maintenance"
    SAFE_STOP = "safe_stop"


class RobotLifecycleStatus(str, Enum):
    REGISTERED = "registered"
    OFFLINE = "offline"
    ONLINE = "online"
    IDLE = "idle"
    READY = "ready"
    INITIALIZING = "initializing"
    CALIBRATING = "calibrating"
    WORKING = "working"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    CHARGING = "charging"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    FAULT = "fault"
    EMERGENCY_STOPPED = "emergency_stopped"
    SHUTDOWN = "shutdown"
    RETIRED = "retired"


DEFAULT_ROBOT_HEARTBEAT_INTERVAL_SECONDS: Final[int] = 5

DEFAULT_ROBOT_OFFLINE_TIMEOUT_SECONDS: Final[int] = 30

DEFAULT_ROBOT_STATE_CACHE_TTL_SECONDS: Final[int] = 10

DEFAULT_ROBOT_HEALTH_CACHE_TTL_SECONDS: Final[int] = 15

MAX_ROBOTS_PER_FLEET_DEFAULT: Final[int] = 1000

ROBOT_CONTROL_REQUIRES_AUTHORIZATION: Final[bool] = True

ROBOT_SAFETY_REVIEW_REQUIRED: Final[bool] = True

PHYSICAL_ACTUATION_ENABLED_BY_DEFAULT: Final[bool] = False


# ============================================================
# Robot Health
# ============================================================


class HealthLevel(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    GOOD = "good"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    MAINTENANCE_REQUIRED = "maintenance_required"


HEALTH_SCORE_MIN: Final[float] = 0.0

HEALTH_SCORE_MAX: Final[float] = 100.0

HEALTH_SCORE_HEALTHY_MIN: Final[float] = 85.0

HEALTH_SCORE_WARNING_MIN: Final[float] = 65.0

HEALTH_SCORE_CRITICAL_BELOW: Final[float] = 40.0

DEFAULT_HEALTH_EVALUATION_INTERVAL_SECONDS: Final[int] = 60

DEFAULT_PREDICTIVE_MAINTENANCE_INTERVAL_SECONDS: Final[int] = 300


# ============================================================
# Telemetry
# ============================================================

DEFAULT_TELEMETRY_CACHE_TTL_SECONDS: Final[int] = 5

DEFAULT_TELEMETRY_BATCH_SIZE: Final[int] = 500

MAX_TELEMETRY_BATCH_SIZE: Final[int] = 10_000

DEFAULT_TELEMETRY_RETENTION_DAYS: Final[int] = 90

DEFAULT_TELEMETRY_AGGREGATION_INTERVAL_SECONDS: Final[int] = 60

DEFAULT_TELEMETRY_STALE_AFTER_SECONDS: Final[int] = 30

DEFAULT_TELEMETRY_QUEUE_MAX_SIZE: Final[int] = 100_000


class TelemetryTransport(str, Enum):
    HTTP = "http"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    KAFKA = "kafka"
    ROS = "ros"
    ROS2 = "ros2"
    GRPC = "grpc"


class TelemetryQualityLevel(str, Enum):
    UNKNOWN = "unknown"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    INVALID = "invalid"
    MISSING = "missing"
    STALE = "stale"


# ============================================================
# Sensors
# ============================================================

DEFAULT_SENSOR_HEALTH_CACHE_TTL_SECONDS: Final[int] = 15

DEFAULT_SENSOR_HEALTH_CHECK_INTERVAL_SECONDS: Final[int] = 60

DEFAULT_SENSOR_CALIBRATION_REMINDER_DAYS: Final[int] = 7

DEFAULT_SENSOR_READING_RETENTION_DAYS: Final[int] = 30


class SensorCategory(str, Enum):
    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    THERMAL_CAMERA = "thermal_camera"
    LIDAR = "lidar"
    RADAR = "radar"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    GPS = "gps"
    GNSS = "gnss"
    ENCODER = "encoder"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    FORCE = "force"
    TORQUE = "torque"
    TACTILE = "tactile"
    VOLTAGE = "voltage"
    CURRENT = "current"
    BATTERY = "battery"
    GAS = "gas"
    SMOKE = "smoke"
    MICROPHONE = "microphone"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


# ============================================================
# Navigation
# ============================================================


class NavigationMode(str, Enum):
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    ASSISTED = "assisted"
    WAYPOINT = "waypoint"
    EXPLORATION = "exploration"
    FOLLOW_TARGET = "follow_target"
    RETURN_TO_BASE = "return_to_base"
    SIMULATION = "simulation"


class NavigationPlanner(str, Enum):
    ASTAR = "astar"
    DSTAR = "dstar"
    RRT = "rrt"
    RRT_STAR = "rrt_star"
    PRM = "prm"
    DWA = "dwa"
    TEB = "teb"
    MPC = "mpc"
    LEARNED = "learned"
    HYBRID = "hybrid"


DEFAULT_NAVIGATION_REPLAN_INTERVAL_SECONDS: Final[float] = 1.0

DEFAULT_NAVIGATION_GOAL_TOLERANCE_METERS: Final[float] = 0.2

DEFAULT_NAVIGATION_TIMEOUT_SECONDS: Final[int] = 300

MAX_NAVIGATION_WAYPOINTS: Final[int] = 1000

NAVIGATION_REQUIRES_SAFETY_VALIDATION: Final[bool] = True


# ============================================================
# Manipulation
# ============================================================


class ManipulationTaskType(str, Enum):
    PICK = "pick"
    PLACE = "place"
    PICK_AND_PLACE = "pick_and_place"
    GRASP = "grasp"
    RELEASE = "release"
    PUSH = "push"
    PULL = "pull"
    LIFT = "lift"
    SORT = "sort"
    ASSEMBLY = "assembly"
    DISASSEMBLY = "disassembly"
    INSPECTION = "inspection"
    TOOL_USE = "tool_use"
    CUSTOM = "custom"


DEFAULT_MANIPULATION_TIMEOUT_SECONDS: Final[int] = 120

MAX_MANIPULATION_RETRY_ATTEMPTS: Final[int] = 3

MANIPULATION_REQUIRES_COLLISION_CHECK: Final[bool] = True

MANIPULATION_REQUIRES_WORKSPACE_VALIDATION: Final[bool] = True


# ============================================================
# Physical AI / VLA / World Models
# ============================================================


class AIModelCategory(str, Enum):
    LANGUAGE_MODEL = "language_model"
    VISION_LANGUAGE_MODEL = "vision_language_model"
    VISION_LANGUAGE_ACTION = "vision_language_action"
    WORLD_MODEL = "world_model"
    PERCEPTION_MODEL = "perception_model"
    NAVIGATION_POLICY = "navigation_policy"
    MANIPULATION_POLICY = "manipulation_policy"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    IMITATION_LEARNING = "imitation_learning"
    ANOMALY_DETECTION = "anomaly_detection"
    FAILURE_PREDICTION = "failure_prediction"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    SENSOR_FUSION = "sensor_fusion"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


class ModelLifecycleStage(str, Enum):
    REGISTERED = "registered"
    VALIDATING = "validating"
    EXPERIMENTAL = "experimental"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


DEFAULT_INFERENCE_TIMEOUT_SECONDS: Final[int] = 60

DEFAULT_MODEL_BATCH_SIZE: Final[int] = 16

DEFAULT_MODEL_CACHE_TTL_SECONDS: Final[int] = 300

MAX_MODEL_INPUT_TOKENS_DEFAULT: Final[int] = 8192

MAX_MODEL_OUTPUT_TOKENS_DEFAULT: Final[int] = 2048

MODEL_DEPLOYMENT_REQUIRES_APPROVAL: Final[bool] = True

PHYSICAL_AI_POLICY_DEPLOYMENT_REQUIRES_EVALUATION: Final[bool] = True

UNVALIDATED_POLICY_HARDWARE_EXECUTION_ALLOWED: Final[bool] = False


# ============================================================
# RAG
# ============================================================

DEFAULT_RAG_TOP_K: Final[int] = 10

MAX_RAG_TOP_K: Final[int] = 100

DEFAULT_RAG_CHUNK_SIZE: Final[int] = 1000

DEFAULT_RAG_CHUNK_OVERLAP: Final[int] = 150

DEFAULT_RAG_RERANK_TOP_K: Final[int] = 5

DEFAULT_RAG_CACHE_TTL_SECONDS: Final[int] = 300

RAG_REQUIRE_SOURCE_CITATIONS: Final[bool] = True

RAG_ALLOW_UNGROUNDED_SAFETY_INSTRUCTIONS: Final[bool] = False


# ============================================================
# Simulation
# ============================================================


class SimulationEngineName(str, Enum):
    ISAAC_SIM = "isaac_sim"
    ISAAC_LAB = "isaac_lab"
    GAZEBO = "gazebo"
    IGNITION_GAZEBO = "ignition_gazebo"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"
    WEBOTS = "webots"
    COPPELIASIM = "coppeliasim"
    UNITY = "unity"
    UNREAL = "unreal"
    CARLA = "carla"
    AIRSIM = "airsim"
    BRAX = "brax"
    CUSTOM = "custom"


DEFAULT_SIMULATION_TIMEOUT_SECONDS: Final[int] = 3600

DEFAULT_SIMULATION_MAX_STEPS: Final[int] = 100_000

DEFAULT_SIMULATION_EPISODES: Final[int] = 100

DEFAULT_SIMULATION_RANDOM_SEED: Final[int] = 42

DEFAULT_SIMULATION_CHECKPOINT_INTERVAL_STEPS: Final[int] = 1000

SIMULATION_REQUIRED_BEFORE_PHYSICAL_DEPLOYMENT: Final[bool] = True


# ============================================================
# Evaluation
# ============================================================


class EvaluationCategory(str, Enum):
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    LOCOMOTION = "locomotion"
    WORLD_MODEL = "world_model"
    VLA = "vla"
    POLICY = "policy"
    RAG = "rag"
    SAFETY = "safety"
    ROBOT_HEALTH = "robot_health"
    REPAIR = "repair"
    LATENCY = "latency"
    RESOURCE = "resource"
    SIMULATION = "simulation"
    END_TO_END = "end_to_end"


class EvaluationResultStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


DEFAULT_EVALUATION_TIMEOUT_SECONDS: Final[int] = 1800

DEFAULT_BENCHMARK_REPETITIONS: Final[int] = 3

DEFAULT_EVALUATION_RANDOM_SEED: Final[int] = 42

EVALUATION_REQUIRE_REPRODUCIBILITY: Final[bool] = True

EVALUATION_REQUIRE_SAFETY_METRICS: Final[bool] = True


# ============================================================
# Predictive Maintenance
# ============================================================


class MaintenancePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MaintenanceType(str, Enum):
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"
    CALIBRATION = "calibration"


class FailureRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


DEFAULT_RUL_PREDICTION_HORIZON_HOURS: Final[int] = 720

DEFAULT_MAINTENANCE_FORECAST_DAYS: Final[int] = 30

DEFAULT_MAINTENANCE_REMINDER_DAYS: Final[int] = 7

DEFAULT_ANOMALY_SCORE_THRESHOLD: Final[float] = 0.8

PREDICTIVE_MAINTENANCE_REQUIRES_HUMAN_REVIEW: Final[bool] = True


# ============================================================
# Repair and Parts
# ============================================================


class RepairDecision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    INSPECT = "inspect"
    MONITOR = "monitor"
    DEFER = "defer"
    UNDETERMINED = "undetermined"


class RepairStatus(str, Enum):
    REQUESTED = "requested"
    INSPECTION_PENDING = "inspection_pending"
    DIAGNOSING = "diagnosing"
    ESTIMATING = "estimating"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


DEFAULT_CURRENCY: Final[str] = "INR"

SUPPORTED_CURRENCIES: Final[frozenset[str]] = frozenset({
    "INR",
    "USD",
    "EUR",
    "GBP",
    "AED",
    "CAD",
    "AUD",
    "SGD",
    "JPY",
})

DEFAULT_REPAIR_ESTIMATE_VALIDITY_DAYS: Final[int] = 30

DEFAULT_PARTS_LOW_STOCK_THRESHOLD: Final[int] = 5

REPAIR_ESTIMATES_REQUIRE_HUMAN_APPROVAL: Final[bool] = True

ROBOT_RETURN_TO_SERVICE_REQUIRES_SAFETY_CHECK: Final[bool] = True


# ============================================================
# Safety
# ============================================================


class SafetySeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyState(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    WARNING = "warning"
    DEGRADED = "degraded"
    UNSAFE = "unsafe"
    EMERGENCY_STOPPED = "emergency_stopped"
    RECOVERY_PENDING = "recovery_pending"


class SafetyAction(str, Enum):
    ALERT = "alert"
    REQUEST_OPERATOR_REVIEW = "request_operator_review"
    RESTRICT_TASK = "restrict_task"
    REJECT_TASK = "reject_task"
    REQUEST_SAFE_STOP = "request_safe_stop"
    ESCALATE_INCIDENT = "escalate_incident"
    REQUIRE_INSPECTION = "require_inspection"


SAFETY_FAIL_CLOSED: Final[bool] = True

SAFETY_HUMAN_APPROVAL_REQUIRED: Final[bool] = True

SAFETY_OVERRIDE_ENABLED_BY_DEFAULT: Final[bool] = False

SAFETY_AUDIT_LOG_REQUIRED: Final[bool] = True

SAFETY_CRITICAL_ALERT_ACKNOWLEDGMENT_REQUIRED: Final[bool] = True

EMERGENCY_STOP_RESET_REQUIRES_AUTHORIZATION: Final[bool] = True

EMERGENCY_STOP_RESET_REQUIRES_PHYSICAL_VERIFICATION: Final[bool] = True

AUTONOMOUS_SAFETY_OVERRIDE_ALLOWED: Final[bool] = False


# ============================================================
# Incidents and Alerts
# ============================================================


class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class IncidentStatus(str, Enum):
    OPEN = "open"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


DEFAULT_ALERT_DEDUPLICATION_SECONDS: Final[int] = 60

DEFAULT_ALERT_ESCALATION_MINUTES: Final[int] = 15

DEFAULT_INCIDENT_REVIEW_REQUIRED: Final[bool] = True

CRITICAL_INCIDENT_AUTOMATIC_CLOSURE_ALLOWED: Final[bool] = False


# ============================================================
# Caching
# ============================================================

CACHE_KEY_PREFIX: Final[str] = "physical_ai"

DEFAULT_CACHE_TTL_SECONDS: Final[int] = 300

DEFAULT_CACHE_MAX_MEMORY_ENTRIES: Final[int] = 10_000

DEFAULT_CACHE_LOCK_TTL_SECONDS: Final[int] = 30

DEFAULT_CACHE_LOCK_WAIT_SECONDS: Final[float] = 5.0

DEFAULT_FLEET_CACHE_TTL_SECONDS: Final[int] = 30

DEFAULT_DASHBOARD_CACHE_TTL_SECONDS: Final[int] = 30


# ============================================================
# Messaging and Events
# ============================================================

DEFAULT_KAFKA_CONSUMER_GROUP: Final[str] = "physical-ai-workers"

DEFAULT_MQTT_QOS: Final[int] = 1

DEFAULT_EVENT_RETRY_ATTEMPTS: Final[int] = 3

DEFAULT_EVENT_RETENTION_DAYS: Final[int] = 90

EVENT_SCHEMA_VERSION: Final[str] = "1.0"


class EventTopic(str, Enum):
    ROBOT_TELEMETRY = "robot.telemetry"
    ROBOT_STATE = "robot.state"
    ROBOT_HEALTH = "robot.health"
    ROBOT_EVENTS = "robot.events"
    SENSOR_READINGS = "sensor.readings"
    SENSOR_HEALTH = "sensor.health"
    NAVIGATION_EVENTS = "navigation.events"
    MANIPULATION_EVENTS = "manipulation.events"
    SAFETY_ALERTS = "safety.alerts"
    INCIDENTS = "incidents"
    MAINTENANCE_EVENTS = "maintenance.events"
    REPAIR_EVENTS = "repair.events"
    FLEET_EVENTS = "fleet.events"
    SIMULATION_EVENTS = "simulation.events"
    MODEL_EVENTS = "model.events"
    EVALUATION_EVENTS = "evaluation.events"
    AUDIT_EVENTS = "audit.events"


# ============================================================
# Storage and Files
# ============================================================

ALLOWED_DOCUMENT_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
    ".json",
    ".xlsx",
    ".md",
})

ALLOWED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tiff",
})

ALLOWED_VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".mp4",
    ".mov",
    ".webm",
})

ALLOWED_ROBOT_ASSET_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".urdf",
    ".xacro",
    ".sdf",
    ".usd",
    ".usda",
    ".usdc",
    ".stl",
    ".obj",
    ".ply",
    ".pcd",
    ".glb",
    ".gltf",
})

DEFAULT_ARTIFACT_RETENTION_DAYS: Final[int] = 90

DEFAULT_LOG_RETENTION_DAYS: Final[int] = 30

MAX_IMAGE_UPLOAD_BYTES: Final[int] = 20 * 1024 * 1024

MAX_DOCUMENT_UPLOAD_BYTES: Final[int] = 50 * 1024 * 1024

MAX_VIDEO_UPLOAD_BYTES: Final[int] = 500 * 1024 * 1024

UPLOAD_MALWARE_SCANNING_REQUIRED: Final[bool] = True

UPLOAD_CONTENT_VALIDATION_REQUIRED: Final[bool] = True


# ============================================================
# Observability
# ============================================================

DEFAULT_LOG_LEVEL: Final[str] = "INFO"

DEFAULT_METRICS_PREFIX: Final[str] = "physical_ai"

DEFAULT_TRACE_SAMPLE_RATE: Final[float] = 0.1

DEFAULT_METRICS_COLLECTION_INTERVAL_SECONDS: Final[int] = 15

ENABLE_STRUCTURED_LOGGING: Final[bool] = True

ENABLE_REQUEST_CORRELATION_IDS: Final[bool] = True

ENABLE_AUDIT_LOGGING: Final[bool] = True

LOG_SENSITIVE_DATA: Final[bool] = False


# ============================================================
# HTTP Status Codes
# ============================================================


class HTTPStatus(int, Enum):
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


# ============================================================
# Application Error Codes
# ============================================================


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_REQUIRED = "authentication_required"
    INVALID_CREDENTIALS = "invalid_credentials"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RESOURCE_CONFLICT = "resource_conflict"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TENANT_ACCESS_DENIED = "tenant_access_denied"

    ROBOT_NOT_FOUND = "robot_not_found"
    ROBOT_OFFLINE = "robot_offline"
    ROBOT_UNAVAILABLE = "robot_unavailable"
    ROBOT_FAULT = "robot_fault"
    ROBOT_EMERGENCY_STOPPED = "robot_emergency_stopped"

    SENSOR_NOT_FOUND = "sensor_not_found"
    SENSOR_FAULT = "sensor_fault"
    SENSOR_CALIBRATION_REQUIRED = "sensor_calibration_required"

    TELEMETRY_INVALID = "telemetry_invalid"
    TELEMETRY_STALE = "telemetry_stale"

    SAFETY_CHECK_FAILED = "safety_check_failed"
    SAFETY_APPROVAL_REQUIRED = "safety_approval_required"
    SAFETY_POLICY_VIOLATION = "safety_policy_violation"

    NAVIGATION_FAILED = "navigation_failed"
    MANIPULATION_FAILED = "manipulation_failed"

    MODEL_NOT_FOUND = "model_not_found"
    MODEL_INFERENCE_FAILED = "model_inference_failed"
    MODEL_EVALUATION_FAILED = "model_evaluation_failed"

    SIMULATION_FAILED = "simulation_failed"
    SIMULATION_TIMEOUT = "simulation_timeout"

    MAINTENANCE_REQUIRED = "maintenance_required"
    REPAIR_APPROVAL_REQUIRED = "repair_approval_required"
    PART_NOT_AVAILABLE = "part_not_available"

    DATABASE_ERROR = "database_error"
    CACHE_ERROR = "cache_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    INTERNAL_ERROR = "internal_error"


# ============================================================
# Utility Constants
# ============================================================

SECONDS_PER_MINUTE: Final[int] = 60

SECONDS_PER_HOUR: Final[int] = 3600

SECONDS_PER_DAY: Final[int] = 86400

MILLISECONDS_PER_SECOND: Final[int] = 1000

BYTES_PER_KIB: Final[int] = 1024

BYTES_PER_MIB: Final[int] = 1024**2

BYTES_PER_GIB: Final[int] = 1024**3

PERCENTAGE_MIN: Final[float] = 0.0

PERCENTAGE_MAX: Final[float] = 100.0

PROBABILITY_MIN: Final[float] = 0.0

PROBABILITY_MAX: Final[float] = 1.0


# ============================================================
# Exports
# ============================================================

__all__ = [
    name
    for name in globals()
    if (
        not name.startswith("_")
        and name.isupper()
    )
] + [
    "Environment",
    "SystemRole",
    "Permission",
    "RobotCategory",
    "RobotOperatingMode",
    "RobotLifecycleStatus",
    "HealthLevel",
    "TelemetryTransport",
    "TelemetryQualityLevel",
    "SensorCategory",
    "NavigationMode",
    "NavigationPlanner",
    "ManipulationTaskType",
    "AIModelCategory",
    "ModelLifecycleStage",
    "SimulationEngineName",
    "EvaluationCategory",
    "EvaluationResultStatus",
    "MaintenancePriority",
    "MaintenanceType",
    "FailureRiskLevel",
    "RepairDecision",
    "RepairStatus",
    "SafetySeverity",
    "SafetyState",
    "SafetyAction",
    "AlertSeverity",
    "AlertStatus",
    "IncidentStatus",
    "EventTopic",
    "HTTPStatus",
    "ErrorCode",
]