"""Exceptions module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Exception Handling

Features:
- Centralized application exceptions
- FastAPI-compatible exception handlers
- Structured JSON error responses
- Authentication and authorization errors
- Multi-tenant access protection
- Robotics and fleet errors
- Sensor and telemetry errors
- Navigation and manipulation errors
- Physical AI / VLA / world model errors
- Simulation and evaluation errors
- Predictive maintenance and repair errors
- Safety and emergency errors
- Database, caching, and event bus errors
- Correlation IDs and structured logging
- Safe production error responses

Security:
- Never expose stack traces in API responses.
- Never expose passwords, tokens, or internal credentials.
- Safety-related exceptions must fail closed.
- Exceptions do not directly command physical hardware.
"""

from __future__ import annotations

import logging
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_error_id() -> str:
    return str(uuid.uuid4())


def safe_string(value: Any, max_length: int = 500) -> str:
    text = str(value)

    if len(text) > max_length:
        return text[:max_length] + "..."

    return text


def get_request_id(request: Request) -> str:
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    if request_id:
        return str(request_id)

    return generate_error_id()


# ============================================================
# Error Categories
# ============================================================


class ErrorCategory(str, Enum):
    GENERAL = "general"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SECURITY = "security"
    TENANT = "tenant"
    DATABASE = "database"
    CACHE = "cache"
    EVENT_BUS = "event_bus"
    ROBOT = "robot"
    FLEET = "fleet"
    SENSOR = "sensor"
    TELEMETRY = "telemetry"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    MODEL = "model"
    INFERENCE = "inference"
    WORLD_MODEL = "world_model"
    VLA = "vla"
    RAG = "rag"
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    PARTS = "parts"
    SAFETY = "safety"
    INCIDENT = "incident"
    STORAGE = "storage"
    EXTERNAL_SERVICE = "external_service"
    INFRASTRUCTURE = "infrastructure"


class ErrorSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCode(str, Enum):
    INTERNAL_ERROR = "internal_error"
    VALIDATION_ERROR = "validation_error"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    OPERATION_TIMEOUT = "operation_timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    AUTHENTICATION_REQUIRED = "authentication_required"
    INVALID_CREDENTIALS = "invalid_credentials"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_INVALID = "token_invalid"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_DISABLED = "account_disabled"
    MFA_REQUIRED = "mfa_required"

    PERMISSION_DENIED = "permission_denied"
    TENANT_ACCESS_DENIED = "tenant_access_denied"
    TENANT_NOT_FOUND = "tenant_not_found"

    DATABASE_ERROR = "database_error"
    DATABASE_CONNECTION_ERROR = "database_connection_error"
    DATABASE_INTEGRITY_ERROR = "database_integrity_error"
    TRANSACTION_ERROR = "transaction_error"

    CACHE_ERROR = "cache_error"
    CACHE_CONNECTION_ERROR = "cache_connection_error"
    CACHE_LOCK_ERROR = "cache_lock_error"

    EVENT_BUS_ERROR = "event_bus_error"
    EVENT_PUBLISH_ERROR = "event_publish_error"
    EVENT_HANDLER_ERROR = "event_handler_error"
    EVENT_QUEUE_FULL = "event_queue_full"

    ROBOT_NOT_FOUND = "robot_not_found"
    ROBOT_OFFLINE = "robot_offline"
    ROBOT_UNAVAILABLE = "robot_unavailable"
    ROBOT_FAULT = "robot_fault"
    ROBOT_BUSY = "robot_busy"
    ROBOT_EMERGENCY_STOPPED = "robot_emergency_stopped"
    ROBOT_COMMAND_REJECTED = "robot_command_rejected"
    ROBOT_CONTROL_NOT_AUTHORIZED = "robot_control_not_authorized"

    FLEET_NOT_FOUND = "fleet_not_found"
    FLEET_OPERATION_FAILED = "fleet_operation_failed"

    SENSOR_NOT_FOUND = "sensor_not_found"
    SENSOR_FAILURE = "sensor_failure"
    SENSOR_CALIBRATION_REQUIRED = "sensor_calibration_required"
    SENSOR_DATA_INVALID = "sensor_data_invalid"

    TELEMETRY_INVALID = "telemetry_invalid"
    TELEMETRY_STALE = "telemetry_stale"
    TELEMETRY_INGESTION_FAILED = "telemetry_ingestion_failed"

    NAVIGATION_FAILED = "navigation_failed"
    NAVIGATION_TIMEOUT = "navigation_timeout"
    NAVIGATION_PATH_NOT_FOUND = "navigation_path_not_found"
    NAVIGATION_COLLISION_RISK = "navigation_collision_risk"

    MANIPULATION_FAILED = "manipulation_failed"
    GRASP_FAILED = "grasp_failed"
    MANIPULATION_WORKSPACE_VIOLATION = (
        "manipulation_workspace_violation"
    )

    PERCEPTION_FAILED = "perception_failed"
    PERCEPTION_CONFIDENCE_LOW = "perception_confidence_low"

    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    MODEL_INFERENCE_FAILED = "model_inference_failed"
    MODEL_DEPLOYMENT_REJECTED = "model_deployment_rejected"
    MODEL_VALIDATION_FAILED = "model_validation_failed"

    WORLD_MODEL_ERROR = "world_model_error"
    WORLD_MODEL_PREDICTION_FAILED = (
        "world_model_prediction_failed"
    )

    VLA_INFERENCE_FAILED = "vla_inference_failed"
    VLA_ACTION_REJECTED = "vla_action_rejected"

    RAG_RETRIEVAL_FAILED = "rag_retrieval_failed"
    RAG_GROUNDING_FAILED = "rag_grounding_failed"

    SIMULATION_FAILED = "simulation_failed"
    SIMULATION_TIMEOUT = "simulation_timeout"
    SIMULATION_ENGINE_UNAVAILABLE = (
        "simulation_engine_unavailable"
    )

    EVALUATION_FAILED = "evaluation_failed"
    BENCHMARK_FAILED = "benchmark_failed"
    EVALUATION_THRESHOLD_NOT_MET = (
        "evaluation_threshold_not_met"
    )

    MAINTENANCE_REQUIRED = "maintenance_required"
    MAINTENANCE_OPERATION_FAILED = (
        "maintenance_operation_failed"
    )
    FAILURE_PREDICTION_FAILED = "failure_prediction_failed"

    REPAIR_NOT_FOUND = "repair_not_found"
    REPAIR_ESTIMATION_FAILED = "repair_estimation_failed"
    REPAIR_APPROVAL_REQUIRED = "repair_approval_required"

    PART_NOT_FOUND = "part_not_found"
    PART_NOT_AVAILABLE = "part_not_available"
    PART_COMPATIBILITY_FAILED = "part_compatibility_failed"

    SAFETY_CHECK_FAILED = "safety_check_failed"
    SAFETY_POLICY_VIOLATION = "safety_policy_violation"
    SAFETY_APPROVAL_REQUIRED = "safety_approval_required"
    EMERGENCY_STOP_ACTIVE = "emergency_stop_active"
    UNSAFE_OPERATION = "unsafe_operation"
    SAFETY_INTERLOCK_ACTIVE = "safety_interlock_active"

    INCIDENT_NOT_FOUND = "incident_not_found"
    INCIDENT_OPERATION_FAILED = "incident_operation_failed"

    FILE_NOT_FOUND = "file_not_found"
    FILE_TOO_LARGE = "file_too_large"
    FILE_TYPE_NOT_ALLOWED = "file_type_not_allowed"
    FILE_SECURITY_CHECK_FAILED = "file_security_check_failed"

    EXTERNAL_SERVICE_ERROR = "external_service_error"
    EXTERNAL_SERVICE_TIMEOUT = "external_service_timeout"

    CONFIGURATION_ERROR = "configuration_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


# ============================================================
# Base Application Exception
# ============================================================


class AppException(Exception):
    """
    Base exception for all application-level errors.

    Internal details are kept separate from public messages.
    """

    default_message = "An application error occurred"
    default_code = ErrorCode.INTERNAL_ERROR
    default_status_code = 500
    default_category = ErrorCategory.GENERAL
    default_severity = ErrorSeverity.ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | str | None = None,
        status_code: int | None = None,
        category: ErrorCategory | str | None = None,
        severity: ErrorSeverity | str | None = None,
        details: Mapping[str, Any] | None = None,
        internal_message: str | None = None,
        retryable: bool = False,
        error_id: str | None = None,
    ) -> None:
        self.message = message or self.default_message

        selected_code = code or self.default_code
        self.code = (
            selected_code.value
            if isinstance(selected_code, Enum)
            else str(selected_code)
        )

        self.status_code = (
            status_code
            if status_code is not None
            else self.default_status_code
        )

        selected_category = category or self.default_category
        self.category = (
            selected_category.value
            if isinstance(selected_category, Enum)
            else str(selected_category)
        )

        selected_severity = severity or self.default_severity
        self.severity = (
            selected_severity.value
            if isinstance(selected_severity, Enum)
            else str(selected_severity)
        )

        self.details = dict(details or {})
        self.internal_message = internal_message
        self.retryable = retryable
        self.error_id = error_id or generate_error_id()

        super().__init__(self.message)

    def to_dict(
        self,
        *,
        request_id: str | None = None,
        include_details: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "success": False,
            "error": {
                "error_id": self.error_id,
                "code": self.code,
                "message": self.message,
                "category": self.category,
                "severity": self.severity,
                "retryable": self.retryable,
            },
            "timestamp": utc_now().isoformat(),
        }

        if request_id:
            result["request_id"] = request_id

        if include_details and self.details:
            result["error"]["details"] = self.details

        return result


# ============================================================
# General Exceptions
# ============================================================


class BadRequestException(AppException):
    default_message = "Invalid request"
    default_code = ErrorCode.BAD_REQUEST
    default_status_code = 400


class ResourceNotFoundException(AppException):
    default_message = "Requested resource was not found"
    default_code = ErrorCode.NOT_FOUND
    default_status_code = 404


class ResourceConflictException(AppException):
    default_message = "Resource conflict"
    default_code = ErrorCode.CONFLICT
    default_status_code = 409


class OperationTimeoutException(AppException):
    default_message = "Operation timed out"
    default_code = ErrorCode.OPERATION_TIMEOUT
    default_status_code = 504


class ServiceUnavailableException(AppException):
    default_message = "Service temporarily unavailable"
    default_code = ErrorCode.SERVICE_UNAVAILABLE
    default_status_code = 503


class RateLimitExceededException(AppException):
    default_message = "Rate limit exceeded"
    default_code = ErrorCode.RATE_LIMIT_EXCEEDED
    default_status_code = 429


class ApplicationValidationException(AppException):
    default_message = "Validation failed"
    default_code = ErrorCode.VALIDATION_ERROR
    default_status_code = 422
    default_category = ErrorCategory.VALIDATION


# ============================================================
# Authentication Exceptions
# ============================================================


class AuthenticationException(AppException):
    default_message = "Authentication required"
    default_code = ErrorCode.AUTHENTICATION_REQUIRED
    default_status_code = 401
    default_category = ErrorCategory.AUTHENTICATION


class InvalidCredentialsException(AuthenticationException):
    default_message = "Invalid credentials"
    default_code = ErrorCode.INVALID_CREDENTIALS


class TokenExpiredException(AuthenticationException):
    default_message = "Authentication token has expired"
    default_code = ErrorCode.TOKEN_EXPIRED


class InvalidTokenException(AuthenticationException):
    default_message = "Invalid authentication token"
    default_code = ErrorCode.TOKEN_INVALID


class AccountLockedException(AuthenticationException):
    default_message = "Account is temporarily locked"
    default_code = ErrorCode.ACCOUNT_LOCKED
    default_status_code = 403


class AccountDisabledException(AuthenticationException):
    default_message = "Account is disabled"
    default_code = ErrorCode.ACCOUNT_DISABLED
    default_status_code = 403


class MFARequiredException(AuthenticationException):
    default_message = "Multi-factor authentication required"
    default_code = ErrorCode.MFA_REQUIRED


# ============================================================
# Authorization and Tenant Exceptions
# ============================================================


class PermissionDeniedException(AppException):
    default_message = "Permission denied"
    default_code = ErrorCode.PERMISSION_DENIED
    default_status_code = 403
    default_category = ErrorCategory.AUTHORIZATION


class TenantAccessDeniedException(PermissionDeniedException):
    default_message = "Tenant access denied"
    default_code = ErrorCode.TENANT_ACCESS_DENIED
    default_category = ErrorCategory.TENANT


class TenantNotFoundException(ResourceNotFoundException):
    default_message = "Tenant not found"
    default_code = ErrorCode.TENANT_NOT_FOUND
    default_category = ErrorCategory.TENANT


# ============================================================
# Database Exceptions
# ============================================================


class DatabaseException(AppException):
    default_message = "Database operation failed"
    default_code = ErrorCode.DATABASE_ERROR
    default_status_code = 500
    default_category = ErrorCategory.DATABASE


class DatabaseConnectionException(DatabaseException):
    default_message = "Database temporarily unavailable"
    default_code = ErrorCode.DATABASE_CONNECTION_ERROR
    default_status_code = 503


class DatabaseIntegrityException(DatabaseException):
    default_message = "Database integrity constraint violated"
    default_code = ErrorCode.DATABASE_INTEGRITY_ERROR
    default_status_code = 409


class TransactionException(DatabaseException):
    default_message = "Database transaction failed"
    default_code = ErrorCode.TRANSACTION_ERROR


# ============================================================
# Cache Exceptions
# ============================================================


class CacheException(AppException):
    default_message = "Cache operation failed"
    default_code = ErrorCode.CACHE_ERROR
    default_category = ErrorCategory.CACHE
    default_status_code = 503


class CacheConnectionException(CacheException):
    default_message = "Cache service unavailable"
    default_code = ErrorCode.CACHE_CONNECTION_ERROR


class CacheLockException(CacheException):
    default_message = "Unable to acquire cache lock"
    default_code = ErrorCode.CACHE_LOCK_ERROR
    default_status_code = 409


# ============================================================
# Event Bus Exceptions
# ============================================================


class EventBusException(AppException):
    default_message = "Event bus operation failed"
    default_code = ErrorCode.EVENT_BUS_ERROR
    default_category = ErrorCategory.EVENT_BUS
    default_status_code = 503


class EventPublishException(EventBusException):
    default_message = "Event publication failed"
    default_code = ErrorCode.EVENT_PUBLISH_ERROR


class EventHandlerException(EventBusException):
    default_message = "Event processing failed"
    default_code = ErrorCode.EVENT_HANDLER_ERROR


class EventQueueFullException(EventBusException):
    default_message = "Event queue is full"
    default_code = ErrorCode.EVENT_QUEUE_FULL


# ============================================================
# Robot Exceptions
# ============================================================


class RobotException(AppException):
    default_message = "Robot operation failed"
    default_code = ErrorCode.ROBOT_UNAVAILABLE
    default_category = ErrorCategory.ROBOT
    default_status_code = 409


class RobotNotFoundException(RobotException):
    default_message = "Robot not found"
    default_code = ErrorCode.ROBOT_NOT_FOUND
    default_status_code = 404


class RobotOfflineException(RobotException):
    default_message = "Robot is offline"
    default_code = ErrorCode.ROBOT_OFFLINE
    default_status_code = 503


class RobotUnavailableException(RobotException):
    default_message = "Robot is unavailable"
    default_code = ErrorCode.ROBOT_UNAVAILABLE
    default_status_code = 503


class RobotFaultException(RobotException):
    default_message = "Robot has an active fault"
    default_code = ErrorCode.ROBOT_FAULT


class RobotBusyException(RobotException):
    default_message = "Robot is currently busy"
    default_code = ErrorCode.ROBOT_BUSY


class RobotEmergencyStoppedException(RobotException):
    default_message = "Robot emergency stop is active"
    default_code = ErrorCode.ROBOT_EMERGENCY_STOPPED


class RobotCommandRejectedException(RobotException):
    default_message = "Robot command was rejected"
    default_code = ErrorCode.ROBOT_COMMAND_REJECTED


class RobotControlUnauthorizedException(RobotException):
    default_message = "Robot control authorization required"
    default_code = ErrorCode.ROBOT_CONTROL_NOT_AUTHORIZED
    default_status_code = 403


# ============================================================
# Fleet Exceptions
# ============================================================


class FleetException(AppException):
    default_message = "Fleet operation failed"
    default_code = ErrorCode.FLEET_OPERATION_FAILED
    default_category = ErrorCategory.FLEET
    default_status_code = 409


class FleetNotFoundException(FleetException):
    default_message = "Fleet not found"
    default_code = ErrorCode.FLEET_NOT_FOUND
    default_status_code = 404


# ============================================================
# Sensor Exceptions
# ============================================================


class SensorException(AppException):
    default_message = "Sensor operation failed"
    default_code = ErrorCode.SENSOR_FAILURE
    default_category = ErrorCategory.SENSOR
    default_status_code = 409


class SensorNotFoundException(SensorException):
    default_message = "Sensor not found"
    default_code = ErrorCode.SENSOR_NOT_FOUND
    default_status_code = 404


class SensorFailureException(SensorException):
    default_message = "Sensor failure detected"
    default_code = ErrorCode.SENSOR_FAILURE


class SensorCalibrationRequiredException(SensorException):
    default_message = "Sensor calibration required"
    default_code = ErrorCode.SENSOR_CALIBRATION_REQUIRED


class SensorDataInvalidException(SensorException):
    default_message = "Invalid sensor data"
    default_code = ErrorCode.SENSOR_DATA_INVALID
    default_status_code = 422


# ============================================================
# Telemetry Exceptions
# ============================================================


class TelemetryException(AppException):
    default_message = "Telemetry operation failed"
    default_code = ErrorCode.TELEMETRY_INVALID
    default_category = ErrorCategory.TELEMETRY
    default_status_code = 422


class TelemetryInvalidException(TelemetryException):
    default_message = "Invalid telemetry data"


class TelemetryStaleException(TelemetryException):
    default_message = "Telemetry data is stale"
    default_code = ErrorCode.TELEMETRY_STALE
    default_status_code = 409


class TelemetryIngestionException(TelemetryException):
    default_message = "Telemetry ingestion failed"
    default_code = ErrorCode.TELEMETRY_INGESTION_FAILED
    default_status_code = 503


# ============================================================
# Navigation Exceptions
# ============================================================


class NavigationException(AppException):
    default_message = "Navigation operation failed"
    default_code = ErrorCode.NAVIGATION_FAILED
    default_category = ErrorCategory.NAVIGATION
    default_status_code = 409


class NavigationTimeoutException(NavigationException):
    default_message = "Navigation operation timed out"
    default_code = ErrorCode.NAVIGATION_TIMEOUT
    default_status_code = 504


class NavigationPathNotFoundException(NavigationException):
    default_message = "No valid navigation path found"
    default_code = ErrorCode.NAVIGATION_PATH_NOT_FOUND


class NavigationCollisionRiskException(NavigationException):
    default_message = "Navigation collision risk detected"
    default_code = ErrorCode.NAVIGATION_COLLISION_RISK


# ============================================================
# Manipulation Exceptions
# ============================================================


class ManipulationException(AppException):
    default_message = "Manipulation operation failed"
    default_code = ErrorCode.MANIPULATION_FAILED
    default_category = ErrorCategory.MANIPULATION
    default_status_code = 409


class GraspFailedException(ManipulationException):
    default_message = "Robot grasp operation failed"
    default_code = ErrorCode.GRASP_FAILED


class ManipulationWorkspaceException(ManipulationException):
    default_message = "Manipulation workspace violation"
    default_code = ErrorCode.MANIPULATION_WORKSPACE_VIOLATION


# ============================================================
# Perception Exceptions
# ============================================================


class PerceptionException(AppException):
    default_message = "Perception processing failed"
    default_code = ErrorCode.PERCEPTION_FAILED
    default_category = ErrorCategory.PERCEPTION
    default_status_code = 422


class LowPerceptionConfidenceException(PerceptionException):
    default_message = "Perception confidence is insufficient"
    default_code = ErrorCode.PERCEPTION_CONFIDENCE_LOW


# ============================================================
# AI Model Exceptions
# ============================================================


class ModelException(AppException):
    default_message = "AI model operation failed"
    default_code = ErrorCode.MODEL_INFERENCE_FAILED
    default_category = ErrorCategory.MODEL
    default_status_code = 500


class ModelNotFoundException(ModelException):
    default_message = "AI model not found"
    default_code = ErrorCode.MODEL_NOT_FOUND
    default_status_code = 404


class ModelLoadException(ModelException):
    default_message = "Unable to load AI model"
    default_code = ErrorCode.MODEL_LOAD_FAILED


class ModelInferenceException(ModelException):
    default_message = "AI model inference failed"
    default_code = ErrorCode.MODEL_INFERENCE_FAILED
    default_category = ErrorCategory.INFERENCE


class ModelDeploymentRejectedException(ModelException):
    default_message = "AI model deployment rejected"
    default_code = ErrorCode.MODEL_DEPLOYMENT_REJECTED
    default_status_code = 409


class ModelValidationException(ModelException):
    default_message = "AI model validation failed"
    default_code = ErrorCode.MODEL_VALIDATION_FAILED
    default_status_code = 422


# ============================================================
# World Model Exceptions
# ============================================================


class WorldModelException(AppException):
    default_message = "World model operation failed"
    default_code = ErrorCode.WORLD_MODEL_ERROR
    default_category = ErrorCategory.WORLD_MODEL
    default_status_code = 500


class WorldModelPredictionException(WorldModelException):
    default_message = "World model prediction failed"
    default_code = ErrorCode.WORLD_MODEL_PREDICTION_FAILED


# ============================================================
# VLA Exceptions
# ============================================================


class VLAException(AppException):
    default_message = "Vision-language-action operation failed"
    default_code = ErrorCode.VLA_INFERENCE_FAILED
    default_category = ErrorCategory.VLA
    default_status_code = 500


class VLAInferenceException(VLAException):
    default_message = "VLA inference failed"


class VLAActionRejectedException(VLAException):
    default_message = "VLA-generated action was rejected"
    default_code = ErrorCode.VLA_ACTION_REJECTED
    default_status_code = 409


# ============================================================
# RAG Exceptions
# ============================================================


class RAGException(AppException):
    default_message = "RAG operation failed"
    default_code = ErrorCode.RAG_RETRIEVAL_FAILED
    default_category = ErrorCategory.RAG
    default_status_code = 500


class RAGRetrievalException(RAGException):
    default_message = "Document retrieval failed"


class RAGGroundingException(RAGException):
    default_message = "Insufficient evidence to support the answer"
    default_code = ErrorCode.RAG_GROUNDING_FAILED
    default_status_code = 422


# ============================================================
# Simulation Exceptions
# ============================================================


class SimulationException(AppException):
    default_message = "Simulation failed"
    default_code = ErrorCode.SIMULATION_FAILED
    default_category = ErrorCategory.SIMULATION
    default_status_code = 500


class SimulationTimeoutException(SimulationException):
    default_message = "Simulation timed out"
    default_code = ErrorCode.SIMULATION_TIMEOUT
    default_status_code = 504


class SimulationEngineUnavailableException(SimulationException):
    default_message = "Simulation engine unavailable"
    default_code = ErrorCode.SIMULATION_ENGINE_UNAVAILABLE
    default_status_code = 503


# ============================================================
# Evaluation Exceptions
# ============================================================


class EvaluationException(AppException):
    default_message = "Evaluation failed"
    default_code = ErrorCode.EVALUATION_FAILED
    default_category = ErrorCategory.EVALUATION
    default_status_code = 500


class BenchmarkException(EvaluationException):
    default_message = "Benchmark execution failed"
    default_code = ErrorCode.BENCHMARK_FAILED


class EvaluationThresholdException(EvaluationException):
    default_message = "Evaluation acceptance criteria not met"
    default_code = ErrorCode.EVALUATION_THRESHOLD_NOT_MET
    default_status_code = 409


# ============================================================
# Maintenance Exceptions
# ============================================================


class MaintenanceException(AppException):
    default_message = "Maintenance operation failed"
    default_code = ErrorCode.MAINTENANCE_OPERATION_FAILED
    default_category = ErrorCategory.MAINTENANCE
    default_status_code = 409


class MaintenanceRequiredException(MaintenanceException):
    default_message = "Robot maintenance is required"
    default_code = ErrorCode.MAINTENANCE_REQUIRED


class FailurePredictionException(MaintenanceException):
    default_message = "Failure prediction failed"
    default_code = ErrorCode.FAILURE_PREDICTION_FAILED
    default_status_code = 500


# ============================================================
# Repair Exceptions
# ============================================================


class RepairException(AppException):
    default_message = "Repair operation failed"
    default_code = ErrorCode.REPAIR_ESTIMATION_FAILED
    default_category = ErrorCategory.REPAIR
    default_status_code = 409


class RepairNotFoundException(RepairException):
    default_message = "Repair record not found"
    default_code = ErrorCode.REPAIR_NOT_FOUND
    default_status_code = 404


class RepairEstimationException(RepairException):
    default_message = "Repair cost estimation failed"
    default_code = ErrorCode.REPAIR_ESTIMATION_FAILED
    default_status_code = 500


class RepairApprovalRequiredException(RepairException):
    default_message = "Repair approval is required"
    default_code = ErrorCode.REPAIR_APPROVAL_REQUIRED


# ============================================================
# Parts Exceptions
# ============================================================


class PartException(AppException):
    default_message = "Robot part operation failed"
    default_code = ErrorCode.PART_NOT_AVAILABLE
    default_category = ErrorCategory.PARTS
    default_status_code = 409


class PartNotFoundException(PartException):
    default_message = "Robot part not found"
    default_code = ErrorCode.PART_NOT_FOUND
    default_status_code = 404


class PartNotAvailableException(PartException):
    default_message = "Robot part is unavailable"


class PartCompatibilityException(PartException):
    default_message = "Robot part is incompatible"
    default_code = ErrorCode.PART_COMPATIBILITY_FAILED


# ============================================================
# Safety Exceptions
# ============================================================


class SafetyException(AppException):
    default_message = "Safety validation failed"
    default_code = ErrorCode.SAFETY_CHECK_FAILED
    default_category = ErrorCategory.SAFETY
    default_severity = ErrorSeverity.CRITICAL
    default_status_code = 409


class SafetyCheckFailedException(SafetyException):
    default_message = "Required safety checks did not pass"


class SafetyPolicyViolationException(SafetyException):
    default_message = "Safety policy violation"
    default_code = ErrorCode.SAFETY_POLICY_VIOLATION


class SafetyApprovalRequiredException(SafetyException):
    default_message = "Authorized safety approval is required"
    default_code = ErrorCode.SAFETY_APPROVAL_REQUIRED
    default_status_code = 403


class EmergencyStopActiveException(SafetyException):
    default_message = "Emergency stop is active"
    default_code = ErrorCode.EMERGENCY_STOP_ACTIVE


class UnsafeOperationException(SafetyException):
    default_message = "Operation rejected due to safety risk"
    default_code = ErrorCode.UNSAFE_OPERATION


class SafetyInterlockException(SafetyException):
    default_message = "Safety interlock prevents operation"
    default_code = ErrorCode.SAFETY_INTERLOCK_ACTIVE


# ============================================================
# Incident Exceptions
# ============================================================


class IncidentException(AppException):
    default_message = "Incident operation failed"
    default_code = ErrorCode.INCIDENT_OPERATION_FAILED
    default_category = ErrorCategory.INCIDENT
    default_status_code = 409


class IncidentNotFoundException(IncidentException):
    default_message = "Incident not found"
    default_code = ErrorCode.INCIDENT_NOT_FOUND
    default_status_code = 404


# ============================================================
# Storage Exceptions
# ============================================================


class StorageException(AppException):
    default_message = "Storage operation failed"
    default_code = ErrorCode.FILE_NOT_FOUND
    default_category = ErrorCategory.STORAGE
    default_status_code = 500


class FileNotFoundException(StorageException):
    default_message = "File not found"
    default_code = ErrorCode.FILE_NOT_FOUND
    default_status_code = 404


class FileTooLargeException(StorageException):
    default_message = "Uploaded file exceeds size limit"
    default_code = ErrorCode.FILE_TOO_LARGE
    default_status_code = 413


class FileTypeNotAllowedException(StorageException):
    default_message = "File type is not allowed"
    default_code = ErrorCode.FILE_TYPE_NOT_ALLOWED
    default_status_code = 415


class FileSecurityException(StorageException):
    default_message = "File failed security validation"
    default_code = ErrorCode.FILE_SECURITY_CHECK_FAILED
    default_status_code = 422


# ============================================================
# External Services
# ============================================================


class ExternalServiceException(AppException):
    default_message = "External service unavailable"
    default_code = ErrorCode.EXTERNAL_SERVICE_ERROR
    default_category = ErrorCategory.EXTERNAL_SERVICE
    default_status_code = 503


class ExternalServiceTimeoutException(ExternalServiceException):
    default_message = "External service timed out"
    default_code = ErrorCode.EXTERNAL_SERVICE_TIMEOUT
    default_status_code = 504


# ============================================================
# Infrastructure
# ============================================================


class ConfigurationException(AppException):
    default_message = "Application configuration error"
    default_code = ErrorCode.CONFIGURATION_ERROR
    default_category = ErrorCategory.INFRASTRUCTURE
    default_status_code = 500


class InfrastructureException(AppException):
    default_message = "Infrastructure operation failed"
    default_code = ErrorCode.INFRASTRUCTURE_ERROR
    default_category = ErrorCategory.INFRASTRUCTURE
    default_status_code = 503


# ============================================================
# Response Utilities
# ============================================================


def create_error_response(
    *,
    code: str,
    message: str,
    status_code: int,
    category: str = ErrorCategory.GENERAL.value,
    severity: str = ErrorSeverity.ERROR.value,
    request_id: str | None = None,
    error_id: str | None = None,
    details: Any = None,
    retryable: bool = False,
) -> JSONResponse:
    error: dict[str, Any] = {
        "error_id": error_id or generate_error_id(),
        "code": code,
        "message": message,
        "category": category,
        "severity": severity,
        "retryable": retryable,
    }

    if details is not None:
        error["details"] = details

    body: dict[str, Any] = {
        "success": False,
        "error": error,
        "timestamp": utc_now().isoformat(),
    }

    if request_id:
        body["request_id"] = request_id

    return JSONResponse(
        status_code=status_code,
        content=body,
    )


# ============================================================
# FastAPI Exception Handlers
# ============================================================


async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    request_id = get_request_id(request)

    log_message = (
        f"Application exception: "
        f"code={exc.code} "
        f"category={exc.category} "
        f"request_id={request_id} "
        f"error_id={exc.error_id}"
    )

    if exc.severity == ErrorSeverity.CRITICAL.value:
        logger.critical(log_message)
    elif exc.status_code >= 500:
        logger.error(log_message)
    else:
        logger.warning(log_message)

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(
            request_id=request_id,
            include_details=False,
        ),
    )


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = get_request_id(request)

    logger.warning(
        "Request validation failed: request_id=%s",
        request_id,
    )

    # Avoid returning raw validation inputs, which may
    # contain passwords, tokens, or sensitive user data.
    safe_errors: list[dict[str, Any]] = []

    for error in exc.errors():
        safe_errors.append(
            {
                "location": [
                    str(item)
                    for item in error.get("loc", ())
                ],
                "type": str(
                    error.get("type", "validation_error")
                ),
                "message": str(
                    error.get("msg", "Invalid value")
                ),
            }
        )

    return create_error_response(
        code=ErrorCode.VALIDATION_ERROR.value,
        message="Request validation failed",
        status_code=422,
        category=ErrorCategory.VALIDATION.value,
        request_id=request_id,
        details=safe_errors,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    request_id = get_request_id(request)

    status_code = exc.status_code

    public_messages = {
        400: "Bad request",
        401: "Authentication required",
        403: "Permission denied",
        404: "Resource not found",
        405: "Method not allowed",
        409: "Resource conflict",
        413: "Request payload too large",
        415: "Unsupported media type",
        422: "Unprocessable request",
        429: "Too many requests",
        500: "Internal server error",
        502: "Bad gateway",
        503: "Service unavailable",
        504: "Gateway timeout",
    }

    message = public_messages.get(
        status_code,
        "Request could not be completed",
    )

    return create_error_response(
        code=f"http_{status_code}",
        message=message,
        status_code=status_code,
        request_id=request_id,
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = get_request_id(request)
    error_id = generate_error_id()

    logger.error(
        "Unhandled application exception: "
        "request_id=%s error_id=%s",
        request_id,
        error_id,
        exc_info=(
            type(exc),
            exc,
            exc.__traceback__,
        ),
    )

    return create_error_response(
        code=ErrorCode.INTERNAL_ERROR.value,
        message="An unexpected server error occurred",
        status_code=500,
        request_id=request_id,
        error_id=error_id,
    )


# ============================================================
# Exception Handler Registration
# ============================================================


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register centralized exception handlers.

    Usage:
        app = FastAPI()
        register_exception_handlers(app)
    """

    app.add_exception_handler(
        AppException,
        app_exception_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        request_validation_exception_handler,
    )

    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )


# ============================================================
# Exception Utilities
# ============================================================


def ensure_condition(
    condition: bool,
    exception: AppException,
) -> None:
    """
    Raise an application exception if a condition is false.
    """

    if not condition:
        raise exception


def require_robot_available(
    *,
    robot_exists: bool,
    robot_online: bool,
    robot_faulted: bool,
    emergency_stop_active: bool,
) -> None:
    """
    Validate high-level robot availability.

    This is not a substitute for hardware safety validation.
    """

    if not robot_exists:
        raise RobotNotFoundException()

    if emergency_stop_active:
        raise EmergencyStopActiveException()

    if robot_faulted:
        raise RobotFaultException()

    if not robot_online:
        raise RobotOfflineException()


def require_safety_approval(
    *,
    approved: bool,
    safety_checks_passed: bool,
    interlocks_clear: bool,
) -> None:
    """
    Validate application-level safety prerequisites.

    Actual physical actuation must independently enforce
    hardware interlocks and authorized operator control.
    """

    if not interlocks_clear:
        raise SafetyInterlockException()

    if not safety_checks_passed:
        raise SafetyCheckFailedException()

    if not approved:
        raise SafetyApprovalRequiredException()


def require_tenant_access(
    *,
    authenticated_tenant_id: str,
    resource_tenant_id: str,
) -> None:
    if not authenticated_tenant_id:
        raise AuthenticationException()

    if authenticated_tenant_id != resource_tenant_id:
        raise TenantAccessDeniedException()


def require_permission(
    *,
    granted_permissions: set[str],
    required_permission: str,
) -> None:
    if required_permission not in granted_permissions:
        raise PermissionDeniedException()


def map_exception(
    exc: Exception,
) -> AppException:
    """
    Convert selected built-in exceptions into
    application exceptions.

    Do not return raw exception messages to clients.
    """

    if isinstance(exc, AppException):
        return exc

    if isinstance(exc, TimeoutError):
        return OperationTimeoutException(
            internal_message=safe_string(exc),
            retryable=True,
        )

    if isinstance(exc, PermissionError):
        return PermissionDeniedException(
            internal_message=safe_string(exc),
        )

    if isinstance(exc, FileNotFoundError):
        return FileNotFoundException(
            internal_message=safe_string(exc),
        )

    if isinstance(exc, (ValueError, TypeError)):
        return ApplicationValidationException(
            internal_message=safe_string(exc),
        )

    return AppException(
        internal_message=safe_string(exc),
    )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorCode",
    "AppException",
    "BadRequestException",
    "ResourceNotFoundException",
    "ResourceConflictException",
    "OperationTimeoutException",
    "ServiceUnavailableException",
    "RateLimitExceededException",
    "ApplicationValidationException",
    "AuthenticationException",
    "InvalidCredentialsException",
    "TokenExpiredException",
    "InvalidTokenException",
    "AccountLockedException",
    "AccountDisabledException",
    "MFARequiredException",
    "PermissionDeniedException",
    "TenantAccessDeniedException",
    "TenantNotFoundException",
    "DatabaseException",
    "DatabaseConnectionException",
    "DatabaseIntegrityException",
    "TransactionException",
    "CacheException",
    "CacheConnectionException",
    "CacheLockException",
    "EventBusException",
    "EventPublishException",
    "EventHandlerException",
    "EventQueueFullException",
    "RobotException",
    "RobotNotFoundException",
    "RobotOfflineException",
    "RobotUnavailableException",
    "RobotFaultException",
    "RobotBusyException",
    "RobotEmergencyStoppedException",
    "RobotCommandRejectedException",
    "RobotControlUnauthorizedException",
    "FleetException",
    "FleetNotFoundException",
    "SensorException",
    "SensorNotFoundException",
    "SensorFailureException",
    "SensorCalibrationRequiredException",
    "SensorDataInvalidException",
    "TelemetryException",
    "TelemetryInvalidException",
    "TelemetryStaleException",
    "TelemetryIngestionException",
    "NavigationException",
    "NavigationTimeoutException",
    "NavigationPathNotFoundException",
    "NavigationCollisionRiskException",
    "ManipulationException",
    "GraspFailedException",
    "ManipulationWorkspaceException",
    "PerceptionException",
    "LowPerceptionConfidenceException",
    "ModelException",
    "ModelNotFoundException",
    "ModelLoadException",
    "ModelInferenceException",
    "ModelDeploymentRejectedException",
    "ModelValidationException",
    "WorldModelException",
    "WorldModelPredictionException",
    "VLAException",
    "VLAInferenceException",
    "VLAActionRejectedException",
    "RAGException",
    "RAGRetrievalException",
    "RAGGroundingException",
    "SimulationException",
    "SimulationTimeoutException",
    "SimulationEngineUnavailableException",
    "EvaluationException",
    "BenchmarkException",
    "EvaluationThresholdException",
    "MaintenanceException",
    "MaintenanceRequiredException",
    "FailurePredictionException",
    "RepairException",
    "RepairNotFoundException",
    "RepairEstimationException",
    "RepairApprovalRequiredException",
    "PartException",
    "PartNotFoundException",
    "PartNotAvailableException",
    "PartCompatibilityException",
    "SafetyException",
    "SafetyCheckFailedException",
    "SafetyPolicyViolationException",
    "SafetyApprovalRequiredException",
    "EmergencyStopActiveException",
    "UnsafeOperationException",
    "SafetyInterlockException",
    "IncidentException",
    "IncidentNotFoundException",
    "StorageException",
    "FileNotFoundException",
    "FileTooLargeException",
    "FileTypeNotAllowedException",
    "FileSecurityException",
    "ExternalServiceException",
    "ExternalServiceTimeoutException",
    "ConfigurationException",
    "InfrastructureException",
    "create_error_response",
    "app_exception_handler",
    "request_validation_exception_handler",
    "http_exception_handler",
    "unhandled_exception_handler",
    "register_exception_handlers",
    "ensure_condition",
    "require_robot_available",
    "require_safety_approval",
    "require_tenant_access",
    "require_permission",
    "map_exception",
]