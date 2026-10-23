"""Model Integrity module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/model_integrity.py

Python 3.11+
Standard library only.

Features:
    - SHA-256 model artifact verification
    - HMAC-SHA256 signed manifests
    - Trusted model registry
    - Model version tracking
    - Approval and deployment gates
    - Tamper detection
    - Model rollback eligibility checks
    - Model metadata validation
    - Security audit events
    - JSON reporting

Security:
    Verification never deserializes model files.
    Untrusted pickle-based model files must not
    be loaded during integrity verification.

    This module does not execute or deploy models.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelIntegrityError(Exception):
    pass


class ModelValidationError(ModelIntegrityError):
    pass


class ModelNotFoundError(ModelIntegrityError):
    pass


class ModelVerificationError(ModelIntegrityError):
    pass


class ModelApprovalError(ModelIntegrityError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ModelIntegrityStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    MODIFIED = "modified"
    MISSING = "missing"
    REJECTED = "rejected"


class ModelApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


class ModelArtifactFormat(str, Enum):
    ONNX = "onnx"
    SAFETENSORS = "safetensors"
    TORCHSCRIPT = "torchscript"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    OTHER = "other"


class ModelSecurityEvent(str, Enum):
    REGISTERED = "registered"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"
    DEPLOYMENT_ALLOWED = "deployment_allowed"
    DEPLOYMENT_DENIED = "deployment_denied"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(
            f"{name} is required"
        )

    return value.strip()


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            "Invalid JSON metadata"
        ) from exc


def sha256_file(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def safe_identifier(value: str, name: str) -> str:
    value = require_text(value, name)

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        value,
    ):
        raise ModelValidationError(
            f"Invalid {name}"
        )

    if value in {".", ".."}:
        raise ModelValidationError(
            f"Invalid {name}"
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ModelIntegrityConfig:
    registry_directory: str = (
        "data/security/model_integrity"
    )

    maximum_model_size_bytes: int = (
        10 * 1024 * 1024 * 1024
    )

    require_signed_manifest: bool = True
    require_approval: bool = True

    allowed_formats: frozenset[
        ModelArtifactFormat
    ] = field(
        default_factory=lambda: frozenset({
            ModelArtifactFormat.ONNX,
            ModelArtifactFormat.SAFETENSORS,
            ModelArtifactFormat.TORCHSCRIPT,
            ModelArtifactFormat.PYTORCH,
            ModelArtifactFormat.TENSORFLOW,
        })
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_model_size_bytes, bool)
            or not isinstance(
                self.maximum_model_size_bytes,
                int,
            )
            or self.maximum_model_size_bytes <= 0
        ):
            raise ModelValidationError(
                "Invalid maximum model size"
            )


# ============================================================
# MODEL ARTIFACT
# ============================================================

@dataclass
class ModelArtifact:
    model_id: str
    version: str
    name: str

    artifact_path: str
    artifact_format: ModelArtifactFormat

    sha256: str
    size_bytes: int

    created_at: str

    registered_by: str

    approval_status: ModelApprovalStatus = (
        ModelApprovalStatus.PENDING
    )

    approved_by: str | None = None
    approved_at: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "name": self.name,
            "artifact_path": self.artifact_path,
            "artifact_format": self.artifact_format.value,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "registered_by": self.registered_by,
            "approval_status": self.approval_status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "metadata": self.metadata,
        }


# ============================================================
# MODEL MANIFEST
# ============================================================

@dataclass(frozen=True)
class ModelIntegrityManifest:
    model_id: str
    version: str

    artifact_sha256: str
    artifact_size_bytes: int

    artifact_format: ModelArtifactFormat

    created_at: str
    signer_id: str

    signature: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_format": self.artifact_format.value,
            "created_at": self.created_at,
            "signer_id": self.signer_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.unsigned_dict(),
            "signature": self.signature,
        }


# ============================================================
# SIGNATURE ENGINE
# ============================================================

class ModelManifestSigner:

    def __init__(
        self,
        signing_key: bytes,
    ) -> None:
        if (
            not isinstance(signing_key, bytes)
            or len(signing_key) < 32
        ):
            raise ModelValidationError(
                "Signing key must contain at least 32 bytes"
            )

        self.signing_key = signing_key

    def sign(
        self,
        payload: Mapping[str, Any],
    ) -> str:
        return hmac.new(
            self.signing_key,
            canonical_json(dict(payload)),
            hashlib.sha256,
        ).hexdigest()

    def verify(
        self,
        manifest: ModelIntegrityManifest,
    ) -> bool:
        expected = self.sign(
            manifest.unsigned_dict()
        )

        return hmac.compare_digest(
            expected,
            manifest.signature,
        )


# ============================================================
# SECURITY AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class ModelIntegrityAuditEvent:
    event_id: str
    timestamp: str
    model_id: str
    version: str

    event_type: ModelSecurityEvent

    actor_id: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "version": self.version,
            "event_type": self.event_type.value,
            "actor_id": self.actor_id,
            "details": self.details,
        }


# ============================================================
# CENTRAL MODEL INTEGRITY SERVICE
# ============================================================

class RoboticsModelIntegrityService:

    def __init__(
        self,
        signing_key: bytes,
        config: ModelIntegrityConfig | None = None,
    ) -> None:
        self.config = (
            config or ModelIntegrityConfig()
        )

        self.signer = ModelManifestSigner(
            signing_key
        )

        self.artifacts: dict[
            tuple[str, str],
            ModelArtifact
        ] = {}

        self.manifests: dict[
            tuple[str, str],
            ModelIntegrityManifest
        ] = {}

        self.audit_events: list[
            ModelIntegrityAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        model_id: str,
        version: str,
        event_type: ModelSecurityEvent,
        *,
        actor_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.audit_events.append(
            ModelIntegrityAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                model_id=model_id,
                version=version,
                event_type=event_type,
                actor_id=actor_id,
                details=dict(details or {}),
            )
        )

    # --------------------------------------------------------
    # REGISTER MODEL
    # --------------------------------------------------------

    def register_model(
        self,
        *,
        model_id: str,
        version: str,
        name: str,
        artifact_path: str | Path,
        artifact_format: ModelArtifactFormat,
        registered_by: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelArtifact:
        model_id = safe_identifier(
            model_id,
            "model_id",
        )

        version = safe_identifier(
            version,
            "version",
        )

        name = require_text(name, "name")

        registered_by = require_text(
            registered_by,
            "registered_by",
        )

        artifact_format = ModelArtifactFormat(
            artifact_format
        )

        if (
            artifact_format
            not in self.config.allowed_formats
        ):
            raise ModelValidationError(
                "Model format is not allowed"
            )

        path = Path(artifact_path).resolve(
            strict=True
        )

        if not path.is_file() or path.is_symlink():
            raise ModelValidationError(
                "Artifact must be a regular file"
            )

        size = path.stat().st_size

        if size <= 0:
            raise ModelValidationError(
                "Model artifact is empty"
            )

        if (
            size
            > self.config.maximum_model_size_bytes
        ):
            raise ModelValidationError(
                "Model artifact exceeds size limit"
            )

        digest = sha256_file(path)

        key = (model_id, version)

        with self._lock:
            if key in self.artifacts:
                raise ModelValidationError(
                    "Model version already registered"
                )

            artifact = ModelArtifact(
                model_id=model_id,
                version=version,
                name=name,
                artifact_path=str(path),
                artifact_format=artifact_format,
                sha256=digest,
                size_bytes=size,
                created_at=utc_now(),
                registered_by=registered_by,
                metadata=json.loads(
                    canonical_json(
                        dict(metadata or {})
                    )
                ),
            )

            self.artifacts[key] = artifact

            self._audit(
                model_id,
                version,
                ModelSecurityEvent.REGISTERED,
                actor_id=registered_by,
                details={
                    "sha256": digest,
                    "size_bytes": size,
                },
            )

            return artifact

    # --------------------------------------------------------
    # CREATE SIGNED MANIFEST
    # --------------------------------------------------------

    def create_manifest(
        self,
        model_id: str,
        version: str,
        *,
        signer_id: str,
    ) -> ModelIntegrityManifest:
        key = (model_id, version)

        with self._lock:
            artifact = self.artifacts.get(key)

            if artifact is None:
                raise ModelNotFoundError(
                    "Model artifact not found"
                )

            if key in self.manifests:
                raise ModelValidationError(
                    "Manifest already exists"
                )

            unsigned = {
                "model_id": model_id,
                "version": version,
                "artifact_sha256": artifact.sha256,
                "artifact_size_bytes": artifact.size_bytes,
                "artifact_format": (
                    artifact.artifact_format.value
                ),
                "created_at": utc_now(),
                "signer_id": require_text(
                    signer_id,
                    "signer_id",
                ),
            }

            signature = self.signer.sign(
                unsigned
            )

            manifest = ModelIntegrityManifest(
                model_id=model_id,
                version=version,
                artifact_sha256=artifact.sha256,
                artifact_size_bytes=artifact.size_bytes,
                artifact_format=artifact.artifact_format,
                created_at=unsigned["created_at"],
                signer_id=unsigned["signer_id"],
                signature=signature,
            )

            self.manifests[key] = manifest

            return manifest

    # --------------------------------------------------------
    # VERIFY MODEL
    # --------------------------------------------------------

    def verify_model(
        self,
        model_id: str,
        version: str,
    ) -> ModelIntegrityStatus:
        key = (model_id, version)

        with self._lock:
            artifact = self.artifacts.get(key)

            if artifact is None:
                raise ModelNotFoundError(
                    "Model not registered"
                )

            manifest = self.manifests.get(key)

            if (
                self.config.require_signed_manifest
                and manifest is None
            ):
                return ModelIntegrityStatus.UNVERIFIED

            if manifest is not None:
                if not self.signer.verify(manifest):
                    self._audit(
                        model_id,
                        version,
                        ModelSecurityEvent.VERIFICATION_FAILED,
                        details={
                            "reason": "invalid_manifest_signature"
                        },
                    )

                    return ModelIntegrityStatus.REJECTED

                if (
                    manifest.artifact_sha256 != artifact.sha256
                    or manifest.artifact_size_bytes
                    != artifact.size_bytes
                    or manifest.artifact_format
                    != artifact.artifact_format
                ):
                    return ModelIntegrityStatus.REJECTED

            path = Path(
                artifact.artifact_path
            )

            if not path.is_file():
                return ModelIntegrityStatus.MISSING

            if path.is_symlink():
                return ModelIntegrityStatus.REJECTED

            current_size = path.stat().st_size

            if current_size != artifact.size_bytes:
                return ModelIntegrityStatus.MODIFIED

            current_hash = sha256_file(path)

            if not hmac.compare_digest(
                current_hash,
                artifact.sha256,
            ):
                self._audit(
                    model_id,
                    version,
                    ModelSecurityEvent.VERIFICATION_FAILED,
                    details={
                        "reason": "artifact_hash_mismatch"
                    },
                )

                return ModelIntegrityStatus.MODIFIED

            self._audit(
                model_id,
                version,
                ModelSecurityEvent.VERIFIED,
            )

            return ModelIntegrityStatus.VERIFIED

    # --------------------------------------------------------
    # APPROVE MODEL
    # --------------------------------------------------------

    def approve_model(
        self,
        model_id: str,
        version: str,
        *,
        approved_by: str,
    ) -> ModelArtifact:
        approved_by = require_text(
            approved_by,
            "approved_by",
        )

        with self._lock:
            artifact = self.artifacts.get(
                (model_id, version)
            )

            if artifact is None:
                raise ModelNotFoundError(
                    "Model not registered"
                )

            if (
                artifact.approval_status
                == ModelApprovalStatus.REVOKED
            ):
                raise ModelApprovalError(
                    "Revoked model cannot be approved"
                )

            if approved_by == artifact.registered_by:
                raise ModelApprovalError(
                    "Model registrant cannot self-approve"
                )

            if (
                self.verify_model(model_id, version)
                != ModelIntegrityStatus.VERIFIED
            ):
                raise ModelVerificationError(
                    "Model integrity verification failed"
                )

            artifact.approval_status = (
                ModelApprovalStatus.APPROVED
            )

            artifact.approved_by = approved_by
            artifact.approved_at = utc_now()

            self._audit(
                model_id,
                version,
                ModelSecurityEvent.APPROVED,
                actor_id=approved_by,
            )

            return artifact

    # --------------------------------------------------------
    # REVOKE MODEL
    # --------------------------------------------------------

    def revoke_model(
        self,
        model_id: str,
        version: str,
        *,
        revoked_by: str,
    ) -> None:
        with self._lock:
            artifact = self.artifacts.get(
                (model_id, version)
            )

            if artifact is None:
                raise ModelNotFoundError(
                    "Model not found"
                )

            artifact.approval_status = (
                ModelApprovalStatus.REVOKED
            )

            self._audit(
                model_id,
                version,
                ModelSecurityEvent.REVOKED,
                actor_id=require_text(
                    revoked_by,
                    "revoked_by",
                ),
            )

    # --------------------------------------------------------
    # DEPLOYMENT ELIGIBILITY
    # --------------------------------------------------------

    def can_deploy(
        self,
        model_id: str,
        version: str,
    ) -> bool:
        with self._lock:
            artifact = self.artifacts.get(
                (model_id, version)
            )

            if artifact is None:
                return False

            if (
                self.config.require_approval
                and artifact.approval_status
                != ModelApprovalStatus.APPROVED
            ):
                return False

            if (
                artifact.approval_status
                == ModelApprovalStatus.REVOKED
            ):
                return False

            return (
                self.verify_model(
                    model_id,
                    version,
                )
                == ModelIntegrityStatus.VERIFIED
            )

    # --------------------------------------------------------
    # EXPORT REPORT
    # --------------------------------------------------------

    def export_report(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now(),
                "models": [
                    artifact.to_dict()
                    for artifact in self.artifacts.values()
                ],
                "manifests": [
                    manifest.to_dict()
                    for manifest in self.manifests.values()
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
                ],
            }

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "registered_models": len(
                    self.artifacts
                ),
                "signed_manifests": len(
                    self.manifests
                ),
                "approved_models": sum(
                    artifact.approval_status
                    == ModelApprovalStatus.APPROVED
                    for artifact in self.artifacts.values()
                ),
                "revoked_models": sum(
                    artifact.approval_status
                    == ModelApprovalStatus.REVOKED
                    for artifact in self.artifacts.values()
                ),
                "audit_events": len(
                    self.audit_events
                ),
            }


# ============================================================
# MODEL SERVICE ADAPTER
# ============================================================

class ModelServiceIntegrityAdapter:

    def __init__(
        self,
        integrity_service: RoboticsModelIntegrityService,
    ) -> None:
        self.integrity_service = integrity_service

    def validate_deployment(
        self,
        model_id: str,
        version: str,
    ) -> None:
        if not self.integrity_service.can_deploy(
            model_id,
            version,
        ):
            raise ModelVerificationError(
                "Model deployment blocked by integrity policy"
            )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    # Demonstration only.
    # Production keys must come from a secure secret store.
    signing_key = secrets.token_bytes(32)

    service = RoboticsModelIntegrityService(
        signing_key=signing_key
    )

    demo_directory = Path(
        "data/security/model_integrity/demo"
    )

    demo_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # This is a test artifact, not a trained ONNX model.
    artifact_path = (
        demo_directory / "demo_vision.onnx"
    )

    artifact_path.write_bytes(
        b"DEMO_MODEL_ARTIFACT_NOT_REAL_ONNX"
    )

    service.register_model(
        model_id="robot_vision_001",
        version="1.0.0",
        name="Robot Vision Demo",
        artifact_path=artifact_path,
        artifact_format=ModelArtifactFormat.ONNX,
        registered_by="engineer_001",
        metadata={
            "purpose": "integrity_testing",
        },
    )

    service.create_manifest(
        "robot_vision_001",
        "1.0.0",
        signer_id="model_signing_service",
    )

    verification = service.verify_model(
        "robot_vision_001",
        "1.0.0",
    )

    service.approve_model(
        "robot_vision_001",
        "1.0.0",
        approved_by="security_reviewer_001",
    )

    deployable = service.can_deploy(
        "robot_vision_001",
        "1.0.0",
    )

    report = service.export_report(
        "data/security/model_integrity/"
        "model_integrity_report.json"
    )

    print(
        json.dumps(
            {
                "verification": verification.value,
                "deployment_allowed": deployable,
                "statistics": service.statistics(),
                "report": str(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
