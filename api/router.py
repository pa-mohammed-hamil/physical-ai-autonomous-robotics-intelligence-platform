"""Router module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Central FastAPI Router

Features:
- API v1 routing
- Modular endpoint registration
- Authentication and authorization integration
- Robot and fleet management
- Telemetry and sensor monitoring
- Predictive maintenance and repair
- AI assistant, RAG and agents
- Navigation and manipulation analysis
- VLA and world models
- Simulation and evaluation
- Model registry and MLOps
- Tenant-isolated endpoints
- Health and readiness checks

Security:
- Protected routes require authentication.
- Physical robot control is not exposed here.
- API routes must enforce RBAC and tenant isolation.
- Missing required routers fail application startup.
"""

from __future__ import annotations

import importlib
import logging

from dataclasses import dataclass
from typing import Any, Iterable

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import (
    get_tenant_context,
    require_permission,
)
from app.api.response_models import (
    MessageResponse,
    ResponseMetadata,
)


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================


API_PREFIX = "/api/v1"

API_VERSION = "v1"

API_TAGS_METADATA = [
    {
        "name": "System",
        "description": "API status and service information",
    },
    {
        "name": "Authentication",
        "description": "Authentication and session management",
    },
    {
        "name": "Users",
        "description": "User management",
    },
    {
        "name": "Organizations",
        "description": "Tenant and organization management",
    },
    {
        "name": "Robots",
        "description": "Robot inventory and status",
    },
    {
        "name": "Fleets",
        "description": "Robot fleet management",
    },
    {
        "name": "Sensors",
        "description": "Sensor health and diagnostics",
    },
    {
        "name": "Telemetry",
        "description": "Robot telemetry analytics",
    },
    {
        "name": "Maintenance",
        "description": "Predictive maintenance",
    },
    {
        "name": "Repairs",
        "description": "Repair and replacement intelligence",
    },
    {
        "name": "Parts",
        "description": "Robot parts and inventory",
    },
    {
        "name": "Perception",
        "description": "Computer vision and perception analysis",
    },
    {
        "name": "Navigation",
        "description": "Navigation planning and analysis",
    },
    {
        "name": "Manipulation",
        "description": "Manipulation planning and simulation",
    },
    {
        "name": "VLA",
        "description": "Vision-language-action research",
    },
    {
        "name": "World Models",
        "description": "Neural world model inference",
    },
    {
        "name": "Simulation",
        "description": "Offline robotic simulations",
    },
    {
        "name": "Evaluation",
        "description": "AI model and robotics evaluation",
    },
    {
        "name": "Models",
        "description": "Model registry and inference",
    },
    {
        "name": "MLOps",
        "description": "Machine learning operations",
    },
    {
        "name": "Assistant",
        "description": "AI assistant and conversations",
    },
    {
        "name": "RAG",
        "description": "Retrieval-augmented generation",
    },
    {
        "name": "Agents",
        "description": "Autonomous analytical agents",
    },
    {
        "name": "Documents",
        "description": "Document intelligence",
    },
    {
        "name": "Analytics",
        "description": "Operational analytics",
    },
    {
        "name": "Safety",
        "description": "Safety monitoring and alerts",
    },
    {
        "name": "Audit",
        "description": "Security and audit events",
    },
    {
        "name": "Settings",
        "description": "Application configuration",
    },
]


# ============================================================
# Router Definitions
# ============================================================


@dataclass(frozen=True)
class RouterDefinition:
    module: str
    prefix: str
    tag: str

    required: bool = True
    protected: bool = True

    def __post_init__(self) -> None:
        if not self.module:
            raise ValueError(
                "Router module is required"
            )

        if not self.prefix.startswith("/"):
            raise ValueError(
                "Router prefix must start with '/'"
            )

        if not self.tag:
            raise ValueError(
                "Router tag is required"
            )


ROUTER_DEFINITIONS = (
    RouterDefinition(
        module="auth",
        prefix="/auth",
        tag="Authentication",
        protected=False,
    ),
    RouterDefinition(
        module="users",
        prefix="/users",
        tag="Users",
    ),
    RouterDefinition(
        module="organizations",
        prefix="/organizations",
        tag="Organizations",
    ),
    RouterDefinition(
        module="robots",
        prefix="/robots",
        tag="Robots",
    ),
    RouterDefinition(
        module="fleets",
        prefix="/fleets",
        tag="Fleets",
    ),
    RouterDefinition(
        module="sensors",
        prefix="/sensors",
        tag="Sensors",
    ),
    RouterDefinition(
        module="telemetry",
        prefix="/telemetry",
        tag="Telemetry",
    ),
    RouterDefinition(
        module="maintenance",
        prefix="/maintenance",
        tag="Maintenance",
    ),
    RouterDefinition(
        module="repairs",
        prefix="/repairs",
        tag="Repairs",
    ),
    RouterDefinition(
        module="parts",
        prefix="/parts",
        tag="Parts",
    ),
    RouterDefinition(
        module="perception",
        prefix="/perception",
        tag="Perception",
    ),
    RouterDefinition(
        module="navigation",
        prefix="/navigation",
        tag="Navigation",
    ),
    RouterDefinition(
        module="manipulation",
        prefix="/manipulation",
        tag="Manipulation",
    ),
    RouterDefinition(
        module="vla",
        prefix="/vla",
        tag="VLA",
    ),
    RouterDefinition(
        module="world_models",
        prefix="/world-models",
        tag="World Models",
    ),
    RouterDefinition(
        module="simulation",
        prefix="/simulation",
        tag="Simulation",
    ),
    RouterDefinition(
        module="evaluation",
        prefix="/evaluation",
        tag="Evaluation",
    ),
    RouterDefinition(
        module="models",
        prefix="/models",
        tag="Models",
    ),
    RouterDefinition(
        module="mlops",
        prefix="/mlops",
        tag="MLOps",
    ),
    RouterDefinition(
        module="assistant",
        prefix="/assistant",
        tag="Assistant",
    ),
    RouterDefinition(
        module="rag",
        prefix="/rag",
        tag="RAG",
    ),
    RouterDefinition(
        module="agents",
        prefix="/agents",
        tag="Agents",
    ),
    RouterDefinition(
        module="documents",
        prefix="/documents",
        tag="Documents",
    ),
    RouterDefinition(
        module="analytics",
        prefix="/analytics",
        tag="Analytics",
    ),
    RouterDefinition(
        module="safety",
        prefix="/safety",
        tag="Safety",
    ),
    RouterDefinition(
        module="audit",
        prefix="/audit",
        tag="Audit",
    ),
    RouterDefinition(
        module="settings",
        prefix="/settings",
        tag="Settings",
    ),
)


# ============================================================
# Router Loading
# ============================================================


class RouterConfigurationError(RuntimeError):
    """Invalid or missing API router configuration."""


def load_router(
    definition: RouterDefinition,
) -> APIRouter:
    module_path = (
        f"app.api.endpoints.{definition.module}"
    )

    try:
        module = importlib.import_module(
            module_path
        )

    except ImportError as exc:
        raise RouterConfigurationError(
            f"Required API module could not be loaded: "
            f"{module_path}"
        ) from exc

    router = getattr(
        module,
        "router",
        None,
    )

    if not isinstance(router, APIRouter):
        raise RouterConfigurationError(
            f"{module_path} must export an APIRouter "
            "instance named 'router'"
        )

    return router


def register_routers(
    parent_router: APIRouter,
    *,
    definitions: Iterable[
        RouterDefinition
    ] = ROUTER_DEFINITIONS,
) -> None:
    registered_prefixes: set[str] = set()

    for definition in definitions:
        if definition.prefix in registered_prefixes:
            raise RouterConfigurationError(
                f"Duplicate router prefix: "
                f"{definition.prefix}"
            )

        registered_prefixes.add(
            definition.prefix
        )

        try:
            child_router = load_router(
                definition
            )

        except RouterConfigurationError:
            if definition.required:
                raise

            logger.warning(
                "Optional router unavailable: %s",
                definition.module,
            )
            continue

        dependencies = []

        if definition.protected:
            dependencies.append(
                Depends(get_tenant_context)
            )

        parent_router.include_router(
            child_router,
            prefix=definition.prefix,
            tags=[definition.tag],
            dependencies=dependencies,
        )

        logger.info(
            "Registered API router: %s",
            definition.prefix,
        )


# ============================================================
# Main API Router
# ============================================================


api_router = APIRouter(
    prefix=API_PREFIX,
)


# ============================================================
# System Endpoints
# ============================================================


@api_router.get(
    "/",
    tags=["System"],
    response_model=MessageResponse,
    summary="API information",
)
async def api_information() -> MessageResponse:
    return MessageResponse(
        message=(
            "Physical AI & Autonomous Robotics "
            "Intelligence Platform API"
        ),
        metadata=ResponseMetadata(
            api_version=API_VERSION
        ),
    )


@api_router.get(
    "/status",
    tags=["System"],
    summary="API status",
)
async def api_status() -> dict[str, Any]:
    return {
        "success": True,
        "service": (
            "physical-ai-autonomous-robotics-intelligence"
        ),
        "api_version": API_VERSION,
        "status": "available",
    }


# ============================================================
# Register Feature Routers
# ============================================================


def build_api_router(
    *,
    definitions: Iterable[
        RouterDefinition
    ] = ROUTER_DEFINITIONS,
) -> APIRouter:
    """
    Build an API router with the supplied modules.

    Required modules must exist before application
    startup. This prevents silently missing routes.
    """

    router = APIRouter(
        prefix=API_PREFIX,
    )

    router.add_api_route(
        "/",
        api_information,
        methods=["GET"],
        tags=["System"],
        response_model=MessageResponse,
    )

    router.add_api_route(
        "/status",
        api_status,
        methods=["GET"],
        tags=["System"],
    )

    register_routers(
        router,
        definitions=definitions,
    )

    return router


# ============================================================
# API Installation
# ============================================================


def include_api_router(
    app: Any,
    *,
    definitions: Iterable[
        RouterDefinition
    ] = ROUTER_DEFINITIONS,
) -> None:
    """
    Install the central API router.

    Example:

        from fastapi import FastAPI
        from app.api.router import include_api_router

        app = FastAPI()
        include_api_router(app)
    """

    router = build_api_router(
        definitions=definitions,
    )

    app.include_router(router)


# ============================================================
# Exports
# ============================================================


__all__ = [
    "API_PREFIX",
    "API_VERSION",
    "API_TAGS_METADATA",
    "RouterDefinition",
    "RouterConfigurationError",
    "ROUTER_DEFINITIONS",
    "api_router",
    "api_information",
    "api_status",
    "load_router",
    "register_routers",
    "build_api_router",
    "include_api_router",
]