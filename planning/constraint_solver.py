# backend/app/robotics/planning/constraint_solver.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: constraint_solver.py

Constraint-based task planning and scheduling for autonomous
robotics, industrial inspection, maintenance, and simulation.

Python 3.11+
Dependencies: Standard library only.

Features
--------
- Constraint satisfaction problems (CSP)
- Backtracking search
- Minimum Remaining Values (MRV)
- Degree heuristic
- Least Constraining Value (LCV)
- Forward checking
- AC-3 arc consistency
- Hard and soft constraints
- Weighted constraint optimization
- Robot capability matching
- Task dependency constraints
- Resource allocation
- Time-window scheduling
- Battery and energy constraints
- Workspace and zone restrictions
- Human approval requirements
- Safety validation
- Multi-robot task assignment
- Constraint violation reporting
- Solver statistics and diagnostics
- JSON export
- Unit tests

Safety
------
This module generates planning recommendations only.

It does not issue commands to robots, actuators, motors,
navigation systems, or emergency-stop equipment.

Real-world execution requires an independently authorized
robot controller with hardware-enforced safety interlocks.

Constraint feasibility is not proof of physical safety.
"""

from __future__ import annotations

import json
import math
import time
import unittest
import uuid

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations
from threading import RLock
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ConstraintSolverError(Exception):
    """Base constraint solver exception."""


class ConstraintValidationError(ConstraintSolverError):
    """Invalid constraint or variable configuration."""


class ConstraintAuthorizationError(ConstraintSolverError):
    """Unauthorized planning operation."""


class ConstraintTimeoutError(ConstraintSolverError):
    """Constraint solving exceeded the permitted duration."""


class ConstraintUnsatisfiableError(ConstraintSolverError):
    """No assignment satisfies the hard constraints."""


class ConstraintSecurityError(ConstraintSolverError):
    """Invalid or unsafe planning input."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstraintValidationError(
            f"{field_name} must be a nonempty string"
        )

    if len(value) > 128:
        raise ConstraintValidationError(
            f"{field_name} exceeds 128 characters"
        )

    return value.strip()


def finite_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ConstraintValidationError(
            f"{field_name} must be finite"
        )

    result = float(value)

    if minimum is not None and result < minimum:
        raise ConstraintValidationError(
            f"{field_name} must be >= {minimum}"
        )

    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: json_safe(getattr(value, key))
            for key in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [json_safe(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return repr(value)


# ============================================================
# ENUMS
# ============================================================

class ConstraintType(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class ConstraintCategory(str, Enum):
    GENERAL = "general"
    SAFETY = "safety"
    CAPABILITY = "capability"
    BATTERY = "battery"
    RESOURCE = "resource"
    TIME_WINDOW = "time_window"
    DEPENDENCY = "dependency"
    WORKSPACE = "workspace"
    HUMAN_APPROVAL = "human_approval"
    COST = "cost"
    PRIORITY = "priority"


class SolverStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    UNSATISFIABLE = "unsatisfiable"
    TIMEOUT = "timeout"
    NODE_LIMIT = "node_limit"
    CANCELLED = "cancelled"
    INVALID = "invalid"


class VariableType(str, Enum):
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


class RobotAvailability(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    FAULTED = "faulted"


class RobotSafetyState(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


class OptimizationDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ConstraintSolverConfig:
    maximum_variables: int = 200
    maximum_domain_size: int = 1000
    maximum_constraints: int = 5000

    maximum_search_nodes: int = 200_000
    maximum_solutions: int = 100

    timeout_seconds: float = 30.0

    use_mrv: bool = True
    use_degree_heuristic: bool = True
    use_lcv: bool = True

    enable_forward_checking: bool = True
    enable_arc_consistency: bool = True

    optimize_soft_constraints: bool = True

    maximum_violation_reports: int = 100

    def __post_init__(self) -> None:
        for name in (
            "maximum_variables",
            "maximum_domain_size",
            "maximum_constraints",
            "maximum_search_nodes",
            "maximum_solutions",
            "maximum_violation_reports",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ConstraintValidationError(
                    f"{name} must be a positive integer"
                )

        finite_number(
            self.timeout_seconds,
            "timeout_seconds",
            minimum=0.001,
        )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class PlanningAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    roles: frozenset[str] = frozenset()

    can_plan: bool = False
    can_modify_constraints: bool = False
    can_view_results: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# CSP VARIABLES
# ============================================================

@dataclass(frozen=True)
class ConstraintVariable:
    name: str
    domain: tuple[Any, ...]

    variable_type: VariableType = VariableType.CATEGORICAL

    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.name, "variable name")

        if not self.domain:
            raise ConstraintValidationError(
                f"Variable {self.name} has an empty domain"
            )

        for value in self.domain:
            try:
                hash(value)
            except TypeError as exc:
                raise ConstraintValidationError(
                    f"Variable {self.name} contains "
                    "an unhashable domain value"
                ) from exc

            if (
                self.variable_type == VariableType.INTEGER
                and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                )
            ):
                raise ConstraintValidationError(
                    f"Variable {self.name} requires integer values"
                )

            if (
                self.variable_type == VariableType.BOOLEAN
                and not isinstance(value, bool)
            ):
                raise ConstraintValidationError(
                    f"Variable {self.name} requires boolean values"
                )

        if len(set(self.domain)) != len(self.domain):
            raise ConstraintValidationError(
                f"Variable {self.name} has duplicate domain values"
            )


# ============================================================
# CONSTRAINT DEFINITIONS
# ============================================================

ConstraintPredicate = Callable[
    [Mapping[str, Any]],
    bool,
]


@dataclass(frozen=True)
class Constraint:
    constraint_id: str
    name: str

    variables: tuple[str, ...]

    predicate: ConstraintPredicate

    constraint_type: ConstraintType = ConstraintType.HARD

    category: ConstraintCategory = ConstraintCategory.GENERAL

    weight: float = 1.0

    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(
            self.constraint_id,
            "constraint_id",
        )

        validate_identifier(
            self.name,
            "constraint name",
        )

        if not self.variables:
            raise ConstraintValidationError(
                "Constraint requires at least one variable"
            )

        if len(set(self.variables)) != len(self.variables):
            raise ConstraintValidationError(
                "Constraint variables must be unique"
            )

        if not callable(self.predicate):
            raise ConstraintValidationError(
                "Constraint predicate must be callable"
            )

        finite_number(
            self.weight,
            "constraint weight",
            minimum=0,
        )

    def is_ready(
        self,
        assignment: Mapping[str, Any],
    ) -> bool:
        return all(
            variable in assignment
            for variable in self.variables
        )

    def evaluate(
        self,
        assignment: Mapping[str, Any],
    ) -> bool | None:
        if not self.is_ready(assignment):
            return None

        scoped_assignment = {
            variable: assignment[variable]
            for variable in self.variables
        }

        try:
            result = self.predicate(scoped_assignment)
        except Exception as exc:
            raise ConstraintSolverError(
                f"Constraint {self.name} evaluation failed: {exc}"
            ) from exc

        if not isinstance(result, bool):
            raise ConstraintValidationError(
                f"Constraint {self.name} must return bool"
            )

        return result


# ============================================================
# CONSTRAINT VIOLATION
# ============================================================

@dataclass(frozen=True)
class ConstraintViolation:
    constraint_id: str
    constraint_name: str

    category: ConstraintCategory
    constraint_type: ConstraintType

    variables: tuple[str, ...]

    weight: float

    description: str


# ============================================================
# SOLVER STATISTICS
# ============================================================

@dataclass
class SolverStatistics:
    search_nodes: int = 0
    backtracks: int = 0
    constraint_checks: int = 0
    domain_reductions: int = 0
    solutions_found: int = 0

    elapsed_seconds: float = 0.0


# ============================================================
# SOLVER RESULT
# ============================================================

@dataclass(frozen=True)
class ConstraintSolverResult:
    request_id: str

    status: SolverStatus

    assignment: dict[str, Any]

    objective_score: float

    hard_constraints_satisfied: bool

    soft_constraints_satisfied: int
    soft_constraints_total: int

    violations: tuple[ConstraintViolation, ...]

    statistics: SolverStatistics

    optimality_proven: bool = False

    warnings: tuple[str, ...] = ()

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# CONSTRAINT SATISFACTION PROBLEM
# ============================================================

class ConstraintSatisfactionProblem:
    """
    Stores variables and trusted Python constraints.

    Predicates must originate from trusted application code.
    Never execute user-provided Python expressions directly.
    """

    def __init__(
        self,
        *,
        problem_id: str | None = None,
        name: str = "Robotics Constraint Problem",
    ) -> None:
        self.problem_id = problem_id or generate_id()

        self.name = name

        self.variables: dict[
            str,
            ConstraintVariable,
        ] = {}

        self.constraints: list[Constraint] = []

        self._constraint_ids: set[str] = set()

        self._lock = RLock()

    def add_variable(
        self,
        variable: ConstraintVariable,
    ) -> None:
        with self._lock:
            if variable.name in self.variables:
                raise ConstraintValidationError(
                    f"Duplicate variable: {variable.name}"
                )

            self.variables[variable.name] = variable

    def add_constraint(
        self,
        constraint: Constraint,
    ) -> None:
        with self._lock:
            if constraint.constraint_id in self._constraint_ids:
                raise ConstraintValidationError(
                    f"Duplicate constraint: "
                    f"{constraint.constraint_id}"
                )

            for variable in constraint.variables:
                if variable not in self.variables:
                    raise ConstraintValidationError(
                        f"Unknown variable: {variable}"
                    )

            self.constraints.append(constraint)

            self._constraint_ids.add(
                constraint.constraint_id
            )

    def hard_constraints(self) -> list[Constraint]:
        return [
            constraint
            for constraint in self.constraints
            if constraint.constraint_type == ConstraintType.HARD
        ]

    def soft_constraints(self) -> list[Constraint]:
        return [
            constraint
            for constraint in self.constraints
            if constraint.constraint_type == ConstraintType.SOFT
        ]

    def validate(
        self,
        config: ConstraintSolverConfig,
    ) -> None:
        if not self.variables:
            raise ConstraintValidationError(
                "Problem contains no variables"
            )

        if len(self.variables) > config.maximum_variables:
            raise ConstraintValidationError(
                "Maximum variable count exceeded"
            )

        if len(self.constraints) > config.maximum_constraints:
            raise ConstraintValidationError(
                "Maximum constraint count exceeded"
            )

        for variable in self.variables.values():
            if len(variable.domain) > config.maximum_domain_size:
                raise ConstraintValidationError(
                    f"Domain too large: {variable.name}"
                )


# ============================================================
# CONSTRAINT FACTORY
# ============================================================

class ConstraintFactory:
    """
    Trusted helpers for constructing common constraints.
    """

    @staticmethod
    def equals(
        variable: str,
        expected: Any,
        *,
        constraint_type: ConstraintType = ConstraintType.HARD,
        weight: float = 1.0,
    ) -> Constraint:
        return Constraint(
            constraint_id=generate_id(),
            name=f"{variable}_equals",
            variables=(variable,),
            predicate=lambda a: a[variable] == expected,
            constraint_type=constraint_type,
            weight=weight,
        )

    @staticmethod
    def not_equals(
        variable: str,
        forbidden: Any,
    ) -> Constraint:
        return Constraint(
            constraint_id=generate_id(),
            name=f"{variable}_not_equals",
            variables=(variable,),
            predicate=lambda a: a[variable] != forbidden,
        )

    @staticmethod
    def all_different(
        variables: Sequence[str],
    ) -> Constraint:
        names = tuple(variables)

        if not names:
            raise ConstraintValidationError(
                "all_different requires variables"
            )

        return Constraint(
            constraint_id=generate_id(),
            name="all_different",
            variables=names,
            predicate=lambda a: (
                len({
                    a[name]
                    for name in names
                }) == len(names)
            ),
            category=ConstraintCategory.RESOURCE,
        )

    @staticmethod
    def less_than(
        left: str,
        right: str,
    ) -> Constraint:
        return Constraint(
            constraint_id=generate_id(),
            name=f"{left}_before_{right}",
            variables=(left, right),
            predicate=lambda a: a[left] < a[right],
            category=ConstraintCategory.DEPENDENCY,
        )

    @staticmethod
    def less_than_or_equal(
        left: str,
        right: str,
    ) -> Constraint:
        return Constraint(
            constraint_id=generate_id(),
            name=f"{left}_lte_{right}",
            variables=(left, right),
            predicate=lambda a: a[left] <= a[right],
        )

    @staticmethod
    def minimum_value(
        variable: str,
        minimum: float,
    ) -> Constraint:
        finite_number(minimum, "minimum")

        return Constraint(
            constraint_id=generate_id(),
            name=f"{variable}_minimum",
            variables=(variable,),
            predicate=lambda a: a[variable] >= minimum,
        )

    @staticmethod
    def maximum_value(
        variable: str,
        maximum: float,
    ) -> Constraint:
        finite_number(maximum, "maximum")

        return Constraint(
            constraint_id=generate_id(),
            name=f"{variable}_maximum",
            variables=(variable,),
            predicate=lambda a: a[variable] <= maximum,
        )

    @staticmethod
    def allowed_values(
        variable: str,
        allowed: Iterable[Any],
    ) -> Constraint:
        allowed_set = frozenset(allowed)

        return Constraint(
            constraint_id=generate_id(),
            name=f"{variable}_allowed_values",
            variables=(variable,),
            predicate=lambda a: a[variable] in allowed_set,
        )

    @staticmethod
    def time_window(
        variable: str,
        earliest: int,
        latest: int,
    ) -> Constraint:
        if earliest > latest:
            raise ConstraintValidationError(
                "Earliest time exceeds latest time"
            )

        return Constraint(
            constraint_id=generate_id(),
            name=f"{variable}_time_window",
            variables=(variable,),
            predicate=lambda a: (
                earliest <= a[variable] <= latest
            ),
            category=ConstraintCategory.TIME_WINDOW,
        )

    @staticmethod
    def dependency(
        predecessor_start: str,
        successor_start: str,
        predecessor_duration: int,
    ) -> Constraint:
        if predecessor_duration < 0:
            raise ConstraintValidationError(
                "Duration cannot be negative"
            )

        return Constraint(
            constraint_id=generate_id(),
            name="task_dependency",
            variables=(
                predecessor_start,
                successor_start,
            ),
            predicate=lambda a: (
                a[successor_start]
                >= a[predecessor_start] + predecessor_duration
            ),
            category=ConstraintCategory.DEPENDENCY,
        )

    @staticmethod
    def no_overlap(
        first_start: str,
        second_start: str,
        first_duration: int,
        second_duration: int,
    ) -> Constraint:
        if first_duration < 0 or second_duration < 0:
            raise ConstraintValidationError(
                "Task durations cannot be negative"
            )

        return Constraint(
            constraint_id=generate_id(),
            name="no_overlap",
            variables=(
                first_start,
                second_start,
            ),
            predicate=lambda a: (
                a[first_start] + first_duration
                <= a[second_start]
                or
                a[second_start] + second_duration
                <= a[first_start]
            ),
            category=ConstraintCategory.RESOURCE,
        )

    @staticmethod
    def preferred_value(
        variable: str,
        preferred: Any,
        *,
        weight: float = 1.0,
    ) -> Constraint:
        return Constraint(
            constraint_id=generate_id(),
            name=f"{variable}_preferred",
            variables=(variable,),
            predicate=lambda a: a[variable] == preferred,
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.PRIORITY,
            weight=weight,
        )


# ============================================================
# CONSTRAINT EVALUATOR
# ============================================================

class ConstraintEvaluator:

    @staticmethod
    def is_consistent(
        assignment: Mapping[str, Any],
        constraints: Sequence[Constraint],
        statistics: SolverStatistics | None = None,
    ) -> bool:
        for constraint in constraints:
            if constraint.constraint_type != ConstraintType.HARD:
                continue

            if statistics is not None:
                statistics.constraint_checks += 1

            result = constraint.evaluate(
                assignment
            )

            if result is False:
                return False

        return True

    @staticmethod
    def soft_score(
        assignment: Mapping[str, Any],
        constraints: Sequence[Constraint],
    ) -> float:
        score = 0.0

        for constraint in constraints:
            if constraint.constraint_type != ConstraintType.SOFT:
                continue

            if constraint.evaluate(assignment) is True:
                score += constraint.weight

        return score

    @staticmethod
    def violations(
        assignment: Mapping[str, Any],
        constraints: Sequence[Constraint],
    ) -> tuple[ConstraintViolation, ...]:
        violations = []

        for constraint in constraints:
            if constraint.evaluate(assignment) is False:
                violations.append(
                    ConstraintViolation(
                        constraint_id=constraint.constraint_id,
                        constraint_name=constraint.name,
                        category=constraint.category,
                        constraint_type=constraint.constraint_type,
                        variables=constraint.variables,
                        weight=constraint.weight,
                        description=constraint.description,
                    )
                )

        return tuple(violations)


# ============================================================
# BACKTRACKING CONSTRAINT SOLVER
# ============================================================

class BacktrackingConstraintSolver:
    """
    Exact finite-domain CSP search.

    Supports:
        - MRV
        - Degree heuristic
        - LCV
        - Forward checking
        - Binary AC-3
        - Soft constraint maximization

    Note:
        General n-ary constraints are checked when all their
        variables are assigned. Forward checking supports
        arbitrary constraint arity by testing assignments
        only when the constraint becomes fully instantiated.

        This solver is appropriate for small to medium
        planning problems. Large industrial schedules may
        require OR-Tools CP-SAT or another specialized solver.
    """

    def __init__(
        self,
        config: ConstraintSolverConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ConstraintSolverConfig()
        )

        self._cancelled = False
        self._lock = RLock()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    def solve(
        self,
        problem: ConstraintSatisfactionProblem,
        *,
        request_id: str | None = None,
    ) -> ConstraintSolverResult:
        problem.validate(self.config)

        request_id = request_id or generate_id()

        with self._lock:
            self._cancelled = False

        started = time.monotonic()

        stats = SolverStatistics()

        domains: dict[str, list[Any]] = {
            name: list(variable.domain)
            for name, variable in problem.variables.items()
        }

        hard_constraints = problem.hard_constraints()
        soft_constraints = problem.soft_constraints()

        constraints_by_variable: dict[
            str, list[Constraint]
        ] = defaultdict(list)

        neighbors: dict[
            str, set[str]
        ] = defaultdict(set)

        for constraint in hard_constraints:
            for variable in constraint.variables:
                constraints_by_variable[variable].append(
                    constraint
                )

                neighbors[variable].update(
                    other
                    for other in constraint.variables
                    if other != variable
                )

        best_assignment: dict[str, Any] | None = None
        best_score = float("-inf")

        complete_search = True
        termination_status: SolverStatus | None = None

        max_possible_score = sum(
            constraint.weight
            for constraint in soft_constraints
        )

        def check_limits() -> bool:
            nonlocal complete_search
            nonlocal termination_status

            with self._lock:
                cancelled = self._cancelled

            if cancelled:
                complete_search = False
                termination_status = SolverStatus.CANCELLED
                return False

            if (
                time.monotonic() - started
                >= self.config.timeout_seconds
            ):
                complete_search = False
                termination_status = SolverStatus.TIMEOUT
                return False

            if (
                stats.search_nodes
                >= self.config.maximum_search_nodes
            ):
                complete_search = False
                termination_status = SolverStatus.NODE_LIMIT
                return False

            return True

        def select_variable(
            assignment: Mapping[str, Any],
            current_domains: Mapping[str, list[Any]],
        ) -> str:
            unassigned = [
                name
                for name in problem.variables
                if name not in assignment
            ]

            if not self.config.use_mrv:
                return unassigned[0]

            def ranking(name: str) -> tuple[int, int, str]:
                degree = (
                    len(neighbors[name] - assignment.keys())
                    if self.config.use_degree_heuristic
                    else 0
                )

                return (
                    len(current_domains[name]),
                    -degree,
                    name,
                )

            return min(
                unassigned,
                key=ranking,
            )

        def order_values(
            variable: str,
            assignment: Mapping[str, Any],
            current_domains: Mapping[str, list[Any]],
        ) -> list[Any]:
            values = list(
                current_domains[variable]
            )

            if not self.config.use_lcv:
                return values

            def elimination_count(value: Any) -> int:
                count = 0

                trial = dict(assignment)
                trial[variable] = value

                for neighbor in neighbors[variable]:
                    if neighbor in assignment:
                        continue

                    for candidate in current_domains[neighbor]:
                        extended = dict(trial)
                        extended[neighbor] = candidate

                        relevant = [
                            constraint
                            for constraint in constraints_by_variable[neighbor]
                            if variable in constraint.variables
                        ]

                        if not ConstraintEvaluator.is_consistent(
                            extended,
                            relevant,
                        ):
                            count += 1

                return count

            return sorted(
                values,
                key=elimination_count,
            )

        def forward_check(
            assignment: Mapping[str, Any],
            current_domains: dict[str, list[Any]],
            assigned_variable: str,
        ) -> bool:
            for neighbor in neighbors[assigned_variable]:
                if neighbor in assignment:
                    continue

                filtered = []

                for candidate in current_domains[neighbor]:
                    trial = dict(assignment)
                    trial[neighbor] = candidate

                    if ConstraintEvaluator.is_consistent(
                        trial,
                        constraints_by_variable[neighbor],
                        stats,
                    ):
                        filtered.append(candidate)

                stats.domain_reductions += (
                    len(current_domains[neighbor])
                    - len(filtered)
                )

                if not filtered:
                    return False

                current_domains[neighbor] = filtered

            return True

        def backtrack(
            assignment: dict[str, Any],
            current_domains: dict[str, list[Any]],
        ) -> None:
            nonlocal best_assignment
            nonlocal best_score

            if not check_limits():
                return

            stats.search_nodes += 1

            if len(assignment) == len(problem.variables):
                stats.solutions_found += 1

                score = ConstraintEvaluator.soft_score(
                    assignment,
                    soft_constraints,
                )

                if (
                    best_assignment is None
                    or score > best_score
                ):
                    best_assignment = dict(assignment)
                    best_score = score

                return

            variable = select_variable(
                assignment,
                current_domains,
            )

            for value in order_values(
                variable,
                assignment,
                current_domains,
            ):
                if not check_limits():
                    return

                assignment[variable] = value

                relevant_constraints = (
                    constraints_by_variable[variable]
                )

                consistent = ConstraintEvaluator.is_consistent(
                    assignment,
                    relevant_constraints,
                    stats,
                )

                if consistent:
                    next_domains = {
                        key: list(values)
                        for key, values in current_domains.items()
                    }

                    next_domains[variable] = [value]

                    if (
                        not self.config.enable_forward_checking
                        or forward_check(
                            assignment,
                            next_domains,
                            variable,
                        )
                    ):
                        backtrack(
                            assignment,
                            next_domains,
                        )

                del assignment[variable]

                if (
                    not self.config.optimize_soft_constraints
                    and best_assignment is not None
                ):
                    return

                if (
                    best_assignment is not None
                    and best_score >= max_possible_score
                ):
                    return

            stats.backtracks += 1

        # AC-3 applies to binary hard constraints.
        if self.config.enable_arc_consistency:
            ac3_ok = self._ac3(
                domains,
                hard_constraints,
                stats,
                deadline=(
                    started + self.config.timeout_seconds
                ),
            )

            if not ac3_ok:
                stats.elapsed_seconds = (
                    time.monotonic() - started
                )

                return ConstraintSolverResult(
                    request_id=request_id,
                    status=SolverStatus.UNSATISFIABLE,
                    assignment={},
                    objective_score=0.0,
                    hard_constraints_satisfied=False,
                    soft_constraints_satisfied=0,
                    soft_constraints_total=len(soft_constraints),
                    violations=(),
                    statistics=stats,
                    optimality_proven=True,
                )

        backtrack(
            {},
            domains,
        )

        stats.elapsed_seconds = (
            time.monotonic() - started
        )

        if best_assignment is None:
            status = (
                termination_status
                if termination_status is not None
                else SolverStatus.UNSATISFIABLE
            )

            return ConstraintSolverResult(
                request_id=request_id,
                status=status,
                assignment={},
                objective_score=0.0,
                hard_constraints_satisfied=False,
                soft_constraints_satisfied=0,
                soft_constraints_total=len(soft_constraints),
                violations=(),
                statistics=stats,
                optimality_proven=(
                    status == SolverStatus.UNSATISFIABLE
                ),
                warnings=(
                    ("Search terminated before finding a solution",)
                    if termination_status is not None
                    else ()
                ),
            )

        violations = ConstraintEvaluator.violations(
            best_assignment,
            problem.constraints,
        )

        soft_satisfied = sum(
            constraint.evaluate(best_assignment) is True
            for constraint in soft_constraints
        )

        optimal = (
            complete_search
            or best_score >= max_possible_score
        )

        status = (
            SolverStatus.OPTIMAL
            if optimal
            else SolverStatus.FEASIBLE
        )

        return ConstraintSolverResult(
            request_id=request_id,
            status=status,
            assignment=best_assignment,
            objective_score=best_score,
            hard_constraints_satisfied=True,
            soft_constraints_satisfied=soft_satisfied,
            soft_constraints_total=len(soft_constraints),
            violations=violations[
                :self.config.maximum_violation_reports
            ],
            statistics=stats,
            optimality_proven=optimal,
            warnings=(
                ()
                if optimal
                else (
                    "Feasible assignment found; "
                    "optimality not proven",
                )
            ),
        )

    # --------------------------------------------------------
    # AC-3 ARC CONSISTENCY
    # --------------------------------------------------------

    @staticmethod
    def _ac3(
        domains: dict[str, list[Any]],
        constraints: Sequence[Constraint],
        stats: SolverStatistics,
        *,
        deadline: float,
    ) -> bool:
        binary_constraints = [
            constraint
            for constraint in constraints
            if len(constraint.variables) == 2
        ]

        arcs: dict[
            tuple[str, str],
            list[Constraint],
        ] = defaultdict(list)

        for constraint in binary_constraints:
            first, second = constraint.variables

            arcs[(first, second)].append(constraint)
            arcs[(second, first)].append(constraint)

        queue = deque(
            arcs.keys()
        )

        def revise(
            first: str,
            second: str,
        ) -> bool:
            revised = False
            filtered = []

            for first_value in domains[first]:
                supported = False

                for second_value in domains[second]:
                    assignment = {
                        first: first_value,
                        second: second_value,
                    }

                    if ConstraintEvaluator.is_consistent(
                        assignment,
                        arcs[(first, second)],
                        stats,
                    ):
                        supported = True
                        break

                if supported:
                    filtered.append(first_value)
                else:
                    revised = True
                    stats.domain_reductions += 1

            if revised:
                domains[first] = filtered

            return revised

        while queue:
            if time.monotonic() >= deadline:
                # Leave remaining propagation to backtracking.
                return True

            first, second = queue.popleft()

            if revise(first, second):
                if not domains[first]:
                    return False

                for neighbor, target in arcs:
                    if target == first and neighbor != second:
                        queue.append(
                            (neighbor, first)
                        )

        return True


# ============================================================
# ROBOT AND TASK MODELS
# ============================================================

@dataclass(frozen=True)
class PlanningRobot:
    robot_id: str
    tenant_id: str

    capabilities: frozenset[str]

    battery_percentage: float

    availability: RobotAvailability = (
        RobotAvailability.AVAILABLE
    )

    safety_state: RobotSafetyState = (
        RobotSafetyState.UNKNOWN
    )

    emergency_stop_active: bool = False

    active_faults: tuple[str, ...] = ()

    allowed_zones: frozenset[str] = frozenset()

    energy_capacity_wh: float = 1000.0

    maximum_payload_kg: float = 0.0

    hourly_operating_cost: float = 0.0

    last_updated: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id, "robot_id")
        validate_identifier(self.tenant_id, "tenant_id")

        if not 0 <= self.battery_percentage <= 100:
            raise ConstraintValidationError(
                "Battery percentage must be 0–100"
            )

        finite_number(
            self.energy_capacity_wh,
            "energy_capacity_wh",
            minimum=0,
        )

        finite_number(
            self.maximum_payload_kg,
            "maximum_payload_kg",
            minimum=0,
        )

        finite_number(
            self.hourly_operating_cost,
            "hourly_operating_cost",
            minimum=0,
        )

        if self.last_updated.tzinfo is None:
            raise ConstraintValidationError(
                "Robot timestamp must be timezone-aware"
            )


@dataclass(frozen=True)
class PlanningTask:
    task_id: str
    tenant_id: str

    name: str

    duration_minutes: int

    required_capabilities: frozenset[str] = frozenset()

    required_zone: str | None = None

    required_payload_kg: float = 0.0

    estimated_energy_wh: float = 0.0

    earliest_start_minute: int = 0

    latest_finish_minute: int = 480

    dependencies: tuple[str, ...] = ()

    priority: int = 1

    requires_human_approval: bool = False

    approval_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.task_id, "task_id")
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.name, "task name")

        if self.duration_minutes <= 0:
            raise ConstraintValidationError(
                "Task duration must be positive"
            )

        if self.earliest_start_minute < 0:
            raise ConstraintValidationError(
                "Earliest start cannot be negative"
            )

        if (
            self.latest_finish_minute
            <= self.earliest_start_minute
        ):
            raise ConstraintValidationError(
                "Invalid task time window"
            )

        finite_number(
            self.required_payload_kg,
            "required_payload_kg",
            minimum=0,
        )

        finite_number(
            self.estimated_energy_wh,
            "estimated_energy_wh",
            minimum=0,
        )

        if self.priority < 1:
            raise ConstraintValidationError(
                "Task priority must be positive"
            )

        if (
            self.requires_human_approval
            and not self.approval_id
        ):
            raise ConstraintValidationError(
                "Approval ID required for approval-gated task"
            )


# ============================================================
# ROBOT TASK ASSIGNMENT
# ============================================================

@dataclass(frozen=True)
class RobotTaskAssignment:
    task_id: str
    robot_id: str

    start_minute: int
    finish_minute: int

    estimated_energy_wh: float

    estimated_cost: float

    approved_for_simulation: bool

    hardware_execution_authorized: bool = False


@dataclass(frozen=True)
class RobotPlanningResult:
    plan_id: str
    tenant_id: str

    status: SolverStatus

    assignments: tuple[RobotTaskAssignment, ...]

    unassigned_tasks: tuple[str, ...]

    total_estimated_cost: float

    total_estimated_energy_wh: float

    solver_result: ConstraintSolverResult

    warnings: tuple[str, ...]

    simulation_only: bool = True

    created_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# ROBOT ELIGIBILITY
# ============================================================

class RobotEligibilityEvaluator:

    @staticmethod
    def evaluate(
        robot: PlanningRobot,
        task: PlanningTask,
        *,
        approvals: frozenset[str],
        minimum_battery_percentage: float,
        maximum_robot_state_age_seconds: float,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []

        if robot.tenant_id != task.tenant_id:
            reasons.append(
                "Tenant mismatch"
            )

        if robot.availability != RobotAvailability.AVAILABLE:
            reasons.append(
                "Robot unavailable"
            )

        if robot.safety_state != RobotSafetyState.SAFE:
            reasons.append(
                "Robot safety state not SAFE"
            )

        if robot.emergency_stop_active:
            reasons.append(
                "Emergency stop active"
            )

        if robot.active_faults:
            reasons.append(
                "Active robot faults"
            )

        age = (
            utc_now() - robot.last_updated
        ).total_seconds()

        if (
            age < 0
            or age > maximum_robot_state_age_seconds
        ):
            reasons.append(
                "Robot state stale"
            )

        if robot.battery_percentage < minimum_battery_percentage:
            reasons.append(
                "Battery below minimum"
            )

        if not task.required_capabilities.issubset(
            robot.capabilities
        ):
            reasons.append(
                "Missing robot capabilities"
            )

        if (
            task.required_zone is not None
            and task.required_zone not in robot.allowed_zones
        ):
            reasons.append(
                "Robot not authorized for task zone"
            )

        if (
            task.required_payload_kg
            > robot.maximum_payload_kg
        ):
            reasons.append(
                "Task payload exceeds robot capacity"
            )

        available_energy_wh = (
            robot.energy_capacity_wh
            * robot.battery_percentage
            / 100.0
        )

        if task.estimated_energy_wh > available_energy_wh:
            reasons.append(
                "Insufficient estimated battery energy"
            )

        if task.requires_human_approval:
            if task.approval_id not in approvals:
                reasons.append(
                    "Human approval missing"
                )

        return (
            not reasons,
            tuple(reasons),
        )


# ============================================================
# MULTI-ROBOT CONSTRAINT PLANNER
# ============================================================

class RoboticsConstraintPlanner:
    """
    Builds a finite-domain CSP for multi-robot task planning.

    Planning variables:
        robot_<task_id>
        start_<task_id>

    Constraints:
        - Robot eligibility
        - Task time windows
        - Task dependencies
        - Same-robot non-overlap
        - Aggregate energy feasibility
        - Task priority preferences

    Scheduling uses integer minutes relative to the
    planning horizon.

    The resulting plan is advisory and simulation-only.
    """

    def __init__(
        self,
        *,
        solver_config: ConstraintSolverConfig | None = None,
        minimum_battery_percentage: float = 10.0,
        robot_state_max_age_seconds: float = 60.0,
        time_step_minutes: int = 5,
    ) -> None:
        self.solver_config = (
            solver_config
            or ConstraintSolverConfig()
        )

        self.minimum_battery_percentage = finite_number(
            minimum_battery_percentage,
            "minimum_battery_percentage",
            minimum=0,
        )

        if self.minimum_battery_percentage > 100:
            raise ConstraintValidationError(
                "Minimum battery cannot exceed 100%"
            )

        self.robot_state_max_age_seconds = finite_number(
            robot_state_max_age_seconds,
            "robot_state_max_age_seconds",
            minimum=0.001,
        )

        if time_step_minutes < 1:
            raise ConstraintValidationError(
                "Time step must be positive"
            )

        self.time_step_minutes = time_step_minutes

        self.solver = BacktrackingConstraintSolver(
            self.solver_config
        )

    def plan(
        self,
        *,
        tenant_id: str,
        robots: Sequence[PlanningRobot],
        tasks: Sequence[PlanningTask],
        access: PlanningAccessContext,
        approvals: frozenset[str] = frozenset(),
    ) -> RobotPlanningResult:
        validate_identifier(tenant_id, "tenant_id")

        if not access.can_plan:
            raise ConstraintAuthorizationError(
                "Planning permission required"
            )

        if access.tenant_id != tenant_id:
            raise ConstraintAuthorizationError(
                "Cross-tenant planning denied"
            )

        if not robots:
            raise ConstraintValidationError(
                "At least one robot is required"
            )

        if not tasks:
            raise ConstraintValidationError(
                "At least one task is required"
            )

        robot_map = {
            robot.robot_id: robot
            for robot in robots
        }

        task_map = {
            task.task_id: task
            for task in tasks
        }

        if len(robot_map) != len(robots):
            raise ConstraintValidationError(
                "Duplicate robot IDs"
            )

        if len(task_map) != len(tasks):
            raise ConstraintValidationError(
                "Duplicate task IDs"
            )

        for robot in robots:
            if robot.tenant_id != tenant_id:
                raise ConstraintAuthorizationError(
                    "Robot tenant mismatch"
                )

            if robot.robot_id not in access.allowed_robot_ids:
                raise ConstraintAuthorizationError(
                    f"Unauthorized robot: {robot.robot_id}"
                )

        for task in tasks:
            if task.tenant_id != tenant_id:
                raise ConstraintAuthorizationError(
                    "Task tenant mismatch"
                )

            for dependency in task.dependencies:
                if dependency not in task_map:
                    raise ConstraintValidationError(
                        f"Unknown dependency: {dependency}"
                    )

        self._validate_dependency_graph(
            task_map
        )

        problem = ConstraintSatisfactionProblem(
            name="Multi-Robot Task Assignment",
        )

        warnings: list[str] = []

        # ----------------------------------------------------
        # CREATE TASK VARIABLES
        # ----------------------------------------------------

        for task in tasks:
            eligible_robots = []

            for robot in robots:
                eligible, reasons = (
                    RobotEligibilityEvaluator.evaluate(
                        robot,
                        task,
                        approvals=approvals,
                        minimum_battery_percentage=(
                            self.minimum_battery_percentage
                        ),
                        maximum_robot_state_age_seconds=(
                            self.robot_state_max_age_seconds
                        ),
                    )
                )

                if eligible:
                    eligible_robots.append(
                        robot.robot_id
                    )

            if not eligible_robots:
                warnings.append(
                    f"No eligible robot for task {task.task_id}"
                )

                return self._unsatisfied_plan(
                    tenant_id=tenant_id,
                    tasks=tasks,
                    warnings=warnings,
                )

            latest_start = (
                task.latest_finish_minute
                - task.duration_minutes
            )

            start_domain = tuple(
                range(
                    task.earliest_start_minute,
                    latest_start + 1,
                    self.time_step_minutes,
                )
            )

            if not start_domain:
                warnings.append(
                    f"No feasible start time for {task.task_id}"
                )

                return self._unsatisfied_plan(
                    tenant_id=tenant_id,
                    tasks=tasks,
                    warnings=warnings,
                )

            problem.add_variable(
                ConstraintVariable(
                    name=f"robot_{task.task_id}",
                    domain=tuple(eligible_robots),
                )
            )

            problem.add_variable(
                ConstraintVariable(
                    name=f"start_{task.task_id}",
                    domain=start_domain,
                    variable_type=VariableType.INTEGER,
                )
            )

        # ----------------------------------------------------
        # TASK DEPENDENCIES
        # ----------------------------------------------------

        for task in tasks:
            for dependency_id in task.dependencies:
                dependency = task_map[
                    dependency_id
                ]

                problem.add_constraint(
                    ConstraintFactory.dependency(
                        predecessor_start=(
                            f"start_{dependency_id}"
                        ),
                        successor_start=(
                            f"start_{task.task_id}"
                        ),
                        predecessor_duration=(
                            dependency.duration_minutes
                        ),
                    )
                )

        # ----------------------------------------------------
        # ROBOT RESOURCE CONFLICTS
        # ----------------------------------------------------

        for first, second in combinations(tasks, 2):
            first_robot = f"robot_{first.task_id}"
            second_robot = f"robot_{second.task_id}"

            first_start = f"start_{first.task_id}"
            second_start = f"start_{second.task_id}"

            problem.add_constraint(
                Constraint(
                    constraint_id=generate_id(),
                    name=(
                        f"resource_conflict_"
                        f"{first.task_id}_{second.task_id}"
                    ),
                    variables=(
                        first_robot,
                        second_robot,
                        first_start,
                        second_start,
                    ),
                    category=ConstraintCategory.RESOURCE,
                    predicate=(
                        lambda a,
                        fr=first_robot,
                        sr=second_robot,
                        fs=first_start,
                        ss=second_start,
                        fd=first.duration_minutes,
                        sd=second.duration_minutes: (
                            a[fr] != a[sr]
                            or a[fs] + fd <= a[ss]
                            or a[ss] + sd <= a[fs]
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # AGGREGATE BATTERY ENERGY
        # ----------------------------------------------------

        robot_variables = tuple(
            f"robot_{task.task_id}"
            for task in tasks
        )

        available_energy = {
            robot.robot_id: (
                robot.energy_capacity_wh
                * max(
                    0.0,
                    (
                        robot.battery_percentage
                        - self.minimum_battery_percentage
                    ) / 100.0,
                )
            )
            for robot in robots
        }

        energy_by_task = {
            task.task_id: task.estimated_energy_wh
            for task in tasks
        }

        task_ids = tuple(
            task.task_id
            for task in tasks
        )

        def energy_constraint(
            assignment: Mapping[str, Any],
        ) -> bool:
            consumption = defaultdict(float)

            for task_id in task_ids:
                robot_id = assignment[
                    f"robot_{task_id}"
                ]

                consumption[robot_id] += (
                    energy_by_task[task_id]
                )

            return all(
                used <= available_energy[robot_id] + 1e-9
                for robot_id, used in consumption.items()
            )

        problem.add_constraint(
            Constraint(
                constraint_id=generate_id(),
                name="aggregate_energy_budget",
                variables=robot_variables,
                predicate=energy_constraint,
                category=ConstraintCategory.BATTERY,
            )
        )

        # ----------------------------------------------------
        # COST PREFERENCES
        # ----------------------------------------------------

        for task in tasks:
            robot_variable = f"robot_{task.task_id}"

            cheapest_robot_id = min(
                problem.variables[robot_variable].domain,
                key=lambda robot_id: (
                    robot_map[robot_id].hourly_operating_cost
                ),
            )

            problem.add_constraint(
                ConstraintFactory.preferred_value(
                    robot_variable,
                    cheapest_robot_id,
                    weight=float(task.priority),
                )
            )

        # ----------------------------------------------------
        # SOLVE
        # ----------------------------------------------------

        result = self.solver.solve(
            problem
        )

        if not result.assignment:
            return RobotPlanningResult(
                plan_id=generate_id(),
                tenant_id=tenant_id,
                status=result.status,
                assignments=(),
                unassigned_tasks=tuple(
                    task.task_id
                    for task in tasks
                ),
                total_estimated_cost=0.0,
                total_estimated_energy_wh=0.0,
                solver_result=result,
                warnings=tuple(warnings)
                + result.warnings,
            )

        assignments = []

        for task in tasks:
            robot_id = result.assignment[
                f"robot_{task.task_id}"
            ]

            start = result.assignment[
                f"start_{task.task_id}"
            ]

            robot = robot_map[
                robot_id
            ]

            cost = (
                task.duration_minutes
                / 60.0
                * robot.hourly_operating_cost
            )

            assignments.append(
                RobotTaskAssignment(
                    task_id=task.task_id,
                    robot_id=robot_id,
                    start_minute=start,
                    finish_minute=(
                        start + task.duration_minutes
                    ),
                    estimated_energy_wh=(
                        task.estimated_energy_wh
                    ),
                    estimated_cost=cost,
                    approved_for_simulation=(
                        not task.requires_human_approval
                        or task.approval_id in approvals
                    ),
                    hardware_execution_authorized=False,
                )
            )

        assignments.sort(
            key=lambda assignment: (
                assignment.start_minute,
                assignment.task_id,
            )
        )

        return RobotPlanningResult(
            plan_id=generate_id(),
            tenant_id=tenant_id,
            status=result.status,
            assignments=tuple(assignments),
            unassigned_tasks=(),
            total_estimated_cost=sum(
                item.estimated_cost
                for item in assignments
            ),
            total_estimated_energy_wh=sum(
                item.estimated_energy_wh
                for item in assignments
            ),
            solver_result=result,
            warnings=tuple(warnings)
            + result.warnings,
        )

    @staticmethod
    def _validate_dependency_graph(
        tasks: Mapping[str, PlanningTask],
    ) -> None:
        visiting = set()
        visited = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ConstraintValidationError(
                    "Circular task dependency detected"
                )

            if task_id in visited:
                return

            visiting.add(task_id)

            for dependency in tasks[task_id].dependencies:
                visit(dependency)

            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in tasks:
            visit(task_id)

    @staticmethod
    def _unsatisfied_plan(
        *,
        tenant_id: str,
        tasks: Sequence[PlanningTask],
        warnings: Sequence[str],
    ) -> RobotPlanningResult:
        solver_result = ConstraintSolverResult(
            request_id=generate_id(),
            status=SolverStatus.UNSATISFIABLE,
            assignment={},
            objective_score=0.0,
            hard_constraints_satisfied=False,
            soft_constraints_satisfied=0,
            soft_constraints_total=0,
            violations=(),
            statistics=SolverStatistics(),
            optimality_proven=True,
            warnings=tuple(warnings),
        )

        return RobotPlanningResult(
            plan_id=generate_id(),
            tenant_id=tenant_id,
            status=SolverStatus.UNSATISFIABLE,
            assignments=(),
            unassigned_tasks=tuple(
                task.task_id
                for task in tasks
            ),
            total_estimated_cost=0.0,
            total_estimated_energy_wh=0.0,
            solver_result=solver_result,
            warnings=tuple(warnings),
        )


# ============================================================
# PLANNING ENGINE
# ============================================================

class RoboticsConstraintSolverEngine:
    """
    Tenant-aware facade for general CSP solving and
    multi-robot planning.

    The caller must provide an authenticated access context
    constructed by trusted server-side authorization logic.
    """

    def __init__(
        self,
        *,
        config: ConstraintSolverConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ConstraintSolverConfig()
        )

        self._solver = BacktrackingConstraintSolver(
            self.config
        )

        self._planner = RoboticsConstraintPlanner(
            solver_config=self.config
        )

        self._results: dict[
            str,
            tuple[str, ConstraintSolverResult],
        ] = {}

        self._planning_results: dict[
            str,
            RobotPlanningResult,
        ] = {}

        self._audit: deque[dict[str, Any]] = deque(
            maxlen=10_000
        )

        self._lock = RLock()

        self._metrics = Counter()

    def solve(
        self,
        problem: ConstraintSatisfactionProblem,
        *,
        access: PlanningAccessContext,
    ) -> ConstraintSolverResult:
        if not access.can_plan:
            raise ConstraintAuthorizationError(
                "Planning permission required"
            )

        result = self._solver.solve(
            problem
        )

        with self._lock:
            self._results[result.request_id] = (
                access.tenant_id,
                result,
            )

            self._metrics["solve_requests"] += 1

            self._metrics[
                f"status_{result.status.value}"
            ] += 1

            self._audit.append({
                "event": "constraint_solved",
                "tenant_id": access.tenant_id,
                "user_id": access.user_id,
                "problem_id": problem.problem_id,
                "request_id": result.request_id,
                "status": result.status.value,
                "timestamp": utc_iso(),
            })

        return result

    def plan_robots(
        self,
        *,
        tenant_id: str,
        robots: Sequence[PlanningRobot],
        tasks: Sequence[PlanningTask],
        access: PlanningAccessContext,
        approvals: frozenset[str] = frozenset(),
    ) -> RobotPlanningResult:
        result = self._planner.plan(
            tenant_id=tenant_id,
            robots=robots,
            tasks=tasks,
            access=access,
            approvals=approvals,
        )

        with self._lock:
            self._planning_results[
                result.plan_id
            ] = result

            self._metrics["robot_plans"] += 1

            self._audit.append({
                "event": "robot_plan_created",
                "tenant_id": tenant_id,
                "user_id": access.user_id,
                "plan_id": result.plan_id,
                "status": result.status.value,
                "task_count": len(tasks),
                "robot_count": len(robots),
                "timestamp": utc_iso(),
            })

        return result

    def get_result(
        self,
        request_id: str,
        *,
        access: PlanningAccessContext,
    ) -> ConstraintSolverResult:
        if not access.can_view_results:
            raise ConstraintAuthorizationError(
                "Result access denied"
            )

        with self._lock:
            entry = self._results.get(
                request_id
            )

            if entry is None:
                raise ConstraintValidationError(
                    "Solver result not found"
                )

            tenant_id, result = entry

            if tenant_id != access.tenant_id:
                raise ConstraintAuthorizationError(
                    "Cross-tenant result access denied"
                )

            return result

    def get_robot_plan(
        self,
        plan_id: str,
        *,
        access: PlanningAccessContext,
    ) -> RobotPlanningResult:
        if not access.can_view_results:
            raise ConstraintAuthorizationError(
                "Result access denied"
            )

        with self._lock:
            result = self._planning_results.get(
                plan_id
            )

            if result is None:
                raise ConstraintValidationError(
                    "Robot plan not found"
                )

            if result.tenant_id != access.tenant_id:
                raise ConstraintAuthorizationError(
                    "Cross-tenant plan access denied"
                )

            for assignment in result.assignments:
                if (
                    assignment.robot_id
                    not in access.allowed_robot_ids
                ):
                    raise ConstraintAuthorizationError(
                        "Robot plan access denied"
                    )

            return result

    def statistics(
        self,
        *,
        access: PlanningAccessContext,
    ) -> dict[str, Any]:
        if not access.can_view_results:
            raise ConstraintAuthorizationError(
                "Statistics access denied"
            )

        with self._lock:
            results = [
                result
                for tenant_id, result in self._results.values()
                if tenant_id == access.tenant_id
            ]

            plans = [
                result
                for result in self._planning_results.values()
                if (
                    result.tenant_id == access.tenant_id
                    and all(
                        assignment.robot_id
                        in access.allowed_robot_ids
                        for assignment in result.assignments
                    )
                )
            ]

        return {
            "tenant_id": access.tenant_id,
            "solver_results": len(results),
            "robot_plans": len(plans),
            "solver_statuses": dict(
                Counter(
                    result.status.value
                    for result in results
                )
            ),
            "plan_statuses": dict(
                Counter(
                    plan.status.value
                    for plan in plans
                )
            ),
            "total_planned_tasks": sum(
                len(plan.assignments)
                for plan in plans
            ),
            "hardware_execution": False,
            "timestamp": utc_iso(),
        }

    def audit_history(
        self,
        *,
        access: PlanningAccessContext,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not access.can_view_results:
            raise ConstraintAuthorizationError(
                "Audit access denied"
            )

        if limit < 1:
            raise ConstraintValidationError(
                "Limit must be positive"
            )

        with self._lock:
            records = [
                dict(record)
                for record in self._audit
                if record["tenant_id"] == access.tenant_id
            ]

        return records[-limit:]

    def export_plan_json(
        self,
        plan_id: str,
        *,
        access: PlanningAccessContext,
    ) -> str:
        result = self.get_robot_plan(
            plan_id,
            access=access,
        )

        payload = {
            "platform": (
                "Physical AI & Autonomous Robotics "
                "Intelligence Platform"
            ),
            "module": "constraint_solver",
            "plan": json_safe(result),
            "simulation_only": True,
            "hardware_execution": False,
            "exported_at": utc_iso(),
        }

        return json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    def cancel(self) -> None:
        self._solver.cancel()
        self._planner.solver.cancel()


# ============================================================
# UNIT TESTS
# ============================================================

class TestConstraintSolver(unittest.TestCase):

    def setUp(self) -> None:
        self.config = ConstraintSolverConfig(
            timeout_seconds=5.0,
            maximum_search_nodes=50_000,
        )

        self.solver = BacktrackingConstraintSolver(
            self.config
        )

    def test_simple_constraint(self) -> None:
        problem = ConstraintSatisfactionProblem()

        problem.add_variable(
            ConstraintVariable(
                name="x",
                domain=(1, 2, 3),
                variable_type=VariableType.INTEGER,
            )
        )

        problem.add_constraint(
            ConstraintFactory.equals(
                "x",
                2,
            )
        )

        result = self.solver.solve(
            problem
        )

        self.assertEqual(
            result.assignment["x"],
            2,
        )

        self.assertTrue(
            result.hard_constraints_satisfied
        )

    def test_all_different(self) -> None:
        problem = ConstraintSatisfactionProblem()

        for name in ("a", "b", "c"):
            problem.add_variable(
                ConstraintVariable(
                    name=name,
                    domain=(1, 2, 3),
                    variable_type=VariableType.INTEGER,
                )
            )

        problem.add_constraint(
            ConstraintFactory.all_different(
                ("a", "b", "c")
            )
        )

        result = self.solver.solve(
            problem
        )

        self.assertEqual(
            len(set(result.assignment.values())),
            3,
        )

    def test_unsatisfiable(self) -> None:
        problem = ConstraintSatisfactionProblem()

        problem.add_variable(
            ConstraintVariable(
                name="x",
                domain=(1, 2),
                variable_type=VariableType.INTEGER,
            )
        )

        problem.add_constraint(
            ConstraintFactory.equals(
                "x",
                1,
            )
        )

        problem.add_constraint(
            ConstraintFactory.equals(
                "x",
                2,
            )
        )

        result = self.solver.solve(
            problem
        )

        self.assertEqual(
            result.status,
            SolverStatus.UNSATISFIABLE,
        )

    def test_soft_constraint(self) -> None:
        problem = ConstraintSatisfactionProblem()

        problem.add_variable(
            ConstraintVariable(
                name="robot",
                domain=("R1", "R2"),
            )
        )

        problem.add_constraint(
            ConstraintFactory.preferred_value(
                "robot",
                "R2",
                weight=10,
            )
        )

        result = self.solver.solve(
            problem
        )

        self.assertEqual(
            result.assignment["robot"],
            "R2",
        )

    def test_time_dependency(self) -> None:
        problem = ConstraintSatisfactionProblem()

        problem.add_variable(
            ConstraintVariable(
                name="start_a",
                domain=(0, 10, 20),
                variable_type=VariableType.INTEGER,
            )
        )

        problem.add_variable(
            ConstraintVariable(
                name="start_b",
                domain=(10, 20, 30),
                variable_type=VariableType.INTEGER,
            )
        )

        problem.add_constraint(
            ConstraintFactory.dependency(
                "start_a",
                "start_b",
                predecessor_duration=10,
            )
        )

        result = self.solver.solve(
            problem
        )

        self.assertGreaterEqual(
            result.assignment["start_b"],
            result.assignment["start_a"] + 10,
        )

    def test_robot_planner(self) -> None:
        access = PlanningAccessContext(
            tenant_id="tenant_alpha",
            user_id="planner",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
                "ROBOT-002",
            }),
            can_plan=True,
            can_view_results=True,
        )

        robots = [
            PlanningRobot(
                robot_id="ROBOT-001",
                tenant_id="tenant_alpha",
                capabilities=frozenset({
                    "inspection",
                }),
                battery_percentage=90,
                availability=RobotAvailability.AVAILABLE,
                safety_state=RobotSafetyState.SAFE,
                allowed_zones=frozenset({
                    "ZONE-A",
                }),
                energy_capacity_wh=1000,
                hourly_operating_cost=15,
            ),
            PlanningRobot(
                robot_id="ROBOT-002",
                tenant_id="tenant_alpha",
                capabilities=frozenset({
                    "inspection",
                    "diagnostics",
                }),
                battery_percentage=85,
                availability=RobotAvailability.AVAILABLE,
                safety_state=RobotSafetyState.SAFE,
                allowed_zones=frozenset({
                    "ZONE-A",
                }),
                energy_capacity_wh=1000,
                hourly_operating_cost=20,
            ),
        ]

        tasks = [
            PlanningTask(
                task_id="TASK-001",
                tenant_id="tenant_alpha",
                name="Visual Inspection",
                duration_minutes=30,
                required_capabilities=frozenset({
                    "inspection",
                }),
                required_zone="ZONE-A",
                estimated_energy_wh=40,
                latest_finish_minute=120,
            ),
            PlanningTask(
                task_id="TASK-002",
                tenant_id="tenant_alpha",
                name="Robot Diagnostics",
                duration_minutes=20,
                required_capabilities=frozenset({
                    "diagnostics",
                }),
                required_zone="ZONE-A",
                estimated_energy_wh=25,
                dependencies=("TASK-001",),
                latest_finish_minute=120,
            ),
        ]

        planner = RoboticsConstraintPlanner(
            solver_config=self.config,
            time_step_minutes=10,
        )

        result = planner.plan(
            tenant_id="tenant_alpha",
            robots=robots,
            tasks=tasks,
            access=access,
        )

        self.assertIn(
            result.status,
            {
                SolverStatus.OPTIMAL,
                SolverStatus.FEASIBLE,
            },
        )

        self.assertEqual(
            len(result.assignments),
            2,
        )

        assignment_map = {
            item.task_id: item
            for item in result.assignments
        }

        self.assertEqual(
            assignment_map["TASK-002"].robot_id,
            "ROBOT-002",
        )

        self.assertGreaterEqual(
            assignment_map["TASK-002"].start_minute,
            assignment_map["TASK-001"].finish_minute,
        )

    def test_unsafe_robot_rejected(self) -> None:
        robot = PlanningRobot(
            robot_id="ROBOT-001",
            tenant_id="tenant_alpha",
            capabilities=frozenset({
                "inspection",
            }),
            battery_percentage=80,
            safety_state=RobotSafetyState.UNSAFE,
        )

        task = PlanningTask(
            task_id="TASK-001",
            tenant_id="tenant_alpha",
            name="Inspection",
            duration_minutes=10,
            required_capabilities=frozenset({
                "inspection",
            }),
        )

        eligible, reasons = RobotEligibilityEvaluator.evaluate(
            robot,
            task,
            approvals=frozenset(),
            minimum_battery_percentage=10,
            maximum_robot_state_age_seconds=60,
        )

        self.assertFalse(
            eligible
        )

        self.assertTrue(
            reasons
        )

    def test_tenant_isolation(self) -> None:
        engine = RoboticsConstraintSolverEngine(
            config=self.config
        )

        problem = ConstraintSatisfactionProblem()

        problem.add_variable(
            ConstraintVariable(
                name="x",
                domain=(1, 2),
            )
        )

        access = PlanningAccessContext(
            tenant_id="tenant_alpha",
            user_id="planner",
            can_plan=True,
            can_view_results=True,
        )

        result = engine.solve(
            problem,
            access=access,
        )

        other_access = PlanningAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            can_view_results=True,
        )

        with self.assertRaises(
            ConstraintAuthorizationError
        ):
            engine.get_result(
                result.request_id,
                access=other_access,
            )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Constraint Solver — Multi-Robot Planning\n"
    )

    access = PlanningAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_planner",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
            "ROBOT-002",
            "ROBOT-003",
        }),
        roles=frozenset({
            "robotics_engineer",
        }),
        can_plan=True,
        can_modify_constraints=True,
        can_view_results=True,
    )

    robots = [
        PlanningRobot(
            robot_id="ROBOT-001",
            tenant_id="tenant_alpha",
            capabilities=frozenset({
                "inspection",
                "diagnostics",
            }),
            battery_percentage=90,
            safety_state=RobotSafetyState.SAFE,
            allowed_zones=frozenset({
                "ZONE-A",
                "ZONE-B",
            }),
            energy_capacity_wh=1500,
            maximum_payload_kg=20,
            hourly_operating_cost=20,
        ),
        PlanningRobot(
            robot_id="ROBOT-002",
            tenant_id="tenant_alpha",
            capabilities=frozenset({
                "inspection",
                "navigation",
            }),
            battery_percentage=85,
            safety_state=RobotSafetyState.SAFE,
            allowed_zones=frozenset({
                "ZONE-A",
                "ZONE-C",
            }),
            energy_capacity_wh=1200,
            maximum_payload_kg=15,
            hourly_operating_cost=15,
        ),
        PlanningRobot(
            robot_id="ROBOT-003",
            tenant_id="tenant_alpha",
            capabilities=frozenset({
                "inspection",
                "diagnostics",
                "navigation",
            }),
            battery_percentage=95,
            safety_state=RobotSafetyState.SAFE,
            allowed_zones=frozenset({
                "ZONE-A",
                "ZONE-B",
                "ZONE-C",
            }),
            energy_capacity_wh=2000,
            maximum_payload_kg=30,
            hourly_operating_cost=25,
        ),
    ]

    tasks = [
        PlanningTask(
            task_id="TASK-001",
            tenant_id="tenant_alpha",
            name="Inspect Robot Motor",
            duration_minutes=30,
            required_capabilities=frozenset({
                "inspection",
            }),
            required_zone="ZONE-A",
            estimated_energy_wh=50,
            earliest_start_minute=0,
            latest_finish_minute=180,
            priority=3,
        ),
        PlanningTask(
            task_id="TASK-002",
            tenant_id="tenant_alpha",
            name="Run Motor Diagnostics",
            duration_minutes=20,
            required_capabilities=frozenset({
                "diagnostics",
            }),
            required_zone="ZONE-B",
            estimated_energy_wh=35,
            earliest_start_minute=0,
            latest_finish_minute=180,
            dependencies=("TASK-001",),
            priority=2,
        ),
        PlanningTask(
            task_id="TASK-003",
            tenant_id="tenant_alpha",
            name="Simulated Workspace Navigation",
            duration_minutes=25,
            required_capabilities=frozenset({
                "navigation",
            }),
            required_zone="ZONE-C",
            estimated_energy_wh=60,
            earliest_start_minute=0,
            latest_finish_minute=180,
            requires_human_approval=True,
            approval_id="APPROVAL-NAV-001",
            priority=2,
        ),
    ]

    engine = RoboticsConstraintSolverEngine(
        config=ConstraintSolverConfig(
            timeout_seconds=10,
            maximum_search_nodes=100_000,
        )
    )

    result = engine.plan_robots(
        tenant_id="tenant_alpha",
        robots=robots,
        tasks=tasks,
        access=access,
        approvals=frozenset({
            "APPROVAL-NAV-001",
        }),
    )

    print("PLANNING RESULT\n")

    print(
        json.dumps(
            json_safe(result),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nPLANNING STATISTICS\n")

    print(
        json.dumps(
            engine.statistics(
                access=access,
            ),
            indent=2,
        )
    )

    print("\nAUDIT HISTORY\n")

    print(
        json.dumps(
            engine.audit_history(
                access=access,
            ),
            indent=2,
        )
    )

    print("\nUNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestConstraintSolver
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()