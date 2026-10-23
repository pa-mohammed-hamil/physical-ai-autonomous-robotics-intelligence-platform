"""Evaluation Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/evaluation_service.py

Robotics Evaluation and Benchmarking Service

Features:
    - Robot navigation evaluation
    - Task success evaluation
    - Collision and safety evaluation
    - Energy efficiency evaluation
    - Robot health evaluation
    - Fault recovery evaluation
    - Weighted evaluation scorecards
    - Hard safety gates
    - Fleet and scenario benchmarking
    - ML prediction evaluation
    - Evaluation history
    - JSON and CSV reporting
    - Integration with physics simulation
    - Simulation-only operation

Python 3.11+
Standard library only.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import uuid

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class EvaluationServiceError(Exception):
    pass


class EvaluationValidationError(EvaluationServiceError):
    pass


class EvaluationNotFoundError(EvaluationServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class EvaluationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class EvaluationMetric(str, Enum):
    NAVIGATION = "navigation"
    TASK_SUCCESS = "task_success"
    SAFETY = "safety"
    ENERGY = "energy"
    HEALTH = "health"
    FAULT_RECOVERY = "fault_recovery"


class PredictionTask(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def finite(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise EvaluationValidationError(
            f"{name} must be a finite number"
        )

    return float(value)


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    return (
        numerator / denominator
        if denominator != 0
        else default
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class EvaluationServiceConfig:
    navigation_weight: float = 0.25
    task_weight: float = 0.20
    safety_weight: float = 0.25
    energy_weight: float = 0.10
    health_weight: float = 0.10
    fault_recovery_weight: float = 0.10

    passing_score: float = 75.0

    goal_tolerance_m: float = 0.35

    maximum_allowed_collisions: int = 0

    target_energy_per_meter_wh: float = 0.15

    minimum_battery_soc_percent: float = 10.0

    minimum_health_score: float = 70.0

    enforce_collision_gate: bool = True
    enforce_battery_gate: bool = True
    enforce_health_gate: bool = True

    maximum_history: int = 10000

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        weights = [
            self.navigation_weight,
            self.task_weight,
            self.safety_weight,
            self.energy_weight,
            self.health_weight,
            self.fault_recovery_weight,
        ]

        for weight in weights:
            finite(weight, "evaluation_weight")

            if weight < 0:
                raise EvaluationValidationError(
                    "Weights cannot be negative"
                )

        if not math.isclose(
            sum(weights),
            1.0,
            abs_tol=1e-9,
        ):
            raise EvaluationValidationError(
                "Evaluation weights must sum to 1"
            )

        if not 0 <= self.passing_score <= 100:
            raise EvaluationValidationError(
                "Invalid passing score"
            )

        if self.goal_tolerance_m <= 0:
            raise EvaluationValidationError(
                "Goal tolerance must be positive"
            )

        if self.target_energy_per_meter_wh <= 0:
            raise EvaluationValidationError(
                "Energy target must be positive"
            )

        if self.maximum_allowed_collisions < 0:
            raise EvaluationValidationError(
                "Maximum collisions cannot be negative"
            )

        if self.maximum_history < 1:
            raise EvaluationValidationError(
                "Maximum history must be positive"
            )

        if self.allow_hardware_control:
            raise EvaluationValidationError(
                "Hardware control is prohibited"
            )


# ============================================================
# ROBOT EVALUATION INPUT
# ============================================================

@dataclass(frozen=True)
class RobotEvaluationData:
    robot_id: str

    start_x: float
    start_y: float

    final_x: float
    final_y: float

    goal_x: float
    goal_y: float

    distance_travelled_m: float

    energy_consumed_wh: float

    battery_soc_percent: float

    collision_count: int

    tasks_completed: int
    tasks_total: int

    health_score: float

    faults_injected: int = 0
    faults_recovered: int = 0

    simulation_duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.robot_id.strip():
            raise EvaluationValidationError(
                "robot_id is required"
            )

        for name in (
            "start_x",
            "start_y",
            "final_x",
            "final_y",
            "goal_x",
            "goal_y",
            "distance_travelled_m",
            "energy_consumed_wh",
            "battery_soc_percent",
            "health_score",
            "simulation_duration_seconds",
        ):
            finite(
                getattr(self, name),
                name,
            )

        for name in (
            "distance_travelled_m",
            "energy_consumed_wh",
            "simulation_duration_seconds",
        ):
            if getattr(self, name) < 0:
                raise EvaluationValidationError(
                    f"{name} cannot be negative"
                )

        if not 0 <= self.battery_soc_percent <= 100:
            raise EvaluationValidationError(
                "Invalid battery SOC"
            )

        if not 0 <= self.health_score <= 100:
            raise EvaluationValidationError(
                "Invalid health score"
            )

        for name in (
            "collision_count",
            "tasks_completed",
            "tasks_total",
            "faults_injected",
            "faults_recovered",
        ):
            if getattr(self, name) < 0:
                raise EvaluationValidationError(
                    f"{name} cannot be negative"
                )

        if self.tasks_completed > self.tasks_total:
            raise EvaluationValidationError(
                "Completed tasks exceed total tasks"
            )

        if self.faults_recovered > self.faults_injected:
            raise EvaluationValidationError(
                "Recovered faults exceed injected faults"
            )


# ============================================================
# METRIC RESULT
# ============================================================

@dataclass(frozen=True)
class MetricEvaluationResult:
    metric: EvaluationMetric

    score: float

    passed: bool

    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "score": round(self.score, 4),
            "passed": self.passed,
            "details": self.details,
        }


# ============================================================
# ROBOT RESULT
# ============================================================

@dataclass
class RobotEvaluationResult:
    evaluation_id: str

    robot_id: str

    status: EvaluationStatus

    overall_score: float

    metric_results: list[
        MetricEvaluationResult
    ]

    safety_violations: list[str]

    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "robot_id": self.robot_id,
            "status": self.status.value,
            "overall_score": round(
                self.overall_score,
                4,
            ),
            "metric_results": [
                metric.to_dict()
                for metric in self.metric_results
            ],
            "safety_violations": (
                self.safety_violations
            ),
            "evaluated_at": self.evaluated_at,
        }


# ============================================================
# NAVIGATION EVALUATOR
# ============================================================

class NavigationEvaluator:

    def __init__(
        self,
        config: EvaluationServiceConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        robot: RobotEvaluationData,
    ) -> MetricEvaluationResult:
        goal_error = math.hypot(
            robot.goal_x - robot.final_x,
            robot.goal_y - robot.final_y,
        )

        direct_distance = math.hypot(
            robot.goal_x - robot.start_x,
            robot.goal_y - robot.start_y,
        )

        goal_reached = (
            goal_error
            <= self.config.goal_tolerance_m
        )

        if direct_distance <= 1e-9:
            path_efficiency = (
                1.0 if goal_reached else 0.0
            )
        else:
            path_efficiency = clamp(
                direct_distance
                / max(
                    robot.distance_travelled_m,
                    direct_distance,
                ),
                0.0,
                1.0,
            )

        progress = (
            1.0
            if goal_reached
            else clamp(
                1.0
                - goal_error
                / max(
                    direct_distance,
                    self.config.goal_tolerance_m,
                ),
                0.0,
                1.0,
            )
        )

        score = (
            70.0 * progress
            + 30.0 * path_efficiency
        )

        return MetricEvaluationResult(
            metric=EvaluationMetric.NAVIGATION,
            score=score,
            passed=goal_reached,
            details={
                "goal_error_m": goal_error,
                "goal_reached": goal_reached,
                "path_efficiency": path_efficiency,
                "progress": progress,
            },
        )


# ============================================================
# TASK EVALUATOR
# ============================================================

class TaskSuccessEvaluator:

    def evaluate(
        self,
        robot: RobotEvaluationData,
    ) -> MetricEvaluationResult:
        if robot.tasks_total == 0:
            score = 100.0
            success_rate = 1.0
        else:
            success_rate = (
                robot.tasks_completed
                / robot.tasks_total
            )

            score = success_rate * 100.0

        return MetricEvaluationResult(
            metric=EvaluationMetric.TASK_SUCCESS,
            score=score,
            passed=score >= 100.0,
            details={
                "tasks_completed": (
                    robot.tasks_completed
                ),
                "tasks_total": robot.tasks_total,
                "success_rate": success_rate,
            },
        )


# ============================================================
# SAFETY EVALUATOR
# ============================================================

class SafetyEvaluator:

    def __init__(
        self,
        config: EvaluationServiceConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        robot: RobotEvaluationData,
    ) -> MetricEvaluationResult:
        excess_collisions = max(
            0,
            robot.collision_count
            - self.config.maximum_allowed_collisions,
        )

        score = clamp(
            100.0
            - 40.0 * excess_collisions
        )

        passed = (
            robot.collision_count
            <= self.config.maximum_allowed_collisions
        )

        return MetricEvaluationResult(
            metric=EvaluationMetric.SAFETY,
            score=score,
            passed=passed,
            details={
                "collision_count": (
                    robot.collision_count
                ),
                "maximum_allowed_collisions": (
                    self.config.maximum_allowed_collisions
                ),
            },
        )


# ============================================================
# ENERGY EVALUATOR
# ============================================================

class EnergyEvaluator:

    def __init__(
        self,
        config: EvaluationServiceConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        robot: RobotEvaluationData,
    ) -> MetricEvaluationResult:
        if robot.distance_travelled_m <= 1e-9:
            energy_per_meter = None

            score = (
                100.0
                if robot.energy_consumed_wh <= 1e-9
                else 0.0
            )

        else:
            energy_per_meter = (
                robot.energy_consumed_wh
                / robot.distance_travelled_m
            )

            score = clamp(
                100.0
                * self.config.target_energy_per_meter_wh
                / max(
                    energy_per_meter,
                    1e-9,
                )
            )

        return MetricEvaluationResult(
            metric=EvaluationMetric.ENERGY,
            score=score,
            passed=score >= 75.0,
            details={
                "energy_consumed_wh": (
                    robot.energy_consumed_wh
                ),
                "energy_per_meter_wh": (
                    energy_per_meter
                ),
            },
        )


# ============================================================
# HEALTH EVALUATOR
# ============================================================

class HealthEvaluator:

    def __init__(
        self,
        config: EvaluationServiceConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        robot: RobotEvaluationData,
    ) -> MetricEvaluationResult:
        return MetricEvaluationResult(
            metric=EvaluationMetric.HEALTH,
            score=robot.health_score,
            passed=(
                robot.health_score
                >= self.config.minimum_health_score
            ),
            details={
                "health_score": robot.health_score,
                "battery_soc_percent": (
                    robot.battery_soc_percent
                ),
            },
        )


# ============================================================
# FAULT RECOVERY EVALUATOR
# ============================================================

class FaultRecoveryEvaluator:

    def evaluate(
        self,
        robot: RobotEvaluationData,
    ) -> MetricEvaluationResult:
        if robot.faults_injected == 0:
            recovery_rate = 1.0
        else:
            recovery_rate = (
                robot.faults_recovered
                / robot.faults_injected
            )

        score = recovery_rate * 100.0

        return MetricEvaluationResult(
            metric=EvaluationMetric.FAULT_RECOVERY,
            score=score,
            passed=score >= 80.0,
            details={
                "faults_injected": (
                    robot.faults_injected
                ),
                "faults_recovered": (
                    robot.faults_recovered
                ),
                "recovery_rate": recovery_rate,
            },
        )


# ============================================================
# UNIFIED ROBOT EVALUATOR
# ============================================================

class RoboticsRobotEvaluator:

    def __init__(
        self,
        config: EvaluationServiceConfig,
    ) -> None:
        self.config = config

        self.evaluators = [
            NavigationEvaluator(config),
            TaskSuccessEvaluator(),
            SafetyEvaluator(config),
            EnergyEvaluator(config),
            HealthEvaluator(config),
            FaultRecoveryEvaluator(),
        ]

        self.weights = {
            EvaluationMetric.NAVIGATION: (
                config.navigation_weight
            ),
            EvaluationMetric.TASK_SUCCESS: (
                config.task_weight
            ),
            EvaluationMetric.SAFETY: (
                config.safety_weight
            ),
            EvaluationMetric.ENERGY: (
                config.energy_weight
            ),
            EvaluationMetric.HEALTH: (
                config.health_weight
            ),
            EvaluationMetric.FAULT_RECOVERY: (
                config.fault_recovery_weight
            ),
        }

    def evaluate(
        self,
        robot: RobotEvaluationData,
    ) -> RobotEvaluationResult:
        metric_results = [
            evaluator.evaluate(robot)
            for evaluator in self.evaluators
        ]

        overall_score = sum(
            metric.score
            * self.weights[metric.metric]
            for metric in metric_results
        )

        violations = []

        if (
            self.config.enforce_collision_gate
            and robot.collision_count
            > self.config.maximum_allowed_collisions
        ):
            violations.append(
                "Collision safety gate failed"
            )

        if (
            self.config.enforce_battery_gate
            and robot.battery_soc_percent
            < self.config.minimum_battery_soc_percent
        ):
            violations.append(
                "Battery safety gate failed"
            )

        if (
            self.config.enforce_health_gate
            and robot.health_score
            < self.config.minimum_health_score
        ):
            violations.append(
                "Robot health gate failed"
            )

        status = (
            EvaluationStatus.PASSED
            if (
                overall_score
                >= self.config.passing_score
                and not violations
            )
            else EvaluationStatus.FAILED
        )

        return RobotEvaluationResult(
            evaluation_id=str(uuid.uuid4()),
            robot_id=robot.robot_id,
            status=status,
            overall_score=overall_score,
            metric_results=metric_results,
            safety_violations=violations,
            evaluated_at=utc_now(),
        )


# ============================================================
# SCENARIO EVALUATION RESULT
# ============================================================

@dataclass
class ScenarioEvaluationReport:
    report_id: str
    scenario_id: str

    status: EvaluationStatus

    overall_score: float

    robot_results: list[
        RobotEvaluationResult
    ]

    passed_robots: int
    failed_robots: int

    evaluated_at: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scenario_id": self.scenario_id,
            "status": self.status.value,
            "overall_score": round(
                self.overall_score,
                4,
            ),
            "robot_results": [
                result.to_dict()
                for result in self.robot_results
            ],
            "passed_robots": self.passed_robots,
            "failed_robots": self.failed_robots,
            "evaluated_at": self.evaluated_at,
            "metadata": self.metadata,
        }


# ============================================================
# SCENARIO EVALUATOR
# ============================================================

class RoboticsScenarioEvaluationService:

    def __init__(
        self,
        config: EvaluationServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or EvaluationServiceConfig()
        )

        self.robot_evaluator = (
            RoboticsRobotEvaluator(
                self.config
            )
        )

        self.history: deque[
            ScenarioEvaluationReport
        ] = deque(
            maxlen=self.config.maximum_history
        )

        self.reports: dict[
            str,
            ScenarioEvaluationReport
        ] = {}

    def evaluate(
        self,
        *,
        scenario_id: str,
        robots: Sequence[RobotEvaluationData],
        metadata: Mapping[str, Any] | None = None,
    ) -> ScenarioEvaluationReport:
        if not scenario_id.strip():
            raise EvaluationValidationError(
                "scenario_id is required"
            )

        if not robots:
            raise EvaluationValidationError(
                "At least one robot is required"
            )

        robot_ids = [
            robot.robot_id
            for robot in robots
        ]

        if len(robot_ids) != len(set(robot_ids)):
            raise EvaluationValidationError(
                "Duplicate robot IDs"
            )

        results = [
            self.robot_evaluator.evaluate(robot)
            for robot in robots
        ]

        overall_score = statistics.mean(
            result.overall_score
            for result in results
        )

        passed_count = sum(
            result.status == EvaluationStatus.PASSED
            for result in results
        )

        failed_count = (
            len(results) - passed_count
        )

        status = (
            EvaluationStatus.PASSED
            if failed_count == 0
            else EvaluationStatus.FAILED
        )

        report = ScenarioEvaluationReport(
            report_id=str(uuid.uuid4()),
            scenario_id=scenario_id,
            status=status,
            overall_score=overall_score,
            robot_results=results,
            passed_robots=passed_count,
            failed_robots=failed_count,
            evaluated_at=utc_now(),
            metadata=dict(metadata or {}),
        )

        self.history.append(report)

        self.reports[
            report.report_id
        ] = report

        return report

    def get_report(
        self,
        report_id: str,
    ) -> ScenarioEvaluationReport:
        report = self.reports.get(
            report_id
        )

        if report is None:
            raise EvaluationNotFoundError(
                f"Report not found: {report_id}"
            )

        return report

    def benchmark_summary(
        self,
    ) -> dict[str, Any]:
        reports = list(
            self.history
        )

        if not reports:
            return {
                "total_evaluations": 0
            }

        return {
            "total_evaluations": len(reports),
            "average_score": statistics.mean(
                report.overall_score
                for report in reports
            ),
            "pass_rate": safe_divide(
                sum(
                    report.status
                    == EvaluationStatus.PASSED
                    for report in reports
                ),
                len(reports),
            ),
            "best_score": max(
                report.overall_score
                for report in reports
            ),
            "worst_score": min(
                report.overall_score
                for report in reports
            ),
        }


# ============================================================
# ML MODEL EVALUATION
# ============================================================

class RoboticsMLEvaluator:

    @staticmethod
    def classification(
        y_true: Sequence[int],
        y_pred: Sequence[int],
    ) -> dict[str, float]:
        if not y_true or len(y_true) != len(y_pred):
            raise EvaluationValidationError(
                "Invalid classification arrays"
            )

        if any(
            value not in (0, 1)
            for value in (*y_true, *y_pred)
        ):
            raise EvaluationValidationError(
                "Binary classification requires 0/1 labels"
            )

        tp = sum(
            truth == 1 and prediction == 1
            for truth, prediction
            in zip(y_true, y_pred)
        )

        tn = sum(
            truth == 0 and prediction == 0
            for truth, prediction
            in zip(y_true, y_pred)
        )

        fp = sum(
            truth == 0 and prediction == 1
            for truth, prediction
            in zip(y_true, y_pred)
        )

        fn = sum(
            truth == 1 and prediction == 0
            for truth, prediction
            in zip(y_true, y_pred)
        )

        precision = safe_divide(
            tp,
            tp + fp,
        )

        recall = safe_divide(
            tp,
            tp + fn,
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall,
        )

        return {
            "accuracy": safe_divide(
                tp + tn,
                len(y_true),
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    @staticmethod
    def regression(
        y_true: Sequence[float],
        y_pred: Sequence[float],
    ) -> dict[str, float | None]:
        if not y_true or len(y_true) != len(y_pred):
            raise EvaluationValidationError(
                "Invalid regression arrays"
            )

        actual = [
            finite(value, "y_true")
            for value in y_true
        ]

        predicted = [
            finite(value, "y_pred")
            for value in y_pred
        ]

        errors = [
            truth - prediction
            for truth, prediction
            in zip(actual, predicted)
        ]

        mae = statistics.mean(
            abs(error)
            for error in errors
        )

        mse = statistics.mean(
            error ** 2
            for error in errors
        )

        rmse = math.sqrt(mse)

        mean_actual = statistics.mean(
            actual
        )

        ss_total = sum(
            (value - mean_actual) ** 2
            for value in actual
        )

        ss_residual = sum(
            error ** 2
            for error in errors
        )

        r2 = (
            1.0 - ss_residual / ss_total
            if ss_total > 0
            else None
        )

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }


# ============================================================
# PHYSICS ADAPTER
# ============================================================

class PhysicsEvaluationAdapter:

    @staticmethod
    def extract(
        physics: Any,
        *,
        goals: Mapping[
            str,
            tuple[float, float]
        ],
        start_positions: Mapping[
            str,
            tuple[float, float]
        ],
        completed_tasks: Mapping[
            str,
            int
        ] | None = None,
        total_tasks: Mapping[
            str,
            int
        ] | None = None,
        health_scores: Mapping[
            str,
            float
        ] | None = None,
        fault_counts: Mapping[
            str,
            tuple[int, int]
        ] | None = None,
    ) -> list[RobotEvaluationData]:
        completed_tasks = completed_tasks or {}
        total_tasks = total_tasks or {}
        health_scores = health_scores or {}
        fault_counts = fault_counts or {}

        results = []

        for robot_id, robot in physics.robots.items():
            start = start_positions.get(
                robot_id,
                (
                    robot.pose.x,
                    robot.pose.y,
                ),
            )

            goal = goals.get(
                robot_id,
                start,
            )

            injected, recovered = (
                fault_counts.get(
                    robot_id,
                    (0, 0),
                )
            )

            results.append(
                RobotEvaluationData(
                    robot_id=robot_id,
                    start_x=start[0],
                    start_y=start[1],
                    final_x=robot.pose.x,
                    final_y=robot.pose.y,
                    goal_x=goal[0],
                    goal_y=goal[1],
                    distance_travelled_m=(
                        robot.distance_travelled_m
                    ),
                    energy_consumed_wh=(
                        robot.energy_consumed_wh
                    ),
                    battery_soc_percent=(
                        robot.battery_soc_percent()
                    ),
                    collision_count=(
                        robot.collision_count
                    ),
                    tasks_completed=(
                        completed_tasks.get(
                            robot_id,
                            0,
                        )
                    ),
                    tasks_total=(
                        total_tasks.get(
                            robot_id,
                            0,
                        )
                    ),
                    health_score=(
                        health_scores.get(
                            robot_id,
                            100.0,
                        )
                    ),
                    faults_injected=injected,
                    faults_recovered=recovered,
                    simulation_duration_seconds=(
                        physics.simulation_time_seconds
                    ),
                )
            )

        return results


# ============================================================
# REPORT EXPORTER
# ============================================================

class RoboticsEvaluationReportExporter:

    @staticmethod
    def export_json(
        report: ScenarioEvaluationReport,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                report.to_dict(),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output

    @staticmethod
    def export_csv(
        report: ScenarioEvaluationReport,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "scenario_id",
                    "robot_id",
                    "status",
                    "overall_score",
                    "metric",
                    "metric_score",
                    "metric_passed",
                    "safety_violations",
                ],
            )

            writer.writeheader()

            for robot in report.robot_results:
                for metric in robot.metric_results:
                    writer.writerow({
                        "scenario_id": report.scenario_id,
                        "robot_id": robot.robot_id,
                        "status": robot.status.value,
                        "overall_score": (
                            robot.overall_score
                        ),
                        "metric": metric.metric.value,
                        "metric_score": metric.score,
                        "metric_passed": metric.passed,
                        "safety_violations": (
                            "; ".join(
                                robot.safety_violations
                            )
                        ),
                    })

        return output


# ============================================================
# UNIFIED EVALUATION SERVICE
# ============================================================

class RoboticsEvaluationService:

    def __init__(
        self,
        config: EvaluationServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or EvaluationServiceConfig()
        )

        self.scenario_service = (
            RoboticsScenarioEvaluationService(
                self.config
            )
        )

        self.ml_evaluator = RoboticsMLEvaluator()

    def evaluate_robot(
        self,
        robot: RobotEvaluationData,
    ) -> RobotEvaluationResult:
        return self.scenario_service.robot_evaluator.evaluate(
            robot
        )

    def evaluate_scenario(
        self,
        scenario_id: str,
        robots: Sequence[RobotEvaluationData],
        metadata: Mapping[str, Any] | None = None,
    ) -> ScenarioEvaluationReport:
        return self.scenario_service.evaluate(
            scenario_id=scenario_id,
            robots=robots,
            metadata=metadata,
        )

    def evaluate_physics(
        self,
        *,
        scenario_id: str,
        physics: Any,
        goals: Mapping[
            str,
            tuple[float, float]
        ],
        start_positions: Mapping[
            str,
            tuple[float, float]
        ],
    ) -> ScenarioEvaluationReport:
        robots = PhysicsEvaluationAdapter.extract(
            physics,
            goals=goals,
            start_positions=start_positions,
        )

        return self.evaluate_scenario(
            scenario_id,
            robots,
        )

    def benchmark_summary(
        self,
    ) -> dict[str, Any]:
        return self.scenario_service.benchmark_summary()


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsEvaluationService()

    robots = [
        RobotEvaluationData(
            robot_id="robot_001",
            start_x=1.0,
            start_y=1.0,
            final_x=9.9,
            final_y=10.0,
            goal_x=10.0,
            goal_y=10.0,
            distance_travelled_m=13.5,
            energy_consumed_wh=2.0,
            battery_soc_percent=85.0,
            collision_count=0,
            tasks_completed=5,
            tasks_total=5,
            health_score=95.0,
            faults_injected=2,
            faults_recovered=2,
            simulation_duration_seconds=120.0,
        ),
        RobotEvaluationData(
            robot_id="robot_002",
            start_x=2.0,
            start_y=2.0,
            final_x=7.0,
            final_y=8.0,
            goal_x=10.0,
            goal_y=10.0,
            distance_travelled_m=10.0,
            energy_consumed_wh=3.0,
            battery_soc_percent=60.0,
            collision_count=1,
            tasks_completed=3,
            tasks_total=5,
            health_score=75.0,
            faults_injected=2,
            faults_recovered=1,
            simulation_duration_seconds=120.0,
        ),
    ]

    report = service.evaluate_scenario(
        "warehouse_navigation_001",
        robots,
        metadata={
            "environment": "warehouse",
            "simulation_only": True,
        },
    )

    classification_metrics = (
        RoboticsMLEvaluator.classification(
            y_true=[0, 1, 1, 0, 1, 0],
            y_pred=[0, 1, 0, 0, 1, 1],
        )
    )

    regression_metrics = (
        RoboticsMLEvaluator.regression(
            y_true=[100, 90, 80, 70],
            y_pred=[98, 92, 78, 73],
        )
    )

    RoboticsEvaluationReportExporter.export_json(
        report,
        "data/evaluation/evaluation_report.json",
    )

    RoboticsEvaluationReportExporter.export_csv(
        report,
        "data/evaluation/evaluation_metrics.csv",
    )

    print(
        json.dumps(
            {
                "scenario_evaluation": report.to_dict(),
                "classification_metrics": (
                    classification_metrics
                ),
                "regression_metrics": (
                    regression_metrics
                ),
                "benchmark_summary": (
                    service.benchmark_summary()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
