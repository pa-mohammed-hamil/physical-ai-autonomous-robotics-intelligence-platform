
#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/simulate_collision_risk.py

Offline robot collision-risk simulation and evaluation.

Features:
    - Mobile robots, robotic arms, humanoids, quadrupeds
    - Multiple robots and dynamic obstacles
    - Human proximity simulation
    - 2D position and velocity tracking
    - Constant-velocity trajectory prediction
    - Time-to-closest-approach (TCA)
    - Time-to-collision (TTC)
    - Collision clearance estimation
    - Stopping-distance analysis
    - Monte Carlo uncertainty simulation
    - Collision-risk classification
    - Safety-event generation
    - Scenario configuration via JSON
    - JSON and CSV reporting
    - Deterministic random seed
    - CLI support
    - No physical robot control

Dependencies:
    Python 3.10+
    Standard library only

Examples:
    python scripts/simulate_collision_risk.py

    python scripts/simulate_collision_risk.py \
        --robots 5 --obstacles 12 --steps 100

    python scripts/simulate_collision_risk.py \
        --scenario simulation/scenarios/navigation.yaml

    python scripts/simulate_collision_risk.py \
        --scenario scenarios/collision.json

    python scripts/simulate_collision_risk.py \
        --monte-carlo 100

    python scripts/simulate_collision_risk.py \
        --output reports/collision_risk.json \
        --csv reports/collision_risk.csv

Note:
    YAML scenario files require PyYAML:
        pip install pyyaml

Safety:
    This is a simplified simulation, not a safety-rated
    collision-avoidance controller. It must not be used
    to authorize physical robot movement.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("collision_simulation")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

EPSILON = 1e-9


# ============================================================
# Enums
# ============================================================

class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EntityType(str, Enum):
    ROBOT = "robot"
    OBSTACLE = "obstacle"
    HUMAN = "human"


# ============================================================
# Geometry
# ============================================================

@dataclass
class Vector2:
    x: float
    y: float

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(
            self.x - other.x,
            self.y - other.y,
        )

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(
            self.x * scalar,
            self.y * scalar,
        )

    def dot(self, other: "Vector2") -> float:
        return (
            self.x * other.x
            + self.y * other.y
        )

    def magnitude(self) -> float:
        return math.hypot(
            self.x,
            self.y,
        )

    def distance_to(self, other: "Vector2") -> float:
        return (self - other).magnitude()

    def normalized(self) -> "Vector2":
        magnitude = self.magnitude()

        if magnitude <= EPSILON:
            return Vector2(0.0, 0.0)

        return Vector2(
            self.x / magnitude,
            self.y / magnitude,
        )


# ============================================================
# Entity Model
# ============================================================

@dataclass
class SimulationEntity:
    entity_id: str
    entity_type: EntityType
    position: Vector2
    velocity: Vector2

    radius_m: float = 0.4
    mass_kg: float = 50.0

    robot_type: str | None = None
    active: bool = True

    def predict_position(self, time_s: float) -> Vector2:
        return (
            self.position
            + self.velocity * time_s
        )

    def update(self, delta_time: float) -> None:
        self.position = self.predict_position(
            delta_time
        )


# ============================================================
# Safety Configuration
# ============================================================

@dataclass
class SafetyConfig:
    prediction_horizon_s: float = 5.0

    minimum_clearance_m: float = 0.3
    human_clearance_m: float = 1.0

    reaction_time_s: float = 0.4
    braking_deceleration_m_s2: float = 1.5

    critical_ttc_s: float = 0.75
    high_ttc_s: float = 1.5
    moderate_ttc_s: float = 3.0

    monte_carlo_position_sigma_m: float = 0.1
    monte_carlo_velocity_sigma_m_s: float = 0.05

    human_risk_multiplier: float = 1.4

    def validate(self) -> None:
        positive_fields = {
            "prediction_horizon_s": self.prediction_horizon_s,
            "minimum_clearance_m": self.minimum_clearance_m,
            "human_clearance_m": self.human_clearance_m,
            "reaction_time_s": self.reaction_time_s,
            "braking_deceleration_m_s2": (
                self.braking_deceleration_m_s2
            ),
            "critical_ttc_s": self.critical_ttc_s,
            "high_ttc_s": self.high_ttc_s,
            "moderate_ttc_s": self.moderate_ttc_s,
        }

        for name, value in positive_fields.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if not (
            self.critical_ttc_s
            < self.high_ttc_s
            < self.moderate_ttc_s
        ):
            raise ValueError(
                "TTC thresholds must be strictly increasing"
            )

        for name in (
            "monte_carlo_position_sigma_m",
            "monte_carlo_velocity_sigma_m_s",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite and nonnegative"
                )


# ============================================================
# Collision Risk Result
# ============================================================

@dataclass
class CollisionRiskResult:
    robot_id: str
    target_id: str
    target_type: str

    timestamp_s: float

    current_distance_m: float
    current_clearance_m: float

    minimum_predicted_distance_m: float
    minimum_predicted_clearance_m: float

    time_to_closest_approach_s: float
    time_to_collision_s: float | None

    relative_speed_m_s: float

    stopping_distance_m: float

    collision_probability: float

    risk_score: float
    risk_level: str

    collision_predicted: bool
    human_proximity_risk: bool

    recommended_action: str

    simulation_only: bool = True


# ============================================================
# Collision Mathematics
# ============================================================

def time_to_closest_approach(
    relative_position: Vector2,
    relative_velocity: Vector2,
    horizon_s: float,
) -> float:
    speed_squared = relative_velocity.dot(
        relative_velocity
    )

    if speed_squared <= EPSILON:
        return 0.0

    time_s = (
        -relative_position.dot(relative_velocity)
        / speed_squared
    )

    return max(
        0.0,
        min(horizon_s, time_s),
    )


def minimum_predicted_distance(
    robot: SimulationEntity,
    target: SimulationEntity,
    horizon_s: float,
) -> tuple[float, float]:
    relative_position = (
        target.position - robot.position
    )

    relative_velocity = (
        target.velocity - robot.velocity
    )

    tca = time_to_closest_approach(
        relative_position,
        relative_velocity,
        horizon_s,
    )

    closest_relative_position = (
        relative_position
        + relative_velocity * tca
    )

    return (
        closest_relative_position.magnitude(),
        tca,
    )


def calculate_time_to_collision(
    robot: SimulationEntity,
    target: SimulationEntity,
    safety_margin_m: float,
    horizon_s: float,
) -> float | None:
    relative_position = (
        target.position - robot.position
    )

    relative_velocity = (
        target.velocity - robot.velocity
    )

    combined_radius = (
        robot.radius_m
        + target.radius_m
        + safety_margin_m
    )

    a = relative_velocity.dot(
        relative_velocity
    )

    b = 2.0 * relative_position.dot(
        relative_velocity
    )

    c = (
        relative_position.dot(relative_position)
        - combined_radius ** 2
    )

    if c <= 0:
        return 0.0

    if a <= EPSILON:
        return None

    discriminant = (
        b * b - 4.0 * a * c
    )

    if discriminant < 0:
        return None

    sqrt_discriminant = math.sqrt(
        discriminant
    )

    t1 = (
        -b - sqrt_discriminant
    ) / (2.0 * a)

    t2 = (
        -b + sqrt_discriminant
    ) / (2.0 * a)

    valid_times = [
        t
        for t in (t1, t2)
        if 0 <= t <= horizon_s
    ]

    if not valid_times:
        return None

    return min(valid_times)


def calculate_stopping_distance(
    speed_m_s: float,
    reaction_time_s: float,
    deceleration_m_s2: float,
) -> float:
    speed = max(
        0.0,
        speed_m_s,
    )

    reaction_distance = (
        speed * reaction_time_s
    )

    braking_distance = (
        speed ** 2
        / (2.0 * deceleration_m_s2)
    )

    return (
        reaction_distance
        + braking_distance
    )


# ============================================================
# Monte Carlo Collision Probability
# ============================================================

class MonteCarloRiskEstimator:
    def __init__(
        self,
        config: SafetyConfig,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.rng = rng

    def perturb_entity(
        self,
        entity: SimulationEntity,
    ) -> SimulationEntity:
        position_sigma = (
            self.config.monte_carlo_position_sigma_m
        )

        velocity_sigma = (
            self.config.monte_carlo_velocity_sigma_m_s
        )

        return SimulationEntity(
            entity_id=entity.entity_id,
            entity_type=entity.entity_type,
            position=Vector2(
                entity.position.x
                + self.rng.gauss(0, position_sigma),
                entity.position.y
                + self.rng.gauss(0, position_sigma),
            ),
            velocity=Vector2(
                entity.velocity.x
                + self.rng.gauss(0, velocity_sigma),
                entity.velocity.y
                + self.rng.gauss(0, velocity_sigma),
            ),
            radius_m=entity.radius_m,
            mass_kg=entity.mass_kg,
            robot_type=entity.robot_type,
            active=entity.active,
        )

    def estimate(
        self,
        robot: SimulationEntity,
        target: SimulationEntity,
        safety_margin_m: float,
        samples: int,
    ) -> float:
        if samples <= 0:
            return 0.0

        collision_count = 0

        for _ in range(samples):
            sampled_robot = self.perturb_entity(
                robot
            )

            sampled_target = self.perturb_entity(
                target
            )

            ttc = calculate_time_to_collision(
                sampled_robot,
                sampled_target,
                safety_margin_m,
                self.config.prediction_horizon_s,
            )

            if ttc is not None:
                collision_count += 1

        return collision_count / samples


# ============================================================
# Risk Classification
# ============================================================

class CollisionRiskEvaluator:
    def __init__(
        self,
        config: SafetyConfig,
        monte_carlo_samples: int = 100,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.config.validate()

        self.monte_carlo_samples = (
            monte_carlo_samples
        )

        self.estimator = MonteCarloRiskEstimator(
            config,
            random.Random(seed),
        )

    def safety_margin(
        self,
        target: SimulationEntity,
    ) -> float:
        if target.entity_type == EntityType.HUMAN:
            return self.config.human_clearance_m

        return self.config.minimum_clearance_m

    def classify(
        self,
        ttc: float | None,
        clearance: float,
        probability: float,
        human_risk: bool,
    ) -> RiskLevel:
        if clearance <= 0:
            return RiskLevel.CRITICAL

        if ttc is not None:
            if ttc <= self.config.critical_ttc_s:
                return RiskLevel.CRITICAL

            if ttc <= self.config.high_ttc_s:
                return RiskLevel.HIGH

            if ttc <= self.config.moderate_ttc_s:
                return RiskLevel.MODERATE

        if probability >= 0.8:
            return RiskLevel.CRITICAL

        if probability >= 0.5:
            return RiskLevel.HIGH

        if probability >= 0.2:
            return RiskLevel.MODERATE

        if human_risk:
            return RiskLevel.MODERATE

        if clearance < 1.0:
            return RiskLevel.LOW

        return RiskLevel.SAFE

    def calculate_score(
        self,
        ttc: float | None,
        clearance: float,
        probability: float,
        human_risk: bool,
    ) -> float:
        if ttc is None:
            ttc_component = 0.0
        else:
            ttc_component = clamp(
                1.0 - (
                    ttc
                    / self.config.prediction_horizon_s
                ),
                0.0,
                1.0,
            )

        clearance_component = clamp(
            1.0 - clearance / 2.0,
            0.0,
            1.0,
        )

        score = (
            0.45 * probability
            + 0.35 * ttc_component
            + 0.20 * clearance_component
        )

        if human_risk:
            score *= self.config.human_risk_multiplier

        return round(
            clamp(score, 0.0, 1.0),
            4,
        )

    def recommended_action(
        self,
        risk: RiskLevel,
        human_risk: bool,
    ) -> str:
        # Advisory simulation labels only.
        # No commands are sent to hardware.
        if risk == RiskLevel.CRITICAL:
            return (
                "SIMULATION: mark unsafe, stop scenario "
                "progression and request human review"
            )

        if risk == RiskLevel.HIGH:
            return (
                "SIMULATION: recommend speed reduction "
                "and trajectory replanning"
            )

        if human_risk:
            return (
                "SIMULATION: increase human separation "
                "and request safety review"
            )

        if risk == RiskLevel.MODERATE:
            return (
                "SIMULATION: evaluate alternative path "
                "and reduce predicted exposure"
            )

        if risk == RiskLevel.LOW:
            return (
                "SIMULATION: continue monitoring clearance"
            )

        return (
            "SIMULATION: no immediate risk identified"
        )

    def evaluate(
        self,
        robot: SimulationEntity,
        target: SimulationEntity,
        timestamp_s: float,
    ) -> CollisionRiskResult:
        margin = self.safety_margin(
            target
        )

        current_distance = robot.position.distance_to(
            target.position
        )

        combined_radius = (
            robot.radius_m
            + target.radius_m
        )

        current_clearance = (
            current_distance - combined_radius
        )

        minimum_distance, tca = (
            minimum_predicted_distance(
                robot,
                target,
                self.config.prediction_horizon_s,
            )
        )

        minimum_clearance = (
            minimum_distance - combined_radius
        )

        ttc = calculate_time_to_collision(
            robot,
            target,
            margin,
            self.config.prediction_horizon_s,
        )

        relative_velocity = (
            target.velocity - robot.velocity
        )

        relative_speed = (
            relative_velocity.magnitude()
        )

        stopping_distance = (
            calculate_stopping_distance(
                robot.velocity.magnitude(),
                self.config.reaction_time_s,
                self.config.braking_deceleration_m_s2,
            )
        )

        collision_probability = (
            self.estimator.estimate(
                robot,
                target,
                margin,
                self.monte_carlo_samples,
            )
        )

        human_risk = (
            target.entity_type == EntityType.HUMAN
            and minimum_clearance
            <= self.config.human_clearance_m
        )

        risk_level = self.classify(
            ttc,
            minimum_clearance - margin,
            collision_probability,
            human_risk,
        )

        risk_score = self.calculate_score(
            ttc,
            minimum_clearance - margin,
            collision_probability,
            human_risk,
        )

        return CollisionRiskResult(
            robot_id=robot.entity_id,
            target_id=target.entity_id,
            target_type=target.entity_type.value,
            timestamp_s=round(timestamp_s, 4),
            current_distance_m=round(
                current_distance, 4
            ),
            current_clearance_m=round(
                current_clearance, 4
            ),
            minimum_predicted_distance_m=round(
                minimum_distance, 4
            ),
            minimum_predicted_clearance_m=round(
                minimum_clearance, 4
            ),
            time_to_closest_approach_s=round(
                tca, 4
            ),
            time_to_collision_s=(
                round(ttc, 4)
                if ttc is not None
                else None
            ),
            relative_speed_m_s=round(
                relative_speed, 4
            ),
            stopping_distance_m=round(
                stopping_distance, 4
            ),
            collision_probability=round(
                collision_probability, 4
            ),
            risk_score=risk_score,
            risk_level=risk_level.value,
            collision_predicted=ttc is not None,
            human_proximity_risk=human_risk,
            recommended_action=(
                self.recommended_action(
                    risk_level,
                    human_risk,
                )
            ),
            simulation_only=True,
        )


# ============================================================
# Simulation Configuration
# ============================================================

@dataclass
class SimulationConfig:
    robot_count: int = 5
    obstacle_count: int = 10
    human_count: int = 3

    steps: int = 100
    delta_time_s: float = 0.1

    world_width_m: float = 30.0
    world_height_m: float = 30.0

    monte_carlo_samples: int = 100
    seed: int = 42

    safety: SafetyConfig = field(
        default_factory=SafetyConfig
    )

    def validate(self) -> None:
        if self.robot_count < 1:
            raise ValueError(
                "At least one robot is required"
            )

        if self.obstacle_count < 0 or self.human_count < 0:
            raise ValueError(
                "Entity counts cannot be negative"
            )

        if self.steps < 1:
            raise ValueError(
                "Simulation steps must be positive"
            )

        if self.delta_time_s <= 0:
            raise ValueError(
                "delta_time_s must be positive"
            )

        if (
            self.world_width_m <= 0
            or self.world_height_m <= 0
        ):
            raise ValueError(
                "World dimensions must be positive"
            )

        if self.monte_carlo_samples < 0:
            raise ValueError(
                "Monte Carlo samples cannot be negative"
            )

        self.safety.validate()


# ============================================================
# Simulation Engine
# ============================================================

class CollisionRiskSimulation:
    ROBOT_TYPES = [
        "mobile_robot",
        "robotic_arm",
        "humanoid",
        "quadruped",
    ]

    ROBOT_RADII = {
        "mobile_robot": 0.5,
        "robotic_arm": 0.7,
        "humanoid": 0.4,
        "quadruped": 0.45,
    }

    def __init__(
        self,
        config: SimulationConfig,
        entities: list[SimulationEntity] | None = None,
    ) -> None:
        self.config = config
        self.config.validate()

        self.rng = random.Random(
            config.seed
        )

        self.evaluator = CollisionRiskEvaluator(
            config.safety,
            monte_carlo_samples=(
                config.monte_carlo_samples
            ),
            seed=config.seed + 100,
        )

        self.entities = (
            entities
            if entities is not None
            else self.generate_entities()
        )

        self.results: list[CollisionRiskResult] = []

    def random_position(self) -> Vector2:
        return Vector2(
            self.rng.uniform(
                0,
                self.config.world_width_m,
            ),
            self.rng.uniform(
                0,
                self.config.world_height_m,
            ),
        )

    def random_velocity(
        self,
        max_speed: float,
    ) -> Vector2:
        angle = self.rng.uniform(
            0,
            2 * math.pi,
        )

        speed = self.rng.uniform(
            0,
            max_speed,
        )

        return Vector2(
            math.cos(angle) * speed,
            math.sin(angle) * speed,
        )

    def generate_entities(
        self,
    ) -> list[SimulationEntity]:
        entities = []

        for index in range(
            self.config.robot_count
        ):
            robot_type = self.ROBOT_TYPES[
                index % len(self.ROBOT_TYPES)
            ]

            velocity = self.random_velocity(1.2)

            # Simplified stationary footprint for arms.
            if robot_type == "robotic_arm":
                velocity = Vector2(0.0, 0.0)

            entities.append(
                SimulationEntity(
                    entity_id=f"ROBOT-{index + 1:03d}",
                    entity_type=EntityType.ROBOT,
                    position=self.random_position(),
                    velocity=velocity,
                    radius_m=self.ROBOT_RADII[robot_type],
                    mass_kg=60.0,
                    robot_type=robot_type,
                )
            )

        for index in range(
            self.config.obstacle_count
        ):
            entities.append(
                SimulationEntity(
                    entity_id=f"OBS-{index + 1:03d}",
                    entity_type=EntityType.OBSTACLE,
                    position=self.random_position(),
                    velocity=self.random_velocity(0.5),
                    radius_m=self.rng.uniform(0.2, 0.8),
                    mass_kg=30.0,
                )
            )

        for index in range(
            self.config.human_count
        ):
            entities.append(
                SimulationEntity(
                    entity_id=f"HUMAN-{index + 1:03d}",
                    entity_type=EntityType.HUMAN,
                    position=self.random_position(),
                    velocity=self.random_velocity(0.8),
                    radius_m=0.3,
                    mass_kg=70.0,
                )
            )

        return entities

    def update_entities(self) -> None:
        for entity in self.entities:
            if not entity.active:
                continue

            entity.update(
                self.config.delta_time_s
            )

            # Reflect simulated motion at world boundaries.
            if entity.position.x < entity.radius_m:
                entity.position.x = entity.radius_m
                entity.velocity.x = abs(
                    entity.velocity.x
                )

            elif entity.position.x > (
                self.config.world_width_m
                - entity.radius_m
            ):
                entity.position.x = (
                    self.config.world_width_m
                    - entity.radius_m
                )
                entity.velocity.x = -abs(
                    entity.velocity.x
                )

            if entity.position.y < entity.radius_m:
                entity.position.y = entity.radius_m
                entity.velocity.y = abs(
                    entity.velocity.y
                )

            elif entity.position.y > (
                self.config.world_height_m
                - entity.radius_m
            ):
                entity.position.y = (
                    self.config.world_height_m
                    - entity.radius_m
                )
                entity.velocity.y = -abs(
                    entity.velocity.y
                )

    def evaluate_step(
        self,
        timestamp_s: float,
    ) -> list[CollisionRiskResult]:
        robots = [
            entity
            for entity in self.entities
            if (
                entity.active
                and entity.entity_type == EntityType.ROBOT
            )
        ]

        targets = [
            entity
            for entity in self.entities
            if entity.active
        ]

        step_results = []

        for robot in robots:
            for target in targets:
                if target.entity_id == robot.entity_id:
                    continue

                # Evaluate each robot pair only once.
                if (
                    target.entity_type == EntityType.ROBOT
                    and target.entity_id < robot.entity_id
                ):
                    continue

                result = self.evaluator.evaluate(
                    robot,
                    target,
                    timestamp_s,
                )

                step_results.append(result)

        return step_results

    def run(self) -> dict[str, Any]:
        LOGGER.info(
            "Starting offline collision-risk simulation"
        )

        for step in range(self.config.steps):
            timestamp_s = (
                step * self.config.delta_time_s
            )

            step_results = self.evaluate_step(
                timestamp_s
            )

            self.results.extend(
                step_results
            )

            self.update_entities()

        return self.build_report()

    def build_report(self) -> dict[str, Any]:
        counts = {
            level.value: 0
            for level in RiskLevel
        }

        for result in self.results:
            counts[result.risk_level] += 1

        critical_events = [
            asdict(result)
            for result in self.results
            if result.risk_level == RiskLevel.CRITICAL.value
        ]

        high_events = [
            asdict(result)
            for result in self.results
            if result.risk_level == RiskLevel.HIGH.value
        ]

        max_risk = max(
            (
                result.risk_score
                for result in self.results
            ),
            default=0.0,
        )

        mean_risk = (
            sum(
                result.risk_score
                for result in self.results
            )
            / len(self.results)
            if self.results
            else 0.0
        )

        return {
            "status": "completed",
            "simulation_only": True,
            "safety_certified": False,
            "physical_control_executed": False,
            "configuration": {
                "robot_count": self.config.robot_count,
                "obstacle_count": self.config.obstacle_count,
                "human_count": self.config.human_count,
                "steps": self.config.steps,
                "delta_time_s": self.config.delta_time_s,
                "monte_carlo_samples": (
                    self.config.monte_carlo_samples
                ),
                "seed": self.config.seed,
            },
            "summary": {
                "total_evaluations": len(self.results),
                "risk_counts": counts,
                "critical_events": len(critical_events),
                "high_events": len(high_events),
                "maximum_risk_score": round(max_risk, 4),
                "mean_risk_score": round(mean_risk, 4),
                "predicted_collision_evaluations": sum(
                    result.collision_predicted
                    for result in self.results
                ),
                "human_proximity_evaluations": sum(
                    result.human_proximity_risk
                    for result in self.results
                ),
            },
            "critical_event_samples": critical_events[:25],
            "high_event_samples": high_events[:25],
            "limitations": [
                "Uses simplified 2D circular geometry.",
                "Assumes constant velocity within the prediction horizon.",
                "Does not model articulated robot geometry.",
                "Does not model braking dynamics during trajectory prediction.",
                "Monte Carlo estimates depend on assumed uncertainty distributions.",
                "Does not provide safety-rated collision avoidance.",
            ],
        }


# ============================================================
# Scenario Loading
# ============================================================

def load_scenario(
    path: str | Path,
) -> tuple[SimulationConfig, list[SimulationEntity] | None]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Scenario file not found: {path}"
        )

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for YAML scenarios"
            ) from exc

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = yaml.safe_load(handle)

    elif path.suffix.lower() == ".json":
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

    else:
        raise ValueError(
            "Scenario must be JSON or YAML"
        )

    if not isinstance(data, dict):
        raise ValueError(
            "Scenario root must be an object"
        )

    # Supports a dedicated collision simulation schema.
    # Other scenario formats may require an adapter.
    config_data = data.get(
        "simulation",
        {},
    )

    safety_data = data.get(
        "safety",
        {},
    )

    if not isinstance(config_data, dict):
        raise ValueError(
            "'simulation' must be an object"
        )

    if not isinstance(safety_data, dict):
        raise ValueError(
            "'safety' must be an object"
        )

    config_fields = set(
        SimulationConfig.__dataclass_fields__
    )

    safety_fields = set(
        SafetyConfig.__dataclass_fields__
    )

    config = SimulationConfig(
        **{
            key: value
            for key, value in config_data.items()
            if key in config_fields and key != "safety"
        },
        safety=SafetyConfig(
            **{
                key: value
                for key, value in safety_data.items()
                if key in safety_fields
            }
        ),
    )

    entities_data = data.get("entities")

    if entities_data is None:
        return config, None

    if not isinstance(entities_data, list):
        raise ValueError(
            "'entities' must be a list"
        )

    entities = []
    seen_ids = set()

    for item in entities_data:
        entity_id = str(item["id"])

        if entity_id in seen_ids:
            raise ValueError(
                f"Duplicate entity ID: {entity_id}"
            )

        seen_ids.add(entity_id)

        position = item["position"]
        velocity = item.get(
            "velocity",
            [0.0, 0.0],
        )

        entities.append(
            SimulationEntity(
                entity_id=entity_id,
                entity_type=EntityType(
                    item["type"]
                ),
                position=Vector2(
                    float(position[0]),
                    float(position[1]),
                ),
                velocity=Vector2(
                    float(velocity[0]),
                    float(velocity[1]),
                ),
                radius_m=float(
                    item.get("radius_m", 0.4)
                ),
                mass_kg=float(
                    item.get("mass_kg", 50.0)
                ),
                robot_type=item.get("robot_type"),
                active=bool(
                    item.get("active", True)
                ),
            )
        )

    return config, entities


# ============================================================
# Report Export
# ============================================================

def save_json(
    path: str | Path,
    report: dict[str, Any],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def save_csv(
    path: str | Path,
    results: list[CollisionRiskResult],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not results:
        return

    rows = [
        asdict(result)
        for result in results
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate robot collision risk offline"
        )
    )

    parser.add_argument(
        "--scenario",
        default=None,
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--obstacles",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--humans",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--csv",
        default=None,
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = build_parser().parse_args()

    if args.scenario:
        config, entities = load_scenario(
            args.scenario
        )
    else:
        config = SimulationConfig()
        entities = None

    overrides = {
        "robot_count": args.robots,
        "obstacle_count": args.obstacles,
        "human_count": args.humans,
        "steps": args.steps,
        "delta_time_s": args.dt,
        "monte_carlo_samples": args.monte_carlo,
        "seed": args.seed,
    }

    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    config.validate()

    if entities is not None:
        if not any(
            entity.entity_type == EntityType.ROBOT
            for entity in entities
        ):
            raise ValueError(
                "Scenario requires at least one robot"
            )

    simulation = CollisionRiskSimulation(
        config,
        entities=entities,
    )

    report = simulation.run()

    if args.output:
        save_json(
            args.output,
            report,
        )

    if args.csv:
        save_csv(
            args.csv,
            simulation.results,
        )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )

    LOGGER.info(
        "Collision-risk simulation completed"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        OSError,
        KeyError,
        TypeError,
    ) as exc:
        LOGGER.error(
            "Simulation failed: %s",
            exc,
        )

        sys.exit(1)
