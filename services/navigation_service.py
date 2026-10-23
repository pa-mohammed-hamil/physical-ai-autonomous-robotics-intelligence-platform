
"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/navigation_service.py

Robotics Navigation Intelligence Service

Python 3.11+
Standard library only.

Features:
    - Robot navigation registration
    - 2D occupancy grid maps
    - A* pathfinding
    - Static and dynamic obstacles
    - Robot-radius obstacle inflation
    - Route planning and validation
    - Waypoint tracking
    - Simulated robot movement
    - Navigation task lifecycle
    - Route replanning
    - Collision-aware grid navigation
    - Navigation performance metrics
    - JSON reporting

Safety:
    Simulation and planning only.
    Does not issue real robot commands.
    Does not replace certified navigation safety systems.
"""

from __future__ import annotations

import heapq
import json
import math
import threading
import uuid

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class NavigationError(Exception):
    pass


class NavigationValidationError(NavigationError):
    pass


class NavigationNotFoundError(NavigationError):
    pass


class NavigationStateError(NavigationError):
    pass


class NavigationPathNotFoundError(NavigationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class NavigationStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    READY = "ready"
    NAVIGATING = "navigating"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ObstacleType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise NavigationValidationError(
            f"{name} must be finite"
        )

    return float(value)


def euclidean_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    return math.hypot(
        x2 - x1,
        y2 - y1,
    )


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __post_init__(self) -> None:
        finite(self.x, "x")
        finite(self.y, "y")

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
        }


@dataclass
class NavigationObstacle:
    obstacle_id: str
    center: Point2D
    radius_m: float

    obstacle_type: ObstacleType = ObstacleType.STATIC

    def __post_init__(self) -> None:
        if not self.obstacle_id.strip():
            raise NavigationValidationError(
                "obstacle_id is required"
            )

        finite(
            self.radius_m,
            "radius_m",
        )

        if self.radius_m <= 0:
            raise NavigationValidationError(
                "Obstacle radius must be positive"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_id": self.obstacle_id,
            "center": self.center.to_dict(),
            "radius_m": self.radius_m,
            "obstacle_type": self.obstacle_type.value,
        }


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class NavigationServiceConfig:
    world_width_m: float = 30.0
    world_height_m: float = 30.0

    grid_resolution_m: float = 0.25

    default_robot_radius_m: float = 0.35

    obstacle_clearance_m: float = 0.1

    default_speed_mps: float = 0.7

    waypoint_tolerance_m: float = 0.2

    maximum_search_nodes: int = 250000

    maximum_navigation_steps: int = 100000

    allow_diagonal_movement: bool = True

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        positive = (
            "world_width_m",
            "world_height_m",
            "grid_resolution_m",
            "default_robot_radius_m",
            "default_speed_mps",
            "waypoint_tolerance_m",
        )

        for name in positive:
            value = finite(
                getattr(self, name),
                name,
            )

            if value <= 0:
                raise NavigationValidationError(
                    f"{name} must be positive"
                )

        if self.obstacle_clearance_m < 0:
            raise NavigationValidationError(
                "Obstacle clearance cannot be negative"
            )

        if self.maximum_search_nodes < 1:
            raise NavigationValidationError(
                "maximum_search_nodes must be positive"
            )

        if self.maximum_navigation_steps < 1:
            raise NavigationValidationError(
                "maximum_navigation_steps must be positive"
            )

        if self.allow_hardware_control:
            raise NavigationValidationError(
                "Hardware control is disabled"
            )


# ============================================================
# OCCUPANCY GRID
# ============================================================

class OccupancyGrid2D:

    def __init__(
        self,
        config: NavigationServiceConfig,
    ) -> None:
        self.config = config

        self.columns = math.ceil(
            config.world_width_m
            / config.grid_resolution_m
        )

        self.rows = math.ceil(
            config.world_height_m
            / config.grid_resolution_m
        )

        self.occupied: set[
            tuple[int, int]
        ] = set()

    def world_to_grid(
        self,
        point: Point2D,
    ) -> tuple[int, int]:
        if not (
            0 <= point.x < self.config.world_width_m
            and 0 <= point.y < self.config.world_height_m
        ):
            raise NavigationValidationError(
                "Point is outside navigation world"
            )

        return (
            int(
                point.x
                / self.config.grid_resolution_m
            ),
            int(
                point.y
                / self.config.grid_resolution_m
            ),
        )

    def grid_to_world(
        self,
        cell: tuple[int, int],
    ) -> Point2D:
        x, y = cell

        if not self.in_bounds(cell):
            raise NavigationValidationError(
                "Grid cell outside world"
            )

        return Point2D(
            x=(
                x + 0.5
            ) * self.config.grid_resolution_m,
            y=(
                y + 0.5
            ) * self.config.grid_resolution_m,
        )

    def in_bounds(
        self,
        cell: tuple[int, int],
    ) -> bool:
        x, y = cell

        return (
            0 <= x < self.columns
            and 0 <= y < self.rows
        )

    def is_free(
        self,
        cell: tuple[int, int],
    ) -> bool:
        return (
            self.in_bounds(cell)
            and cell not in self.occupied
        )

    def build(
        self,
        obstacles: Sequence[NavigationObstacle],
        *,
        robot_radius_m: float,
    ) -> None:
        self.occupied.clear()

        resolution = (
            self.config.grid_resolution_m
        )

        for obstacle in obstacles:
            inflated_radius = (
                obstacle.radius_m
                + robot_radius_m
                + self.config.obstacle_clearance_m
                + resolution * math.sqrt(2) / 2
            )

            min_x = max(
                0,
                int(
                    math.floor(
                        (obstacle.center.x - inflated_radius)
                        / resolution
                    )
                ),
            )

            max_x = min(
                self.columns - 1,
                int(
                    math.floor(
                        (obstacle.center.x + inflated_radius)
                        / resolution
                    )
                ),
            )

            min_y = max(
                0,
                int(
                    math.floor(
                        (obstacle.center.y - inflated_radius)
                        / resolution
                    )
                ),
            )

            max_y = min(
                self.rows - 1,
                int(
                    math.floor(
                        (obstacle.center.y + inflated_radius)
                        / resolution
                    )
                ),
            )

            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    point = self.grid_to_world(
                        (x, y)
                    )

                    if euclidean_distance(
                        point.x,
                        point.y,
                        obstacle.center.x,
                        obstacle.center.y,
                    ) <= inflated_radius:
                        self.occupied.add(
                            (x, y)
                        )

    def neighbors(
        self,
        cell: tuple[int, int],
    ) -> list[tuple[tuple[int, int], float]]:
        x, y = cell

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]

        if self.config.allow_diagonal_movement:
            directions += [
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]

        result = []

        for dx, dy in directions:
            neighbor = (
                x + dx,
                y + dy,
            )

            if not self.is_free(neighbor):
                continue

            if dx != 0 and dy != 0:
                if not (
                    self.is_free((x + dx, y))
                    and self.is_free((x, y + dy))
                ):
                    continue

            result.append(
                (
                    neighbor,
                    math.hypot(dx, dy)
                    * self.config.grid_resolution_m,
                )
            )

        return result


# ============================================================
# A* PATH PLANNER
# ============================================================

class AStarPathPlanner:

    def __init__(
        self,
        config: NavigationServiceConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def heuristic(
        a: tuple[int, int],
        b: tuple[int, int],
        resolution: float,
    ) -> float:
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        ) * resolution

    def plan(
        self,
        grid: OccupancyGrid2D,
        start: Point2D,
        goal: Point2D,
    ) -> list[Point2D]:
        start_cell = grid.world_to_grid(
            start
        )

        goal_cell = grid.world_to_grid(
            goal
        )

        if not grid.is_free(start_cell):
            raise NavigationPathNotFoundError(
                "Start cell is blocked"
            )

        if not grid.is_free(goal_cell):
            raise NavigationPathNotFoundError(
                "Goal cell is blocked"
            )

        frontier: list[
            tuple[
                float,
                int,
                tuple[int, int]
            ]
        ] = []

        sequence = 0

        heapq.heappush(
            frontier,
            (
                0.0,
                sequence,
                start_cell,
            ),
        )

        came_from: dict[
            tuple[int, int],
            tuple[int, int]
        ] = {}

        cost_so_far = {
            start_cell: 0.0
        }

        explored = 0

        while frontier:
            _, _, current = heapq.heappop(
                frontier
            )

            explored += 1

            if explored > self.config.maximum_search_nodes:
                raise NavigationPathNotFoundError(
                    "A* search node limit exceeded"
                )

            if current == goal_cell:
                path_cells = [current]

                while current in came_from:
                    current = came_from[current]
                    path_cells.append(current)

                path_cells.reverse()

                return [
                    start,
                    *[
                        grid.grid_to_world(cell)
                        for cell in path_cells[1:-1]
                    ],
                    goal,
                ]

            for neighbor, step_cost in grid.neighbors(
                current
            ):
                new_cost = (
                    cost_so_far[current]
                    + step_cost
                )

                if (
                    neighbor not in cost_so_far
                    or new_cost < cost_so_far[neighbor]
                ):
                    cost_so_far[neighbor] = new_cost

                    sequence += 1

                    priority = (
                        new_cost
                        + self.heuristic(
                            neighbor,
                            goal_cell,
                            grid.config.grid_resolution_m,
                        )
                    )

                    heapq.heappush(
                        frontier,
                        (
                            priority,
                            sequence,
                            neighbor,
                        ),
                    )

                    came_from[
                        neighbor
                    ] = current

        raise NavigationPathNotFoundError(
            "No path found"
        )


# ============================================================
# ROBOT NAVIGATION STATE
# ============================================================

@dataclass
class NavigationRobot:
    robot_id: str

    position: Point2D

    radius_m: float

    speed_mps: float

    heading_rad: float = 0.0

    active_task_id: str | None = None

    distance_travelled_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "position": self.position.to_dict(),
            "radius_m": self.radius_m,
            "speed_mps": self.speed_mps,
            "heading_rad": self.heading_rad,
            "active_task_id": self.active_task_id,
            "distance_travelled_m": (
                self.distance_travelled_m
            ),
        }


# ============================================================
# NAVIGATION TASK
# ============================================================

@dataclass
class NavigationTask:
    task_id: str
    robot_id: str

    start: Point2D
    goal: Point2D

    status: NavigationStatus = (
        NavigationStatus.IDLE
    )

    path: list[Point2D] = field(
        default_factory=list
    )

    waypoint_index: int = 0

    replanning_count: int = 0

    navigation_steps: int = 0

    created_at: str = field(
        default_factory=utc_now
    )

    completed_at: str | None = None

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "start": self.start.to_dict(),
            "goal": self.goal.to_dict(),
            "status": self.status.value,
            "path": [
                point.to_dict()
                for point in self.path
            ],
            "waypoint_index": self.waypoint_index,
            "replanning_count": (
                self.replanning_count
            ),
            "navigation_steps": (
                self.navigation_steps
            ),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ============================================================
# NAVIGATION SERVICE
# ============================================================

class RoboticsNavigationService:

    def __init__(
        self,
        config: NavigationServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or NavigationServiceConfig()
        )

        self.robots: dict[
            str,
            NavigationRobot
        ] = {}

        self.obstacles: dict[
            str,
            NavigationObstacle
        ] = {}

        self.tasks: dict[
            str,
            NavigationTask
        ] = {}

        self.planner = AStarPathPlanner(
            self.config
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # REGISTER ROBOT
    # --------------------------------------------------------

    def register_robot(
        self,
        *,
        robot_id: str,
        position: Point2D,
        radius_m: float | None = None,
        speed_mps: float | None = None,
    ) -> NavigationRobot:
        with self._lock:
            if not robot_id.strip():
                raise NavigationValidationError(
                    "robot_id is required"
                )

            if robot_id in self.robots:
                raise NavigationStateError(
                    "Robot already registered"
                )

            robot = NavigationRobot(
                robot_id=robot_id,
                position=position,
                radius_m=(
                    radius_m
                    if radius_m is not None
                    else self.config.default_robot_radius_m
                ),
                speed_mps=(
                    speed_mps
                    if speed_mps is not None
                    else self.config.default_speed_mps
                ),
            )

            if (
                robot.radius_m <= 0
                or robot.speed_mps <= 0
            ):
                raise NavigationValidationError(
                    "Robot radius and speed must be positive"
                )

            self.robots[
                robot_id
            ] = robot

            return robot

    # --------------------------------------------------------
    # REGISTER OBSTACLE
    # --------------------------------------------------------

    def add_obstacle(
        self,
        obstacle: NavigationObstacle,
    ) -> None:
        with self._lock:
            self.obstacles[
                obstacle.obstacle_id
            ] = obstacle

    def remove_obstacle(
        self,
        obstacle_id: str,
    ) -> None:
        with self._lock:
            self.obstacles.pop(
                obstacle_id,
                None,
            )

    # --------------------------------------------------------
    # GET ROBOT
    # --------------------------------------------------------

    def get_robot(
        self,
        robot_id: str,
    ) -> NavigationRobot:
        robot = self.robots.get(
            robot_id
        )

        if robot is None:
            raise NavigationNotFoundError(
                f"Robot not found: {robot_id}"
            )

        return robot

    # --------------------------------------------------------
    # CREATE TASK
    # --------------------------------------------------------

    def create_task(
        self,
        *,
        robot_id: str,
        goal: Point2D,
    ) -> NavigationTask:
        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            if robot.active_task_id is not None:
                raise NavigationStateError(
                    "Robot already has an active navigation task"
                )

            task = NavigationTask(
                task_id=str(uuid.uuid4()),
                robot_id=robot_id,
                start=robot.position,
                goal=goal,
            )

            self.tasks[
                task.task_id
            ] = task

            robot.active_task_id = task.task_id

            return task

    # --------------------------------------------------------
    # GET TASK
    # --------------------------------------------------------

    def get_task(
        self,
        task_id: str,
    ) -> NavigationTask:
        task = self.tasks.get(
            task_id
        )

        if task is None:
            raise NavigationNotFoundError(
                f"Task not found: {task_id}"
            )

        return task

    # --------------------------------------------------------
    # BUILD GRID
    # --------------------------------------------------------

    def _build_grid(
        self,
        robot: NavigationRobot,
    ) -> OccupancyGrid2D:
        grid = OccupancyGrid2D(
            self.config
        )

        grid.build(
            list(
                self.obstacles.values()
            ),
            robot_radius_m=robot.radius_m,
        )

        return grid

    # --------------------------------------------------------
    # PLAN ROUTE
    # --------------------------------------------------------

    def plan_task(
        self,
        task_id: str,
    ) -> NavigationTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status not in (
                NavigationStatus.IDLE,
                NavigationStatus.READY,
            ):
                raise NavigationStateError(
                    "Task cannot be planned in current state"
                )

            robot = self.get_robot(
                task.robot_id
            )

            task.status = NavigationStatus.PLANNING

            try:
                grid = self._build_grid(
                    robot
                )

                task.path = self.planner.plan(
                    grid,
                    robot.position,
                    task.goal,
                )

                task.waypoint_index = (
                    1 if len(task.path) > 1 else 0
                )

                task.status = NavigationStatus.READY

            except NavigationError as exc:
                task.status = NavigationStatus.FAILED
                task.error = str(exc)
                task.completed_at = utc_now()
                robot.active_task_id = None

            return task

    # --------------------------------------------------------
    # START NAVIGATION
    # --------------------------------------------------------

    def start_task(
        self,
        task_id: str,
    ) -> NavigationTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status != NavigationStatus.READY:
                raise NavigationStateError(
                    "Task must be planned first"
                )

            task.status = NavigationStatus.NAVIGATING

            return task

    # --------------------------------------------------------
    # PAUSE / RESUME
    # --------------------------------------------------------

    def pause_task(
        self,
        task_id: str,
    ) -> NavigationTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status != NavigationStatus.NAVIGATING:
                raise NavigationStateError(
                    "Task is not navigating"
                )

            task.status = NavigationStatus.PAUSED

            return task

    def resume_task(
        self,
        task_id: str,
    ) -> NavigationTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status != NavigationStatus.PAUSED:
                raise NavigationStateError(
                    "Task is not paused"
                )

            task.status = NavigationStatus.NAVIGATING

            return task

    # --------------------------------------------------------
    # REPLAN
    # --------------------------------------------------------

    def replan_task(
        self,
        task_id: str,
    ) -> NavigationTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status not in (
                NavigationStatus.READY,
                NavigationStatus.NAVIGATING,
                NavigationStatus.PAUSED,
            ):
                raise NavigationStateError(
                    "Task cannot be replanned"
                )

            robot = self.get_robot(
                task.robot_id
            )

            previous_status = task.status

            try:
                grid = self._build_grid(
                    robot
                )

                new_path = self.planner.plan(
                    grid,
                    robot.position,
                    task.goal,
                )

                task.path = new_path
                task.waypoint_index = (
                    1 if len(new_path) > 1 else 0
                )

                task.replanning_count += 1

                task.status = previous_status

                task.error = None

            except NavigationError as exc:
                task.status = NavigationStatus.FAILED
                task.error = str(exc)
                task.completed_at = utc_now()
                robot.active_task_id = None

            return task

    # --------------------------------------------------------
    # SIMULATION STEP
    # --------------------------------------------------------

    def step(
        self,
        task_id: str,
        *,
        dt_seconds: float = 0.1,
    ) -> NavigationTask:
        finite(
            dt_seconds,
            "dt_seconds",
        )

        if dt_seconds <= 0:
            raise NavigationValidationError(
                "dt_seconds must be positive"
            )

        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status != NavigationStatus.NAVIGATING:
                raise NavigationStateError(
                    "Task is not navigating"
                )

            robot = self.get_robot(
                task.robot_id
            )

            if (
                task.navigation_steps
                >= self.config.maximum_navigation_steps
            ):
                task.status = NavigationStatus.FAILED
                task.error = "Navigation step limit exceeded"
                task.completed_at = utc_now()
                robot.active_task_id = None
                return task

            grid = self._build_grid(
                robot
            )

            if not grid.is_free(
                grid.world_to_grid(robot.position)
            ):
                task.status = NavigationStatus.FAILED
                task.error = "Robot is inside an occupied cell"
                task.completed_at = utc_now()
                robot.active_task_id = None
                return task

            if task.waypoint_index >= len(task.path):
                self._complete_task(
                    task,
                    robot,
                )
                return task

            waypoint = task.path[
                task.waypoint_index
            ]

            try:
                if not grid.is_free(
                    grid.world_to_grid(waypoint)
                ):
                    task = self.replan_task(
                        task_id
                    )

                    if task.status == NavigationStatus.FAILED:
                        return task

                    waypoint = task.path[
                        task.waypoint_index
                    ]

            except NavigationError as exc:
                task.status = NavigationStatus.FAILED
                task.error = str(exc)
                task.completed_at = utc_now()
                robot.active_task_id = None
                return task

            dx = waypoint.x - robot.position.x
            dy = waypoint.y - robot.position.y

            remaining = math.hypot(
                dx,
                dy,
            )

            if remaining <= self.config.waypoint_tolerance_m:
                task.waypoint_index += 1
                task.navigation_steps += 1

                if task.waypoint_index >= len(task.path):
                    self._complete_task(
                        task,
                        robot,
                    )

                return task

            movement = min(
                robot.speed_mps * dt_seconds,
                remaining,
            )

            next_position = Point2D(
                x=(
                    robot.position.x
                    + dx / remaining * movement
                ),
                y=(
                    robot.position.y
                    + dy / remaining * movement
                ),
            )

            try:
                if not grid.is_free(
                    grid.world_to_grid(next_position)
                ):
                    return self.replan_task(
                        task_id
                    )

            except NavigationValidationError:
                task.status = NavigationStatus.FAILED
                task.error = "Movement outside navigation world"
                task.completed_at = utc_now()
                robot.active_task_id = None
                return task

            robot.heading_rad = math.atan2(
                dy,
                dx,
            )

            robot.position = next_position

            robot.distance_travelled_m += movement

            task.navigation_steps += 1

            if euclidean_distance(
                robot.position.x,
                robot.position.y,
                task.goal.x,
                task.goal.y,
            ) <= self.config.waypoint_tolerance_m:
                self._complete_task(
                    task,
                    robot,
                )

            return task

    # --------------------------------------------------------
    # COMPLETE TASK
    # --------------------------------------------------------

    def _complete_task(
        self,
        task: NavigationTask,
        robot: NavigationRobot,
    ) -> None:
        if euclidean_distance(
            robot.position.x,
            robot.position.y,
            task.goal.x,
            task.goal.y,
        ) > self.config.waypoint_tolerance_m:
            task.status = NavigationStatus.FAILED
            task.error = "Goal tolerance not satisfied"
        else:
            task.status = NavigationStatus.COMPLETED
            task.error = None

        task.completed_at = utc_now()
        robot.active_task_id = None

    # --------------------------------------------------------
    # RUN SIMULATION
    # --------------------------------------------------------

    def run_simulation(
        self,
        task_id: str,
        *,
        dt_seconds: float = 0.1,
    ) -> NavigationTask:
        task = self.get_task(
            task_id
        )

        if task.status == NavigationStatus.IDLE:
            task = self.plan_task(
                task_id
            )

        if task.status == NavigationStatus.READY:
            task = self.start_task(
                task_id
            )

        for _ in range(
            self.config.maximum_navigation_steps
        ):
            if task.status != NavigationStatus.NAVIGATING:
                break

            task = self.step(
                task_id,
                dt_seconds=dt_seconds,
            )

        return task

    # --------------------------------------------------------
    # CANCEL TASK
    # --------------------------------------------------------

    def cancel_task(
        self,
        task_id: str,
    ) -> NavigationTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status in (
                NavigationStatus.COMPLETED,
                NavigationStatus.FAILED,
                NavigationStatus.CANCELLED,
            ):
                raise NavigationStateError(
                    "Task is already terminal"
                )

            task.status = NavigationStatus.CANCELLED
            task.completed_at = utc_now()

            robot = self.get_robot(
                task.robot_id
            )

            robot.active_task_id = None

            return task

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        tasks = list(
            self.tasks.values()
        )

        completed = sum(
            task.status == NavigationStatus.COMPLETED
            for task in tasks
        )

        failed = sum(
            task.status == NavigationStatus.FAILED
            for task in tasks
        )

        return {
            "registered_robots": len(self.robots),
            "registered_obstacles": len(self.obstacles),
            "total_tasks": len(tasks),
            "completed_tasks": completed,
            "failed_tasks": failed,
            "success_rate": (
                completed / (completed + failed)
                if completed + failed
                else None
            ),
            "total_replanning_events": sum(
                task.replanning_count
                for task in tasks
            ),
            "total_navigation_steps": sum(
                task.navigation_steps
                for task in tasks
            ),
            "total_distance_travelled_m": sum(
                robot.distance_travelled_m
                for robot in self.robots.values()
            ),
            "status_distribution": dict(
                Counter(
                    task.status.value
                    for task in tasks
                )
            ),
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "generated_at": utc_now(),
            "statistics": self.statistics(),
            "robots": [
                robot.to_dict()
                for robot in self.robots.values()
            ],
            "obstacles": [
                obstacle.to_dict()
                for obstacle in self.obstacles.values()
            ],
            "tasks": [
                task.to_dict()
                for task in self.tasks.values()
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


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsNavigationService(
        NavigationServiceConfig(
            world_width_m=20.0,
            world_height_m=20.0,
            grid_resolution_m=0.25,
        )
    )

    service.register_robot(
        robot_id="robot_001",
        position=Point2D(1.0, 1.0),
        speed_mps=1.0,
    )

    service.add_obstacle(
        NavigationObstacle(
            obstacle_id="obstacle_001",
            center=Point2D(5.0, 5.0),
            radius_m=1.2,
        )
    )

    service.add_obstacle(
        NavigationObstacle(
            obstacle_id="obstacle_002",
            center=Point2D(8.0, 8.0),
            radius_m=1.0,
        )
    )

    task = service.create_task(
        robot_id="robot_001",
        goal=Point2D(15.0, 15.0),
    )

    result = service.run_simulation(
        task.task_id,
        dt_seconds=0.1,
    )

    print(
        json.dumps(
            {
                "task": result.to_dict(),
                "statistics": service.statistics(),
            },
            indent=2,
        )
    )

    service.export_json(
        "data/navigation/navigation_report.json"
    )


if __name__ == "__main__":
    main()
