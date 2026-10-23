"""Time Synchronization module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/time_synchronization.py

Python 3.11+
Standard library only.

TIME SYNCHRONIZATION MANAGER — SIMULATION ONLY

Features:
    - Multi-tenant robot clock registry
    - Simulated remote-clock offset estimation
    - Round-trip-time (RTT) measurement
    - Clock drift estimation
    - Clock quality classification
    - Remote timestamp correction
    - Sensor timestamp alignment
    - Out-of-order timestamp detection
    - Bounded synchronization history
    - Audit events
    - JSON report export

Important:
    - No system clock is modified.
    - No real NTP/PTP synchronization occurs.
    - No physical robot is controlled.
    - Estimates are not safety-rated.
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class TimeSynchronizationError(Exception):
    pass


class TimeSynchronizationValidationError(
    TimeSynchronizationError
):
    pass


class ClockNotFound(TimeSynchronizationError):
    pass


class ClockConflict(TimeSynchronizationError):
    pass


class ClockNotSynchronized(TimeSynchronizationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ClockSource(str, Enum):
    SIMULATION = "simulation"
    SYSTEM = "system"
    ROS2 = "ros2"
    NTP = "ntp"
    PTP = "ptp"
    GPS = "gps"
    CUSTOM = "custom"


class ClockStatus(str, Enum):
    REGISTERED = "registered"
    SYNCHRONIZED = "synchronized"
    DEGRADED = "degraded"
    STALE = "stale"
    UNSYNCHRONIZED = "unsynchronized"


class ClockEventType(str, Enum):
    REGISTERED = "registered"
    SAMPLE_ACCEPTED = "sample_accepted"
    SAMPLE_REJECTED = "sample_rejected"
    STATUS_CHANGED = "status_changed"
    CLOCK_RESET = "clock_reset"
    FRAME_ALIGNED = "frame_aligned"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise TimeSynchronizationValidationError(
            f"Invalid {name}"
        )

    return value


def finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise TimeSynchronizationValidationError(
            f"{name} must be finite"
        )

    return float(value)


def positive(value: Any, name: str) -> float:
    result = finite(value, name)

    if result <= 0:
        raise TimeSynchronizationValidationError(
            f"{name} must be positive"
        )

    return result


def timestamp_utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise TimeSynchronizationValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


def milliseconds_between(
    later: datetime,
    earlier: datetime,
) -> float:
    return (
        timestamp_utc(later)
        - timestamp_utc(earlier)
    ).total_seconds() * 1000.0


def json_safe(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
        )
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TimeSynchronizationConfig:
    maximum_clocks: int = 10000
    maximum_samples_per_clock: int = 200
    maximum_audit_events: int = 20000

    maximum_rtt_ms: float = 200.0
    maximum_offset_ms: float = 1000.0

    good_offset_ms: float = 10.0
    good_rtt_ms: float = 50.0

    maximum_drift_ppm: float = 100.0

    synchronization_timeout_seconds: float = 10.0
    alignment_tolerance_ms: float = 25.0

    minimum_samples_for_sync: int = 3

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_clocks",
            "maximum_samples_per_clock",
            "maximum_audit_events",
            "minimum_samples_for_sync",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise TimeSynchronizationValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "maximum_rtt_ms",
            "maximum_offset_ms",
            "good_offset_ms",
            "good_rtt_ms",
            "maximum_drift_ppm",
            "synchronization_timeout_seconds",
            "alignment_tolerance_ms",
        ):
            positive(
                getattr(self, name),
                name,
            )

        if (
            self.good_offset_ms
            > self.maximum_offset_ms
            or self.good_rtt_ms
            > self.maximum_rtt_ms
        ):
            raise TimeSynchronizationValidationError(
                "Invalid clock quality thresholds"
            )

        if (
            self.minimum_samples_for_sync
            > self.maximum_samples_per_clock
        ):
            raise TimeSynchronizationValidationError(
                "Minimum samples exceed history capacity"
            )

        if not self.simulation_only:
            raise TimeSynchronizationValidationError(
                "Real clock synchronization unsupported"
            )


# ============================================================
# CLOCK DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotClockDefinition:
    tenant_id: str
    robot_id: str
    clock_id: str

    source: ClockSource = ClockSource.SIMULATION

    description: str = ""

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "clock_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not isinstance(
            self.source,
            ClockSource,
        ):
            raise TimeSynchronizationValidationError(
                "Invalid clock source"
            )

        if (
            not isinstance(self.description, str)
            or len(self.description) > 512
        ):
            raise TimeSynchronizationValidationError(
                "Invalid clock description"
            )


# ============================================================
# CLOCK EXCHANGE SAMPLE
# ============================================================

@dataclass(frozen=True)
class ClockExchangeSample:
    """
    Four-timestamp exchange:

        t1: Local request sent
        t2: Remote request received
        t3: Remote response sent
        t4: Local response received

    Offset convention:

        offset = remote_clock - local_clock

    Correct remote timestamp:

        local_time = remote_time - offset

    The formula assumes reasonably symmetric
    communication delay.
    """

    tenant_id: str
    robot_id: str
    clock_id: str

    sequence_number: int

    t1: datetime
    t2: datetime
    t3: datetime
    t4: datetime

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "clock_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise TimeSynchronizationValidationError(
                "Invalid sequence number"
            )

        for name in (
            "t1",
            "t2",
            "t3",
            "t4",
        ):
            timestamp_utc(
                getattr(self, name)
            )

        if self.t4 < self.t1:
            raise TimeSynchronizationValidationError(
                "Local receive precedes local send"
            )

        if self.t3 < self.t2:
            raise TimeSynchronizationValidationError(
                "Remote send precedes remote receive"
            )

    def estimate(self) -> tuple[float, float]:
        offset_ms = (
            milliseconds_between(
                self.t2,
                self.t1,
            )
            + milliseconds_between(
                self.t3,
                self.t4,
            )
        ) / 2.0

        rtt_ms = (
            milliseconds_between(
                self.t4,
                self.t1,
            )
            - milliseconds_between(
                self.t3,
                self.t2,
            )
        )

        if rtt_ms < -1e-6:
            raise TimeSynchronizationValidationError(
                "Negative network round-trip delay"
            )

        return offset_ms, max(0.0, rtt_ms)


# ============================================================
# ESTIMATE
# ============================================================

@dataclass(frozen=True)
class ClockEstimate:
    sequence_number: int

    offset_ms: float
    rtt_ms: float

    drift_ppm: float

    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_number": self.sequence_number,
            "offset_ms": self.offset_ms,
            "rtt_ms": self.rtt_ms,
            "drift_ppm": self.drift_ppm,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================
# SENSOR TIMESTAMP
# ============================================================

@dataclass(frozen=True)
class SensorTimestamp:
    tenant_id: str
    robot_id: str
    clock_id: str
    sensor_id: str

    sequence_number: int
    timestamp: datetime

    payload_reference: str = ""

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "clock_id",
            "sensor_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise TimeSynchronizationValidationError(
                "Invalid sensor sequence"
            )

        timestamp_utc(
            self.timestamp
        )

        if (
            not isinstance(self.payload_reference, str)
            or len(self.payload_reference) > 1024
        ):
            raise TimeSynchronizationValidationError(
                "Invalid payload reference"
            )


# ============================================================
# CLOCK RECORD
# ============================================================

@dataclass
class ClockRecord:
    definition: RobotClockDefinition

    status: ClockStatus = ClockStatus.REGISTERED

    last_sequence_number: int = -1

    offset_ms: float | None = None
    rtt_ms: float | None = None
    drift_ppm: float = 0.0

    last_sample_monotonic: float | None = None

    samples: deque[ClockEstimate] = field(
        default_factory=deque
    )

    sensor_sequences: dict[str, int] = field(
        default_factory=dict
    )

    sensor_timestamps: dict[
        str,
        datetime,
    ] = field(default_factory=dict)

    registered_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        d = self.definition

        return {
            "tenant_id": d.tenant_id,
            "robot_id": d.robot_id,
            "clock_id": d.clock_id,
            "source": d.source.value,
            "description": d.description,
            "status": self.status.value,
            "offset_ms": self.offset_ms,
            "rtt_ms": self.rtt_ms,
            "drift_ppm": self.drift_ppm,
            "last_sequence_number": (
                self.last_sequence_number
            ),
            "sample_count": len(self.samples),
            "registered_at": (
                self.registered_at.isoformat()
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class ClockAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    clock_id: str

    event_type: ClockEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "clock_id": self.clock_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# TIME SYNCHRONIZATION MANAGER
# ============================================================

class RoboticsTimeSynchronizationManager:

    """
    Thread-safe simulated clock synchronization.

    Flow:
        register_clock()
            |
        ingest_exchange()
            |
        evaluate_clock()
            |
        correct_timestamp()
            |
        align_sensor_frames()

    Does not modify operating-system clocks.
    """

    def __init__(
        self,
        config: TimeSynchronizationConfig | None = None,
    ) -> None:
        self.config = (
            config or TimeSynchronizationConfig()
        )

        self._clocks: dict[
            tuple[str, str, str],
            ClockRecord,
        ] = {}

        self._audit_events: deque[
            ClockAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        clock_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(clock_id, "clock_id"),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        clock_id: str,
    ) -> ClockRecord:
        record = self._clocks.get(
            self._key(
                tenant_id,
                robot_id,
                clock_id,
            )
        )

        if record is None:
            raise ClockNotFound(
                "Robot clock not registered"
            )

        return record

    def _audit(
        self,
        record: ClockRecord,
        event_type: ClockEventType,
        details: str,
    ) -> None:
        d = record.definition

        self._audit_events.append(
            ClockAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=d.tenant_id,
                robot_id=d.robot_id,
                clock_id=d.clock_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # REGISTER CLOCK
    # --------------------------------------------------------

    def register_clock(
        self,
        definition: RobotClockDefinition,
    ) -> dict[str, Any]:
        if not isinstance(
            definition,
            RobotClockDefinition,
        ):
            raise TimeSynchronizationValidationError(
                "Invalid clock definition"
            )

        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.clock_id,
            )

            if key in self._clocks:
                raise ClockConflict(
                    "Clock already registered"
                )

            if (
                len(self._clocks)
                >= self.config.maximum_clocks
            ):
                raise TimeSynchronizationError(
                    "Clock capacity reached"
                )

            record = ClockRecord(
                definition=definition,
                samples=deque(
                    maxlen=(
                        self.config.maximum_samples_per_clock
                    )
                ),
            )

            self._clocks[key] = record

            self._audit(
                record,
                ClockEventType.REGISTERED,
                "Clock registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # DRIFT ESTIMATION
    # --------------------------------------------------------

    @staticmethod
    def _estimate_drift(
        samples: deque[ClockEstimate],
    ) -> float:
        if len(samples) < 2:
            return 0.0

        first = samples[0]
        last = samples[-1]

        elapsed_seconds = (
            last.timestamp - first.timestamp
        ).total_seconds()

        if elapsed_seconds <= 0:
            return 0.0

        offset_change_ms = (
            last.offset_ms - first.offset_ms
        )

        # 1 ms/s = 1000 ppm.
        return (
            offset_change_ms
            / elapsed_seconds
        ) * 1000.0

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def _evaluate(
        self,
        record: ClockRecord,
    ) -> None:
        previous = record.status

        if not record.samples:
            current = ClockStatus.UNSYNCHRONIZED

        elif (
            record.last_sample_monotonic is None
            or time.monotonic()
            - record.last_sample_monotonic
            > self.config.synchronization_timeout_seconds
        ):
            current = ClockStatus.STALE

        elif (
            len(record.samples)
            < self.config.minimum_samples_for_sync
        ):
            current = ClockStatus.UNSYNCHRONIZED

        elif (
            record.offset_ms is None
            or record.rtt_ms is None
        ):
            current = ClockStatus.UNSYNCHRONIZED

        elif (
            abs(record.offset_ms)
            > self.config.maximum_offset_ms
            or record.rtt_ms
            > self.config.maximum_rtt_ms
            or abs(record.drift_ppm)
            > self.config.maximum_drift_ppm
        ):
            current = ClockStatus.DEGRADED

        elif (
            abs(record.offset_ms)
            <= self.config.good_offset_ms
            and record.rtt_ms
            <= self.config.good_rtt_ms
        ):
            current = ClockStatus.SYNCHRONIZED

        else:
            current = ClockStatus.DEGRADED

        record.status = current

        if previous != current:
            self._audit(
                record,
                ClockEventType.STATUS_CHANGED,
                (
                    f"{previous.value} -> "
                    f"{current.value}"
                ),
            )

    # --------------------------------------------------------
    # INGEST CLOCK EXCHANGE
    # --------------------------------------------------------

    def ingest_exchange(
        self,
        sample: ClockExchangeSample,
    ) -> dict[str, Any]:
        if not isinstance(
            sample,
            ClockExchangeSample,
        ):
            raise TimeSynchronizationValidationError(
                "Invalid clock exchange"
            )

        offset_ms, rtt_ms = sample.estimate()

        if (
            rtt_ms > self.config.maximum_rtt_ms
        ):
            raise TimeSynchronizationValidationError(
                "Round-trip delay exceeds limit"
            )

        if (
            abs(offset_ms)
            > self.config.maximum_offset_ms
        ):
            raise TimeSynchronizationValidationError(
                "Clock offset exceeds limit"
            )

        with self._lock:
            record = self._get(
                sample.tenant_id,
                sample.robot_id,
                sample.clock_id,
            )

            if (
                sample.sequence_number
                <= record.last_sequence_number
            ):
                raise TimeSynchronizationValidationError(
                    "Out-of-order clock exchange"
                )

            # A local monotonic timestamp is used to
            # estimate offset drift independently of
            # the remote clock's wall-time offset.
            received_monotonic = time.monotonic()

            # Store UTC receipt timestamp for reporting.
            received_utc = utc_now()

            estimate = ClockEstimate(
                sequence_number=sample.sequence_number,
                offset_ms=offset_ms,
                rtt_ms=rtt_ms,
                drift_ppm=0.0,
                timestamp=received_utc,
            )

            record.samples.append(
                estimate
            )

            record.last_sequence_number = (
                sample.sequence_number
            )

            record.last_sample_monotonic = (
                received_monotonic
            )

            # Select the lowest-RTT estimate from
            # the recent history to reduce effects
            # of asymmetric communication delays.
            best = min(
                record.samples,
                key=lambda item: item.rtt_ms,
            )

            record.offset_ms = best.offset_ms
            record.rtt_ms = best.rtt_ms

            record.drift_ppm = (
                self._estimate_drift(
                    record.samples
                )
            )

            self._evaluate(record)

            self._audit(
                record,
                ClockEventType.SAMPLE_ACCEPTED,
                (
                    f"offset={offset_ms:.3f}ms, "
                    f"rtt={rtt_ms:.3f}ms"
                ),
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CORRECT TIMESTAMP
    # --------------------------------------------------------

    def correct_timestamp(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        clock_id: str,
        remote_timestamp: datetime,
    ) -> datetime:
        remote = timestamp_utc(
            remote_timestamp
        )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                clock_id,
            )

            self._evaluate(record)

            if record.status != ClockStatus.SYNCHRONIZED:
                raise ClockNotSynchronized(
                    "Clock quality insufficient"
                )

            if record.offset_ms is None:
                raise ClockNotSynchronized(
                    "Clock offset unavailable"
                )

            # remote = local + offset
            return remote - timedelta(
                milliseconds=record.offset_ms
            )

    # --------------------------------------------------------
    # ALIGN SENSOR FRAMES
    # --------------------------------------------------------

    def align_sensor_frames(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        frames: list[SensorTimestamp],
        tolerance_ms: float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(frames, list) or not frames:
            raise TimeSynchronizationValidationError(
                "At least one sensor frame required"
            )

        tolerance = (
            self.config.alignment_tolerance_ms
            if tolerance_ms is None
            else positive(
                tolerance_ms,
                "tolerance_ms",
            )
        )

        corrected = []

        with self._lock:
            # Validate the complete batch before
            # changing sensor sequence records.
            pending: dict[
                tuple[str, str, str, str],
                tuple[int, datetime],
            ] = {}

            for frame in frames:
                if not isinstance(
                    frame,
                    SensorTimestamp,
                ):
                    raise TimeSynchronizationValidationError(
                        "Invalid sensor frame"
                    )

                if (
                    frame.tenant_id != tenant_id
                    or frame.robot_id != robot_id
                ):
                    raise TimeSynchronizationValidationError(
                        "Sensor frame identity mismatch"
                    )

                record = self._get(
                    tenant_id,
                    robot_id,
                    frame.clock_id,
                )

                key = (
                    tenant_id,
                    robot_id,
                    frame.clock_id,
                    frame.sensor_id,
                )

                previous_sequence = (
                    record.sensor_sequences.get(
                        frame.sensor_id,
                        -1,
                    )
                )

                previous_timestamp = (
                    record.sensor_timestamps.get(
                        frame.sensor_id
                    )
                )

                if key in pending:
                    previous_sequence, previous_timestamp = (
                        pending[key]
                    )

                if (
                    frame.sequence_number
                    <= previous_sequence
                ):
                    raise TimeSynchronizationValidationError(
                        "Out-of-order sensor sequence"
                    )

                local_timestamp = (
                    self.correct_timestamp(
                        tenant_id=tenant_id,
                        robot_id=robot_id,
                        clock_id=frame.clock_id,
                        remote_timestamp=frame.timestamp,
                    )
                )

                if (
                    previous_timestamp is not None
                    and local_timestamp
                    <= previous_timestamp
                ):
                    raise TimeSynchronizationValidationError(
                        "Out-of-order corrected timestamp"
                    )

                pending[key] = (
                    frame.sequence_number,
                    local_timestamp,
                )

                corrected.append(
                    {
                        "sensor_id": frame.sensor_id,
                        "clock_id": frame.clock_id,
                        "sequence_number": (
                            frame.sequence_number
                        ),
                        "original_timestamp": (
                            frame.timestamp.isoformat()
                        ),
                        "corrected_timestamp": (
                            local_timestamp.isoformat()
                        ),
                        "payload_reference": (
                            frame.payload_reference
                        ),
                        "_timestamp": local_timestamp,
                    }
                )

            timestamps = [
                item["_timestamp"]
                for item in corrected
            ]

            spread_ms = (
                max(timestamps) - min(timestamps)
            ).total_seconds() * 1000.0

            aligned = spread_ms <= tolerance

            # Commit sequence tracking only if
            # all frames pass validation and alignment.
            if aligned:
                for key, (
                    sequence,
                    corrected_timestamp,
                ) in pending.items():
                    record = self._get(
                        key[0],
                        key[1],
                        key[2],
                    )

                    record.sensor_sequences[
                        key[3]
                    ] = sequence

                    record.sensor_timestamps[
                        key[3]
                    ] = corrected_timestamp

                for frame in frames:
                    record = self._get(
                        tenant_id,
                        robot_id,
                        frame.clock_id,
                    )

                    self._audit(
                        record,
                        ClockEventType.FRAME_ALIGNED,
                        (
                            f"sensor={frame.sensor_id}, "
                            f"spread={spread_ms:.3f}ms"
                        ),
                    )

        for item in corrected:
            item.pop("_timestamp")

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "aligned": aligned,
            "spread_ms": spread_ms,
            "tolerance_ms": tolerance,
            "frames": corrected,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # RESET CLOCK
    # --------------------------------------------------------

    def reset_clock(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        clock_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ClockConflict(
                "Clock reset authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                clock_id,
            )

            record.samples.clear()
            record.sensor_sequences.clear()
            record.sensor_timestamps.clear()

            record.offset_ms = None
            record.rtt_ms = None
            record.drift_ppm = 0.0

            record.last_sequence_number = -1
            record.last_sample_monotonic = None

            record.status = ClockStatus.UNSYNCHRONIZED

            self._audit(
                record,
                ClockEventType.CLOCK_RESET,
                "Simulation clock estimate reset",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET CLOCK
    # --------------------------------------------------------

    def get_clock(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        clock_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                clock_id,
            )

            self._evaluate(record)

            return record.to_dict()

    # --------------------------------------------------------
    # GET HISTORY
    # --------------------------------------------------------

    def get_history(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        clock_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
        ):
            raise TimeSynchronizationValidationError(
                "Invalid history limit"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                clock_id,
            )

            return [
                item.to_dict()
                for item in list(
                    record.samples
                )[-limit:]
            ]

    # --------------------------------------------------------
    # LIST CLOCKS
    # --------------------------------------------------------

    def list_clocks(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        if robot_id is not None:
            validate_id(
                robot_id,
                "robot_id",
            )

        with self._lock:
            result = []

            for record in self._clocks.values():
                d = record.definition

                if d.tenant_id != tenant_id:
                    continue

                if (
                    robot_id is not None
                    and d.robot_id != robot_id
                ):
                    continue

                self._evaluate(record)

                result.append(
                    record.to_dict()
                )

            return sorted(
                result,
                key=lambda item: (
                    item["robot_id"],
                    item["clock_id"],
                ),
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        clocks = self.list_clocks(
            tenant_id=tenant_id
        )

        counts = {
            status.value: 0
            for status in ClockStatus
        }

        for clock in clocks:
            counts[
                clock["status"]
            ] += 1

        return {
            "tenant_id": tenant_id,
            "total_clocks": len(clocks),
            "status_counts": counts,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "clocks": self.list_clocks(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
                    if event.tenant_id == tenant_id
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

def create_simulated_exchange(
    *,
    tenant_id: str,
    robot_id: str,
    clock_id: str,
    sequence_number: int,
    offset_ms: float,
    network_delay_ms: float = 5.0,
) -> ClockExchangeSample:
    """
    Create a synthetic four-timestamp exchange.

    Assumes equal one-way network delays.
    """

    base = utc_now()

    t1 = base

    t2 = (
        base
        + timedelta(
            milliseconds=(
                network_delay_ms + offset_ms
            )
        )
    )

    t3 = t2 + timedelta(
        milliseconds=1.0
    )

    t4 = (
        base
        + timedelta(
            milliseconds=(
                network_delay_ms * 2.0 + 1.0
            )
        )
    )

    return ClockExchangeSample(
        tenant_id=tenant_id,
        robot_id=robot_id,
        clock_id=clock_id,
        sequence_number=sequence_number,
        t1=t1,
        t2=t2,
        t3=t3,
        t4=t4,
    )


def main() -> None:
    manager = RoboticsTimeSynchronizationManager()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"
    clock_id = "robot_clock"

    manager.register_clock(
        RobotClockDefinition(
            tenant_id=tenant_id,
            robot_id=robot_id,
            clock_id=clock_id,
            source=ClockSource.SIMULATION,
            description="Simulated robot clock",
        )
    )

    for sequence in range(1, 5):
        manager.ingest_exchange(
            create_simulated_exchange(
                tenant_id=tenant_id,
                robot_id=robot_id,
                clock_id=clock_id,
                sequence_number=sequence,
                offset_ms=5.0,
            )
        )

    print(
        "Clock status:",
        json.dumps(
            manager.get_clock(
                tenant_id=tenant_id,
                robot_id=robot_id,
                clock_id=clock_id,
            ),
            indent=2,
        ),
    )

    # Two simulated sensor frames separated by 5 ms.
    frame_time = utc_now() + timedelta(
        milliseconds=5.0
    )

    alignment = manager.align_sensor_frames(
        tenant_id=tenant_id,
        robot_id=robot_id,
        frames=[
            SensorTimestamp(
                tenant_id=tenant_id,
                robot_id=robot_id,
                clock_id=clock_id,
                sensor_id="camera_001",
                sequence_number=1,
                timestamp=frame_time,
                payload_reference="frame_camera_001",
            ),
            SensorTimestamp(
                tenant_id=tenant_id,
                robot_id=robot_id,
                clock_id=clock_id,
                sensor_id="lidar_001",
                sequence_number=1,
                timestamp=(
                    frame_time
                    + timedelta(milliseconds=5)
                ),
                payload_reference="frame_lidar_001",
            ),
        ],
    )

    print(
        "\nSensor alignment:",
        json.dumps(
            alignment,
            indent=2,
        ),
    )

    report = manager.export_json(
        "data/robotics/time_synchronization_report.json",
        tenant_id=tenant_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()
