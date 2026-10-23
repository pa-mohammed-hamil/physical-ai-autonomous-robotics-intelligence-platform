
"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/report_service.py

Centralized Robotics Reporting Service

Python 3.11+
Standard library only.

Features:
    - Fleet and robot reporting
    - Health and maintenance reporting
    - Repair and cost reporting
    - Navigation and task reporting
    - Perception and safety reporting
    - AI model inference reporting
    - Simulation evaluation reporting
    - KPI aggregation
    - Time-period filtering
    - Report generation and storage
    - JSON / CSV / HTML export
    - Report comparison
    - Report history

Integration:
    Accepts normalized records or reports from
    existing services.

Safety:
    Read-only reporting.
    No hardware or actuator control.
"""

from __future__ import annotations

import csv
import html
import json
import math
import threading
import uuid

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ReportServiceError(Exception):
    pass


class ReportValidationError(ReportServiceError):
    pass


class ReportNotFoundError(ReportServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ReportType(str, Enum):
    FLEET = "fleet"
    ROBOT_HEALTH = "robot_health"
    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    COST = "cost"
    SAFETY = "safety"
    MODEL = "model"
    SIMULATION = "simulation"
    OPERATIONS = "operations"
    COMPREHENSIVE = "comprehensive"


class ReportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    HTML = "html"


class MetricAggregation(str, Enum):
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def require_text(
    value: str,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportValidationError(
            f"{name} is required"
        )

    return value.strip()


def finite(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ReportValidationError(
            f"{name} must be finite"
        )

    return float(value)


def parse_datetime(
    value: str | datetime,
) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError) as exc:
            raise ReportValidationError(
                "Invalid ISO datetime"
            ) from exc

    if result.tzinfo is None:
        raise ReportValidationError(
            "Datetime must include timezone"
        )

    return result.astimezone(
        timezone.utc
    )


def safe_json(
    value: Any,
) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ReportValidationError(
            "Data must be JSON serializable"
        ) from exc


def nested_value(
    record: Mapping[str, Any],
    path: str,
) -> Any:
    current: Any = record

    for key in path.split("."):
        if not isinstance(current, Mapping):
            return None

        current = current.get(key)

    return current


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ReportServiceConfig:
    report_directory: str = "data/reports"

    currency: str = "INR"

    maximum_records_per_source: int = 100000

    maximum_saved_reports: int = 10000

    maximum_report_rows: int = 100000

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        require_text(
            self.report_directory,
            "report_directory",
        )

        require_text(
            self.currency,
            "currency",
        )

        for name in (
            "maximum_records_per_source",
            "maximum_saved_reports",
            "maximum_report_rows",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ReportValidationError(
                    f"{name} must be a positive integer"
                )

        if self.allow_hardware_control:
            raise ReportValidationError(
                "Hardware control is disabled"
            )


# ============================================================
# REPORT PERIOD
# ============================================================

@dataclass(frozen=True)
class ReportPeriod:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start = parse_datetime(self.start)
        end = parse_datetime(self.end)

        if start > end:
            raise ReportValidationError(
                "Report start must be before end"
            )

        object.__setattr__(
            self,
            "start",
            start,
        )

        object.__setattr__(
            self,
            "end",
            end,
        )

    def contains(
        self,
        timestamp: str | datetime,
    ) -> bool:
        value = parse_datetime(
            timestamp
        )

        return self.start <= value <= self.end

    def to_dict(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


# ============================================================
# REPORT DATA RECORD
# ============================================================

@dataclass(frozen=True)
class ReportRecord:
    record_id: str
    source: ReportType
    robot_id: str | None

    timestamp: str

    data: dict[str, Any]

    def __post_init__(self) -> None:
        require_text(
            self.record_id,
            "record_id",
        )

        parse_datetime(
            self.timestamp
        )

        safe_json(
            self.data
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source": self.source.value,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# ============================================================
# KPI DEFINITION
# ============================================================

@dataclass(frozen=True)
class KPI:
    name: str
    source: ReportType

    field_path: str

    aggregation: MetricAggregation

    unit: str = ""

    def __post_init__(self) -> None:
        require_text(
            self.name,
            "name",
        )

        require_text(
            self.field_path,
            "field_path",
        )


# ============================================================
# KPI RESULT
# ============================================================

@dataclass(frozen=True)
class KPIResult:
    name: str
    value: float | int | None

    unit: str

    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "sample_count": self.sample_count,
        }


# ============================================================
# KPI AGGREGATOR
# ============================================================

class RoboticsKPIAggregator:

    @staticmethod
    def calculate(
        records: Sequence[ReportRecord],
        kpi: KPI,
    ) -> KPIResult:
        matching = [
            record
            for record in records
            if record.source == kpi.source
        ]

        if kpi.aggregation == MetricAggregation.COUNT:
            if kpi.field_path == "*":
                count = len(matching)
            else:
                count = sum(
                    nested_value(
                        record.data,
                        kpi.field_path,
                    ) is not None
                    for record in matching
                )

            return KPIResult(
                name=kpi.name,
                value=count,
                unit=kpi.unit,
                sample_count=count,
            )

        values = []

        for record in matching:
            value = nested_value(
                record.data,
                kpi.field_path,
            )

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                values.append(float(value))

        if not values:
            return KPIResult(
                name=kpi.name,
                value=None,
                unit=kpi.unit,
                sample_count=0,
            )

        if kpi.aggregation == MetricAggregation.SUM:
            result = sum(values)

        elif kpi.aggregation == MetricAggregation.AVERAGE:
            result = sum(values) / len(values)

        elif kpi.aggregation == MetricAggregation.MINIMUM:
            result = min(values)

        elif kpi.aggregation == MetricAggregation.MAXIMUM:
            result = max(values)

        else:
            raise ReportValidationError(
                "Unsupported aggregation"
            )

        return KPIResult(
            name=kpi.name,
            value=round(result, 4),
            unit=kpi.unit,
            sample_count=len(values),
        )


# ============================================================
# GENERATED REPORT
# ============================================================

@dataclass
class GeneratedReport:
    report_id: str
    title: str

    report_type: ReportType

    created_at: str

    period: ReportPeriod | None

    robot_id: str | None

    record_count: int

    kpis: list[KPIResult]

    sections: dict[str, Any]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "report_type": self.report_type.value,
            "created_at": self.created_at,
            "period": (
                self.period.to_dict()
                if self.period
                else None
            ),
            "robot_id": self.robot_id,
            "record_count": self.record_count,
            "kpis": [
                kpi.to_dict()
                for kpi in self.kpis
            ],
            "sections": self.sections,
            "metadata": self.metadata,
        }


# ============================================================
# REPORT SERVICE
# ============================================================

class RoboticsReportService:

    def __init__(
        self,
        config: ReportServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or ReportServiceConfig()
        )

        self.records: dict[
            ReportType,
            list[ReportRecord]
        ] = defaultdict(list)

        self.reports: dict[
            str,
            GeneratedReport
        ] = {}

        self.kpi_definitions: dict[
            str,
            KPI
        ] = {}

        self.aggregator = RoboticsKPIAggregator()

        self._lock = threading.RLock()

        self.report_root = Path(
            self.config.report_directory
        )

        self.report_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._register_default_kpis()

    # --------------------------------------------------------
    # DEFAULT KPIs
    # --------------------------------------------------------

    def _register_default_kpis(
        self,
    ) -> None:
        defaults = [
            KPI(
                "Average Robot Health",
                ReportType.ROBOT_HEALTH,
                "overall_score",
                MetricAggregation.AVERAGE,
                "%",
            ),
            KPI(
                "Navigation Tasks",
                ReportType.NAVIGATION,
                "*",
                MetricAggregation.COUNT,
                "tasks",
            ),
            KPI(
                "Total Navigation Distance",
                ReportType.NAVIGATION,
                "distance_travelled_m",
                MetricAggregation.SUM,
                "m",
            ),
            KPI(
                "Repair Cost",
                ReportType.REPAIR,
                "total_cost",
                MetricAggregation.SUM,
                self.config.currency,
            ),
            KPI(
                "Maintenance Cost",
                ReportType.MAINTENANCE,
                "total_cost",
                MetricAggregation.SUM,
                self.config.currency,
            ),
            KPI(
                "Energy Consumption",
                ReportType.COST,
                "energy_consumed_kwh",
                MetricAggregation.SUM,
                "kWh",
            ),
            KPI(
                "Perception Detections",
                ReportType.PERCEPTION,
                "detection_count",
                MetricAggregation.SUM,
                "detections",
            ),
            KPI(
                "Safety Alerts",
                ReportType.SAFETY,
                "*",
                MetricAggregation.COUNT,
                "alerts",
            ),
            KPI(
                "Average Inference Latency",
                ReportType.MODEL,
                "latency_ms",
                MetricAggregation.AVERAGE,
                "ms",
            ),
            KPI(
                "Average Simulation Score",
                ReportType.SIMULATION,
                "overall_score",
                MetricAggregation.AVERAGE,
                "%",
            ),
        ]

        for kpi in defaults:
            self.kpi_definitions[
                kpi.name
            ] = kpi

    # --------------------------------------------------------
    # REGISTER CUSTOM KPI
    # --------------------------------------------------------

    def register_kpi(
        self,
        kpi: KPI,
    ) -> None:
        with self._lock:
            self.kpi_definitions[
                kpi.name
            ] = kpi

    # --------------------------------------------------------
    # INGEST RECORD
    # --------------------------------------------------------

    def ingest_record(
        self,
        *,
        source: ReportType,
        data: Mapping[str, Any],
        robot_id: str | None = None,
        timestamp: str | None = None,
    ) -> ReportRecord:
        source = ReportType(source)

        record = ReportRecord(
            record_id=str(uuid.uuid4()),
            source=source,
            robot_id=robot_id,
            timestamp=timestamp or utc_now(),
            data=safe_json(dict(data)),
        )

        with self._lock:
            bucket = self.records[source]

            if (
                len(bucket)
                >= self.config.maximum_records_per_source
            ):
                raise ReportValidationError(
                    "Record capacity reached for source"
                )

            bucket.append(record)

        return record

    # --------------------------------------------------------
    # INGEST BATCH
    # --------------------------------------------------------

    def ingest_batch(
        self,
        *,
        source: ReportType,
        records: Iterable[Mapping[str, Any]],
        robot_id_field: str = "robot_id",
        timestamp_field: str = "created_at",
    ) -> int:
        count = 0

        for item in records:
            robot_id = item.get(
                robot_id_field
            )

            timestamp = item.get(
                timestamp_field
            )

            self.ingest_record(
                source=source,
                data=item,
                robot_id=(
                    str(robot_id)
                    if robot_id is not None
                    else None
                ),
                timestamp=(
                    str(timestamp)
                    if timestamp is not None
                    else None
                ),
            )

            count += 1

        return count

    # --------------------------------------------------------
    # FILTER RECORDS
    # --------------------------------------------------------

    def query_records(
        self,
        *,
        source: ReportType | None = None,
        robot_id: str | None = None,
        period: ReportPeriod | None = None,
    ) -> list[ReportRecord]:
        with self._lock:
            if source is None:
                records = [
                    record
                    for bucket in self.records.values()
                    for record in bucket
                ]
            else:
                records = list(
                    self.records.get(
                        ReportType(source),
                        (),
                    )
                )

        if robot_id is not None:
            records = [
                record
                for record in records
                if record.robot_id == robot_id
            ]

        if period is not None:
            records = [
                record
                for record in records
                if period.contains(
                    record.timestamp
                )
            ]

        return sorted(
            records,
            key=lambda item: item.timestamp,
        )

    # --------------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------------

    def generate_report(
        self,
        *,
        title: str,
        report_type: ReportType,
        period: ReportPeriod | None = None,
        robot_id: str | None = None,
        include_records: bool = False,
    ) -> GeneratedReport:
        require_text(
            title,
            "title",
        )

        report_type = ReportType(
            report_type
        )

        with self._lock:
            if (
                len(self.reports)
                >= self.config.maximum_saved_reports
            ):
                raise ReportValidationError(
                    "Maximum saved reports reached"
                )

            records = self.query_records(
                source=(
                    None
                    if report_type
                    in (
                        ReportType.COMPREHENSIVE,
                        ReportType.OPERATIONS,
                    )
                    else report_type
                ),
                robot_id=robot_id,
                period=period,
            )

            if (
                len(records)
                > self.config.maximum_report_rows
            ):
                raise ReportValidationError(
                    "Report exceeds maximum row count"
                )

            if report_type == ReportType.OPERATIONS:
                allowed_sources = {
                    ReportType.FLEET,
                    ReportType.NAVIGATION,
                    ReportType.MAINTENANCE,
                    ReportType.REPAIR,
                    ReportType.ROBOT_HEALTH,
                }

                records = [
                    record
                    for record in records
                    if record.source in allowed_sources
                ]

            source_counts = Counter(
                record.source.value
                for record in records
            )

            robot_counts = Counter(
                record.robot_id
                for record in records
                if record.robot_id is not None
            )

            kpis = []

            for definition in self.kpi_definitions.values():
                if (
                    report_type
                    not in (
                        ReportType.COMPREHENSIVE,
                        ReportType.OPERATIONS,
                    )
                    and definition.source != report_type
                ):
                    continue

                if (
                    report_type == ReportType.OPERATIONS
                    and definition.source not in {
                        ReportType.FLEET,
                        ReportType.NAVIGATION,
                        ReportType.MAINTENANCE,
                        ReportType.REPAIR,
                        ReportType.ROBOT_HEALTH,
                    }
                ):
                    continue

                kpis.append(
                    self.aggregator.calculate(
                        records,
                        definition,
                    )
                )

            sections = {
                "source_distribution": dict(
                    source_counts
                ),
                "robot_distribution": dict(
                    robot_counts
                ),
                "summary": {
                    "total_records": len(records),
                    "unique_robots": len(robot_counts),
                    "sources": len(source_counts),
                },
            }

            if include_records:
                sections["records"] = [
                    record.to_dict()
                    for record in records
                ]

            report = GeneratedReport(
                report_id=str(uuid.uuid4()),
                title=title,
                report_type=report_type,
                created_at=utc_now(),
                period=period,
                robot_id=robot_id,
                record_count=len(records),
                kpis=kpis,
                sections=sections,
                metadata={
                    "currency": self.config.currency,
                    "simulation_only": True,
                },
            )

            self.reports[
                report.report_id
            ] = report

            return report

    # --------------------------------------------------------
    # GET REPORT
    # --------------------------------------------------------

    def get_report(
        self,
        report_id: str,
    ) -> GeneratedReport:
        report = self.reports.get(
            report_id
        )

        if report is None:
            raise ReportNotFoundError(
                f"Report not found: {report_id}"
            )

        return report

    # --------------------------------------------------------
    # REPORT COMPARISON
    # --------------------------------------------------------

    def compare_reports(
        self,
        first_report_id: str,
        second_report_id: str,
    ) -> dict[str, Any]:
        first = self.get_report(
            first_report_id
        )

        second = self.get_report(
            second_report_id
        )

        first_kpis = {
            kpi.name: kpi.value
            for kpi in first.kpis
        }

        second_kpis = {
            kpi.name: kpi.value
            for kpi in second.kpis
        }

        comparison = {}

        for name in (
            first_kpis.keys()
            & second_kpis.keys()
        ):
            old = first_kpis[name]
            new = second_kpis[name]

            if old is None or new is None:
                comparison[name] = {
                    "first": old,
                    "second": new,
                    "change": None,
                    "change_percent": None,
                }

                continue

            difference = new - old

            percent = (
                difference / abs(old) * 100.0
                if old != 0
                else None
            )

            comparison[name] = {
                "first": old,
                "second": new,
                "change": round(
                    difference,
                    4,
                ),
                "change_percent": (
                    round(percent, 2)
                    if percent is not None
                    else None
                ),
            }

        return {
            "first_report_id": first_report_id,
            "second_report_id": second_report_id,
            "comparison": comparison,
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        report_id: str,
        path: str | Path | None = None,
    ) -> Path:
        report = self.get_report(
            report_id
        )

        output = (
            Path(path)
            if path is not None
            else self.report_root
            / f"{report_id}.json"
        )

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

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    def export_csv(
        self,
        report_id: str,
        path: str | Path | None = None,
    ) -> Path:
        report = self.get_report(
            report_id
        )

        output = (
            Path(path)
            if path is not None
            else self.report_root
            / f"{report_id}.csv"
        )

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
                    "metric",
                    "value",
                    "unit",
                    "sample_count",
                ],
            )

            writer.writeheader()

            for kpi in report.kpis:
                writer.writerow({
                    "metric": kpi.name,
                    "value": kpi.value,
                    "unit": kpi.unit,
                    "sample_count": kpi.sample_count,
                })

        return output

    # --------------------------------------------------------
    # EXPORT HTML
    # --------------------------------------------------------

    def export_html(
        self,
        report_id: str,
        path: str | Path | None = None,
    ) -> Path:
        report = self.get_report(
            report_id
        )

        output = (
            Path(path)
            if path is not None
            else self.report_root
            / f"{report_id}.html"
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []

        for kpi in report.kpis:
            value = (
                "N/A"
                if kpi.value is None
                else str(kpi.value)
            )

            rows.append(
                "<tr>"
                f"<td>{html.escape(kpi.name)}</td>"
                f"<td>{html.escape(value)}</td>"
                f"<td>{html.escape(kpi.unit)}</td>"
                f"<td>{kpi.sample_count}</td>"
                "</tr>"
            )

        document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">
<title>{html.escape(report.title)}</title>
<style>
:root {{
    color-scheme: light;
    font-family: Arial, sans-serif;
}}
body {{
    margin: 0;
    background: #f3f5f9;
    color: #18212f;
}}
main {{
    max-width: 1100px;
    margin: 40px auto;
    padding: 24px;
}}
header {{
    background: #152b4a;
    color: white;
    border-radius: 12px;
    padding: 30px;
}}
section {{
    background: white;
    padding: 24px;
    border-radius: 12px;
    margin-top: 20px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
}}
th, td {{
    text-align: left;
    padding: 12px;
    border-bottom: 1px solid #e4e8ef;
}}
th {{
    background: #f0f3f8;
}}
.meta {{
    opacity: 0.85;
}}
</style>
</head>
<body>
<main>
<header>
<h1>{html.escape(report.title)}</h1>
<p class="meta">
Type: {html.escape(report.report_type.value)}
</p>
<p class="meta">
Generated: {html.escape(report.created_at)}
</p>
</header>
<section>
<h2>Report Summary</h2>
<p>Total records: {report.record_count}</p>
<p>Robot: {html.escape(report.robot_id or "All")}</p>
</section>
<section>
<h2>Key Performance Indicators</h2>
<table>
<thead>
<tr>
<th>Metric</th>
<th>Value</th>
<th>Unit</th>
<th>Samples</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</section>
</main>
</body>
</html>
"""

        output.write_text(
            document,
            encoding="utf-8",
        )

        return output

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "total_records": sum(
                    len(bucket)
                    for bucket in self.records.values()
                ),
                "total_reports": len(self.reports),
                "registered_kpis": len(
                    self.kpi_definitions
                ),
                "source_distribution": {
                    source.value: len(bucket)
                    for source, bucket
                    in self.records.items()
                },
                "report_type_distribution": dict(
                    Counter(
                        report.report_type.value
                        for report in self.reports.values()
                    )
                ),
                "simulation_only": True,
            }


# ============================================================
# SERVICE ADAPTERS
# ============================================================

class RoboticsReportAdapters:
    """
    Normalize outputs from existing service modules.

    These adapters accept dictionary payloads
    produced by service .to_dict() methods.
    """

    @staticmethod
    def robot_health(
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "robot_id": report.get("robot_id"),
            "overall_score": report.get(
                "overall_score"
            ),
            "overall_status": report.get(
                "overall_status"
            ),
            "rul_hours": report.get(
                "rul_hours"
            ),
            "alert_count": len(
                report.get("alerts", [])
            ),
            "recommendation_count": len(
                report.get("recommendations", [])
            ),
        }

    @staticmethod
    def navigation(
        task: Mapping[str, Any],
        *,
        distance_travelled_m: float | None = None,
    ) -> dict[str, Any]:
        return {
            "robot_id": task.get("robot_id"),
            "task_id": task.get("task_id"),
            "status": task.get("status"),
            "navigation_steps": task.get(
                "navigation_steps",
                0,
            ),
            "replanning_count": task.get(
                "replanning_count",
                0,
            ),
            "distance_travelled_m": (
                distance_travelled_m
            ),
        }

    @staticmethod
    def perception(
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "robot_id": result.get("robot_id"),
            "detection_count": len(
                result.get("detections", [])
            ),
            "track_count": len(
                result.get("tracks", [])
            ),
            "alert_count": len(
                result.get("alerts", [])
            ),
        }

    @staticmethod
    def repair(
        work_order: Mapping[str, Any],
        *,
        total_cost: float | None = None,
    ) -> dict[str, Any]:
        return {
            "robot_id": work_order.get("robot_id"),
            "work_order_id": work_order.get(
                "work_order_id"
            ),
            "status": work_order.get("status"),
            "component": work_order.get("component"),
            "severity": work_order.get("severity"),
            "total_cost": total_cost,
        }

    @staticmethod
    def model_inference(
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "model_id": result.get("model_id"),
            "status": result.get("status"),
            "latency_ms": result.get(
                "latency_ms"
            ),
        }

    @staticmethod
    def simulation(
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "robot_id": result.get("robot_id"),
            "overall_score": result.get(
                "overall_score"
            ),
            "status": result.get("status"),
        }


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsReportService()

    service.ingest_record(
        source=ReportType.ROBOT_HEALTH,
        robot_id="robot_001",
        data={
            "overall_score": 91.5,
            "overall_status": "healthy",
            "rul_hours": 420.0,
        },
    )

    service.ingest_record(
        source=ReportType.ROBOT_HEALTH,
        robot_id="robot_002",
        data={
            "overall_score": 64.0,
            "overall_status": "warning",
            "rul_hours": 85.0,
        },
    )

    service.ingest_record(
        source=ReportType.NAVIGATION,
        robot_id="robot_001",
        data={
            "task_id": "navigation_001",
            "status": "completed",
            "distance_travelled_m": 24.5,
        },
    )

    service.ingest_record(
        source=ReportType.PERCEPTION,
        robot_id="robot_001",
        data={
            "detection_count": 18,
            "track_count": 7,
            "alert_count": 2,
        },
    )

    service.ingest_record(
        source=ReportType.REPAIR,
        robot_id="robot_002",
        data={
            "work_order_id": "repair_001",
            "status": "completed",
            "total_cost": 5600.0,
        },
    )

    service.ingest_record(
        source=ReportType.MODEL,
        data={
            "model_id": "model_001",
            "status": "success",
            "latency_ms": 32.5,
        },
    )

    report = service.generate_report(
        title="Robotics Fleet Intelligence Report",
        report_type=ReportType.COMPREHENSIVE,
        include_records=True,
    )

    json_path = service.export_json(
        report.report_id,
        "data/reports/robotics_report.json",
    )

    csv_path = service.export_csv(
        report.report_id,
        "data/reports/robotics_kpis.csv",
    )

    html_path = service.export_html(
        report.report_id,
        "data/reports/robotics_dashboard.html",
    )

    print(
        json.dumps(
            {
                "report": report.to_dict(),
                "files": [
                    str(json_path),
                    str(csv_path),
                    str(html_path),
                ],
                "statistics": service.statistics(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
