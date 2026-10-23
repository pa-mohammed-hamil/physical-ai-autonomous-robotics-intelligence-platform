"""
Image-Based Damage Assessment Module

Real-time analysis of uploaded robot part images to determine:
- Damage type and severity
- Part identification
- Repair vs replacement recommendation
- Cost estimation
- Detailed damage report
"""

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


class ImageDamageError(Exception):
    """Base exception for image damage assessment."""
    pass


class ImageProcessingError(ImageDamageError):
    """Raised when image processing fails."""
    pass


class PartIdentificationError(ImageDamageError):
    """Raised when part cannot be identified."""
    pass


class DamageType(str, Enum):
    """Types of damage detected."""
    CRACK = "crack"
    DENT = "dent"
    CORROSION = "corrosion"
    SCRATCH = "scratch"
    WEAR = "wear"
    FRACTURE = "fracture"
    DEFORMATION = "deformation"
    BURN = "burn"
    RUST = "rust"
    BROKEN = "broken"
    LOOSE = "loose"
    MISSING_PART = "missing_part"


class SeverityLevel(str, Enum):
    """Damage severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class RecommendationType(str, Enum):
    """Repair or replacement recommendation."""
    REPAIR = "repair"
    REPLACE = "replace"
    MONITOR = "monitor"
    URGENT_REPLACE = "urgent_replace"


class PartCondition(str, Enum):
    """Overall part condition."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class DamageRegion:
    """Represents a damaged region in the image."""
    damage_type: DamageType
    severity: SeverityLevel
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    area_percentage: float
    description: str


@dataclass
class PartIdentification:
    """Identified robot part information."""
    part_name: str
    part_number: str
    category: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    confidence: float = 0.0


@dataclass
class CostBreakdown:
    """Cost breakdown for repair or replacement."""
    parts_cost: float
    labor_cost: float
    shipping_cost: float
    overhead_cost: float
    total_cost: float
    currency: str = "USD"
    estimated_time_hours: float = 0.0


@dataclass
class ImageDamageAssessment:
    """Complete damage assessment result."""
    assessment_id: str
    timestamp: datetime
    part_info: PartIdentification
    damage_regions: List[DamageRegion]
    overall_severity: SeverityLevel
    condition: PartCondition
    recommendation: RecommendationType
    repair_cost: Optional[CostBreakdown]
    replacement_cost: Optional[CostBreakdown]
    recommended_action: str
    detailed_findings: List[str]
    safety_concerns: List[str]
    estimated_remaining_life: Optional[str]
    confidence_score: float
    image_quality: str
    metadata: Dict = field(default_factory=dict)


class ImageDamageAssessor:
    """
    AI-powered image damage assessment for robot parts.
    
    Analyzes uploaded images to:
    1. Identify the robot part
    2. Detect and classify damage
    3. Assess severity
    4. Recommend repair or replacement
    5. Calculate costs
    """
    
    def __init__(
        self,
        parts_database: Dict[str, Dict] = None,
        cost_database: Dict[str, Dict] = None,
    ):
        """
        Initialize damage assessor.
        
        Args:
            parts_database: Database of known parts
            cost_database: Database of repair/replacement costs
        """
        self.parts_database = parts_database or self._load_default_parts()
        self.cost_database = cost_database or self._load_default_costs()
        
    def _load_default_parts(self) -> Dict[str, Dict]:
        """Load default parts database."""
        return {
            "motor_servo": {
                "name": "Servo Motor",
                "category": "Actuator",
                "avg_replacement_cost": 150.0,
                "avg_repair_cost": 45.0,
                "critical": True,
            },
            "wheel_assembly": {
                "name": "Wheel Assembly",
                "category": "Mobility",
                "avg_replacement_cost": 80.0,
                "avg_repair_cost": 25.0,
                "critical": False,
            },
            "camera_module": {
                "name": "Camera Module",
                "category": "Sensor",
                "avg_replacement_cost": 200.0,
                "avg_repair_cost": 60.0,
                "critical": True,
            },
            "lidar_sensor": {
                "name": "LiDAR Sensor",
                "category": "Sensor",
                "avg_replacement_cost": 500.0,
                "avg_repair_cost": 150.0,
                "critical": True,
            },
            "gripper_finger": {
                "name": "Gripper Finger",
                "category": "End Effector",
                "avg_replacement_cost": 120.0,
                "avg_repair_cost": 35.0,
                "critical": False,
            },
            "battery_pack": {
                "name": "Battery Pack",
                "category": "Power",
                "avg_replacement_cost": 300.0,
                "avg_repair_cost": 0.0,  # Batteries usually replaced
                "critical": True,
            },
            "joint_bearing": {
                "name": "Joint Bearing",
                "category": "Mechanical",
                "avg_replacement_cost": 60.0,
                "avg_repair_cost": 20.0,
                "critical": False,
            },
            "control_board": {
                "name": "Control Board",
                "category": "Electronics",
                "avg_replacement_cost": 250.0,
                "avg_repair_cost": 80.0,
                "critical": True,
            },
            "cable_harness": {
                "name": "Cable Harness",
                "category": "Wiring",
                "avg_replacement_cost": 40.0,
                "avg_repair_cost": 15.0,
                "critical": False,
            },
            "arm_link": {
                "name": "Arm Link",
                "category": "Structure",
                "avg_replacement_cost": 180.0,
                "avg_repair_cost": 55.0,
                "critical": True,
            },
        }
    
    def _load_default_costs(self) -> Dict[str, Dict]:
        """Load default cost database."""
        return {
            "labor_rates": {
                "technician": 75.0,  # per hour
                "specialist": 120.0,
                "engineer": 150.0,
            },
            "shipping": {
                "standard": 15.0,
                "expedited": 45.0,
                "international": 85.0,
            },
            "overhead_multiplier": 1.15,  # 15% overhead
        }
    
    async def assess_damage(
        self,
        image_data: bytes,
        user_selected_part: Optional[str] = None,
        robot_model: Optional[str] = None,
    ) -> ImageDamageAssessment:
        """
        Assess damage from uploaded image.
        
        Args:
            image_data: Raw image bytes
            user_selected_part: Optional user-specified part type
            robot_model: Optional robot model identifier
            
        Returns:
            Complete damage assessment
        """
        logger.info("Starting image damage assessment...")
        
        # Step 1: Preprocess image
        image = self._preprocess_image(image_data)
        image_quality = self._assess_image_quality(image)
        
        # Step 2: Identify part
        if user_selected_part:
            part_info = self._identify_part_from_selection(user_selected_part)
        else:
            part_info = await self._identify_part_from_image(image)
        
        logger.info(f"Identified part: {part_info.part_name}")
        
        # Step 3: Detect damage regions
        damage_regions = await self._detect_damage_regions(image, part_info)
        
        # Step 4: Assess overall severity
        overall_severity = self._calculate_overall_severity(damage_regions)
        condition = self._assess_condition(overall_severity, damage_regions)
        
        # Step 5: Generate recommendation
        recommendation = self._generate_recommendation(
            damage_regions,
            overall_severity,
            part_info,
        )
        
        # Step 6: Calculate costs
        repair_cost = self._calculate_repair_cost(
            part_info,
            damage_regions,
            overall_severity,
        )
        replacement_cost = self._calculate_replacement_cost(part_info)
        
        # Step 7: Generate detailed findings
        detailed_findings = self._generate_detailed_findings(
            damage_regions,
            part_info,
        )
        
        # Step 8: Identify safety concerns
        safety_concerns = self._identify_safety_concerns(
            damage_regions,
            part_info,
        )
        
        # Step 9: Estimate remaining life
        remaining_life = self._estimate_remaining_life(
            overall_severity,
            part_info,
        )
        
        # Step 10: Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            damage_regions,
            part_info,
            image_quality,
        )
        
        # Step 11: Determine recommended action
        recommended_action = self._get_recommended_action(
            recommendation,
            repair_cost,
            replacement_cost,
        )
        
        # Build final assessment
        assessment = ImageDamageAssessment(
            assessment_id=self._generate_assessment_id(),
            timestamp=datetime.utcnow(),
            part_info=part_info,
            damage_regions=damage_regions,
            overall_severity=overall_severity,
            condition=condition,
            recommendation=recommendation,
            repair_cost=repair_cost,
            replacement_cost=replacement_cost,
            recommended_action=recommended_action,
            detailed_findings=detailed_findings,
            safety_concerns=safety_concerns,
            estimated_remaining_life=remaining_life,
            confidence_score=confidence_score,
            image_quality=image_quality,
            metadata={
                "robot_model": robot_model,
                "user_selected_part": user_selected_part,
                "num_damage_regions": len(damage_regions),
            },
        )
        
        logger.info(f"Assessment complete: {recommendation.value} recommended")
        return assessment
    
    def _preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess uploaded image."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ImageProcessingError("Failed to decode image")
            
            # Resize if too large
            max_dimension = 1024
            height, width = image.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ImageProcessingError(f"Failed to process image: {e}")
    
    def _assess_image_quality(self, image: np.ndarray) -> str:
        """Assess image quality."""
        # Calculate sharpness (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        if laplacian_var > 100 and 50 < brightness < 200:
            return "excellent"
        elif laplacian_var > 50 and 30 < brightness < 220:
            return "good"
        elif laplacian_var > 20:
            return "fair"
        else:
            return "poor"
    
    def _identify_part_from_selection(self, part_key: str) -> PartIdentification:
        """Identify part from user selection."""
        part_data = self.parts_database.get(part_key)
        
        if not part_data:
            # Return generic part
            return PartIdentification(
                part_name=part_key.replace("_", " ").title(),
                part_number=f"UNKNOWN-{part_key}",
                category="Unknown",
                confidence=0.5,
            )
        
        return PartIdentification(
            part_name=part_data["name"],
            part_number=part_key.upper(),
            category=part_data["category"],
            confidence=1.0,
        )
    
    async def _identify_part_from_image(
        self,
        image: np.ndarray,
    ) -> PartIdentification:
        """Identify part from image using AI."""
        # In production, this would use a trained ML model
        # For now, return a simulated result
        
        # Analyze image features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Simple heuristic (would be ML model in production)
        if edge_density > 0.3:
            part_key = "motor_servo"
        elif edge_density > 0.2:
            part_key = "camera_module"
        else:
            part_key = "wheel_assembly"
        
        return self._identify_part_from_selection(part_key)
    
    async def _detect_damage_regions(
        self,
        image: np.ndarray,
        part_info: PartIdentification,
    ) -> List[DamageRegion]:
        """Detect damaged regions in the image."""
        damage_regions = []
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        height, width = image.shape[:2]
        total_area = height * width
        
        # Detect cracks (using edge detection)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Long thin contours might be cracks
                if aspect_ratio > 3 or aspect_ratio < 0.33:
                    damage_regions.append(DamageRegion(
                        damage_type=DamageType.CRACK,
                        severity=SeverityLevel.MODERATE,
                        bbox=(x, y, w, h),
                        confidence=0.75,
                        area_percentage=(area / total_area) * 100,
                        description=f"Crack detected in {part_info.part_name}",
                    ))
        
        # Detect corrosion (brownish/reddish areas)
        lower_rust = np.array([0, 50, 50])
        upper_rust = np.array([20, 255, 255])
        rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
        rust_area = np.sum(rust_mask > 0)
        
        if rust_area > 500:
            damage_regions.append(DamageRegion(
                damage_type=DamageType.CORROSION,
                severity=SeverityLevel.SEVERE if rust_area > 2000 else SeverityLevel.MODERATE,
                bbox=(0, 0, width, height),  # Simplified
                confidence=0.80,
                area_percentage=(rust_area / total_area) * 100,
                description=f"Corrosion detected on {part_info.part_name}",
            ))
        
        # Detect scratches (texture analysis)
        texture_variance = np.var(gray)
        if texture_variance > 1000:
            damage_regions.append(DamageRegion(
                damage_type=DamageType.SCRATCH,
                severity=SeverityLevel.MINOR,
                bbox=(0, 0, width, height),
                confidence=0.60,
                area_percentage=5.0,
                description=f"Surface scratches on {part_info.part_name}",
            ))
        
        # If no damage detected, create a minimal wear entry
        if not damage_regions:
            damage_regions.append(DamageRegion(
                damage_type=DamageType.WEAR,
                severity=SeverityLevel.MINOR,
                bbox=(0, 0, width, height),
                confidence=0.50,
                area_percentage=2.0,
                description=f"Minor wear on {part_info.part_name}",
            ))
        
        return damage_regions
    
    def _calculate_overall_severity(
        self,
        damage_regions: List[DamageRegion],
    ) -> SeverityLevel:
        """Calculate overall damage severity."""
        if not damage_regions:
            return SeverityLevel.MINOR
        
        # Weight by severity
        severity_scores = {
            SeverityLevel.MINOR: 1,
            SeverityLevel.MODERATE: 2,
            SeverityLevel.SEVERE: 3,
            SeverityLevel.CRITICAL: 4,
        }
        
        max_severity = max(
            severity_scores[region.severity]
            for region in damage_regions
        )
        
        # Reverse lookup
        for severity, score in severity_scores.items():
            if score == max_severity:
                return severity
        
        return SeverityLevel.MODERATE
    
    def _assess_condition(
        self,
        overall_severity: SeverityLevel,
        damage_regions: List[DamageRegion],
    ) -> PartCondition:
        """Assess overall part condition."""
        if overall_severity == SeverityLevel.CRITICAL:
            return PartCondition.FAILED
        elif overall_severity == SeverityLevel.SEVERE:
            return PartCondition.POOR
        elif overall_severity == SeverityLevel.MODERATE:
            return PartCondition.FAIR
        elif len(damage_regions) == 0:
            return PartCondition.EXCELLENT
        else:
            return PartCondition.GOOD
    
    def _generate_recommendation(
        self,
        damage_regions: List[DamageRegion],
        overall_severity: SeverityLevel,
        part_info: PartIdentification,
    ) -> RecommendationType:
        """Generate repair or replacement recommendation."""
        part_data = self.parts_database.get(part_info.part_number.lower())
        is_critical = part_data.get("critical", False) if part_data else False
        
        if overall_severity == SeverityLevel.CRITICAL:
            return RecommendationType.URGENT_REPLACE
        elif overall_severity == SeverityLevel.SEVERE:
            if is_critical:
                return RecommendationType.REPLACE
            else:
                return RecommendationType.REPAIR
        elif overall_severity == SeverityLevel.MODERATE:
            # Check damage types
            critical_damage = any(
                r.damage_type in [DamageType.FRACTURE, DamageType.BROKEN]
                for r in damage_regions
            )
            if critical_damage:
                return RecommendationType.REPLACE
            else:
                return RecommendationType.REPAIR
        else:
            return RecommendationType.MONITOR
    
    def _calculate_repair_cost(
        self,
        part_info: PartIdentification,
        damage_regions: List[DamageRegion],
        overall_severity: SeverityLevel,
    ) -> Optional[CostBreakdown]:
        """Calculate repair cost."""
        part_data = self.parts_database.get(part_info.part_number.lower())
        
        if not part_data or part_data["avg_repair_cost"] == 0:
            return None  # Not repairable
        
        base_repair_cost = part_data["avg_repair_cost"]
        
        # Adjust based on severity
        severity_multipliers = {
            SeverityLevel.MINOR: 0.8,
            SeverityLevel.MODERATE: 1.0,
            SeverityLevel.SEVERE: 1.5,
            SeverityLevel.CRITICAL: 2.0,
        }
        
        parts_cost = base_repair_cost * severity_multipliers[overall_severity]
        
        # Labor cost (estimate hours)
        estimated_hours = 1.0 + (0.5 * len(damage_regions))
        labor_cost = estimated_hours * self.cost_database["labor_rates"]["technician"]
        
        shipping_cost = self.cost_database["shipping"]["standard"]
        
        subtotal = parts_cost + labor_cost + shipping_cost
        overhead_cost = subtotal * (self.cost_database["overhead_multiplier"] - 1)
        total_cost = subtotal + overhead_cost
        
        return CostBreakdown(
            parts_cost=round(parts_cost, 2),
            labor_cost=round(labor_cost, 2),
            shipping_cost=round(shipping_cost, 2),
            overhead_cost=round(overhead_cost, 2),
            total_cost=round(total_cost, 2),
            estimated_time_hours=round(estimated_hours, 1),
        )
    
    def _calculate_replacement_cost(
        self,
        part_info: PartIdentification,
    ) -> CostBreakdown:
        """Calculate replacement cost."""
        part_data = self.parts_database.get(part_info.part_number.lower())
        
        parts_cost = part_data["avg_replacement_cost"] if part_data else 100.0
        
        # Labor for replacement (usually faster than repair)
        estimated_hours = 0.5
        labor_cost = estimated_hours * self.cost_database["labor_rates"]["technician"]
        
        shipping_cost = self.cost_database["shipping"]["expedited"]
        
        subtotal = parts_cost + labor_cost + shipping_cost
        overhead_cost = subtotal * (self.cost_database["overhead_multiplier"] - 1)
        total_cost = subtotal + overhead_cost
        
        return CostBreakdown(
            parts_cost=round(parts_cost, 2),
            labor_cost=round(labor_cost, 2),
            shipping_cost=round(shipping_cost, 2),
            overhead_cost=round(overhead_cost, 2),
            total_cost=round(total_cost, 2),
            estimated_time_hours=round(estimated_hours, 1),
        )
    
    def _generate_detailed_findings(
        self,
        damage_regions: List[DamageRegion],
        part_info: PartIdentification,
    ) -> List[str]:
        """Generate detailed findings list."""
        findings = []
        
        findings.append(f"Part identified: {part_info.part_name} ({part_info.category})")
        findings.append(f"Total damage regions detected: {len(damage_regions)}")
        
        for i, region in enumerate(damage_regions, 1):
            findings.append(
                f"Region {i}: {region.damage_type.value.title()} - "
                f"{region.severity.value.title()} severity "
                f"({region.area_percentage:.1f}% affected, "
                f"{region.confidence*100:.0f}% confidence)"
            )
        
        return findings
    
    def _identify_safety_concerns(
        self,
        damage_regions: List[DamageRegion],
        part_info: PartIdentification,
    ) -> List[str]:
        """Identify safety concerns."""
        concerns = []
        
        part_data = self.parts_database.get(part_info.part_number.lower())
        is_critical = part_data.get("critical", False) if part_data else False
        
        if is_critical:
            concerns.append(f"⚠️ {part_info.part_name} is a critical safety component")
        
        for region in damage_regions:
            if region.damage_type in [DamageType.FRACTURE, DamageType.BROKEN]:
                concerns.append(f"⚠️ Structural failure risk due to {region.damage_type.value}")
            
            if region.severity in [SeverityLevel.SEVERE, SeverityLevel.CRITICAL]:
                concerns.append(f"⚠️ {region.severity.value.title()} damage may cause operational failure")
        
        if not concerns:
            concerns.append("✓ No immediate safety concerns identified")
        
        return concerns
    
    def _estimate_remaining_life(
        self,
        overall_severity: SeverityLevel,
        part_info: PartIdentification,
    ) -> str:
        """Estimate remaining operational life."""
        estimates = {
            SeverityLevel.MINOR: "6-12 months with monitoring",
            SeverityLevel.MODERATE: "2-6 months with regular inspection",
            SeverityLevel.SEVERE: "Less than 1 month - immediate action required",
            SeverityLevel.CRITICAL: "Immediate replacement required",
        }
        return estimates[overall_severity]
    
    def _calculate_confidence_score(
        self,
        damage_regions: List[DamageRegion],
        part_info: PartIdentification,
        image_quality: str,
    ) -> float:
        """Calculate overall confidence score."""
        quality_scores = {
            "excellent": 1.0,
            "good": 0.9,
            "fair": 0.7,
            "poor": 0.5,
        }
        
        image_score = quality_scores.get(image_quality, 0.5)
        part_score = part_info.confidence
        
        if damage_regions:
            damage_score = np.mean([r.confidence for r in damage_regions])
        else:
            damage_score = 0.5
        
        overall_confidence = (image_score + part_score + damage_score) / 3
        return round(overall_confidence, 2)
    
    def _get_recommended_action(
        self,
        recommendation: RecommendationType,
        repair_cost: Optional[CostBreakdown],
        replacement_cost: CostBreakdown,
    ) -> str:
        """Get recommended action with cost justification."""
        if recommendation == RecommendationType.URGENT_REPLACE:
            return (
                f"🔴 URGENT: Replace immediately for safety. "
                f"Replacement cost: ${replacement_cost.total_cost:.2f}"
            )
        elif recommendation == RecommendationType.REPLACE:
            return (
                f"🟠 RECOMMEND REPLACEMENT: Cost ${replacement_cost.total_cost:.2f}, "
                f"Time: {replacement_cost.estimated_time_hours}h"
            )
        elif recommendation == RecommendationType.REPAIR:
            if repair_cost:
                savings = replacement_cost.total_cost - repair_cost.total_cost
                return (
                    f"🟢 RECOMMEND REPAIR: Cost ${repair_cost.total_cost:.2f}, "
                    f"Time: {repair_cost.estimated_time_hours}h "
                    f"(Save ${savings:.2f} vs replacement)"
                )
            else:
                return f"🟠 RECOMMEND REPLACEMENT: Part not repairable"
        else:
            return "🟢 MONITOR: Continue operation with regular inspection"
    
    def _generate_assessment_id(self) -> str:
        """Generate unique assessment ID."""
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        import random
        random_suffix = random.randint(1000, 9999)
        return f"ASSESS-{timestamp}-{random_suffix}"
