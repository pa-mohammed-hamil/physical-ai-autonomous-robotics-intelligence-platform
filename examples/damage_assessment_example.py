"""
Image-Based Damage Assessment Examples

Demonstrates how to use the damage assessment API to analyze robot parts
and get repair vs replacement recommendations with cost estimates.
"""

import asyncio
from pathlib import Path
from typing import Optional

import httpx


API_BASE_URL = "http://localhost:8000/api/v1/damage-assessment"


class DamageAssessmentClient:
    """Client for damage assessment API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        """Initialize client."""
        self.base_url = base_url
    
    async def upload_image(
        self,
        image_path: str,
        part_type: Optional[str] = None,
        robot_model: Optional[str] = None,
    ) -> dict:
        """
        Upload image for damage assessment.
        
        Args:
            image_path: Path to image file
            part_type: Optional part type identifier
            robot_model: Optional robot model
            
        Returns:
            Assessment results
        """
        async with httpx.AsyncClient() as client:
            # Prepare form data
            with open(image_path, 'rb') as f:
                files = {'image': (Path(image_path).name, f, 'image/jpeg')}
                
                data = {}
                if part_type:
                    data['part_type'] = part_type
                if robot_model:
                    data['robot_model'] = robot_model
                
                response = await client.post(
                    f"{self.base_url}/upload",
                    files=files,
                    data=data,
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()
    
    async def get_available_parts(self) -> dict:
        """Get list of available parts."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/parts")
            response.raise_for_status()
            return response.json()
    
    async def get_cost_estimate(
        self,
        part_type: str,
        severity: str = "moderate",
    ) -> dict:
        """
        Get cost estimate for a part.
        
        Args:
            part_type: Part identifier
            severity: Damage severity (minor, moderate, severe, critical)
            
        Returns:
            Cost estimates
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/cost-estimate/{part_type}",
                params={"severity": severity},
            )
            response.raise_for_status()
            return response.json()


# ============================================================================
# Example 1: Assess Damaged Motor
# ============================================================================


async def example_motor_assessment():
    """Assess a damaged servo motor."""
    print("\n" + "="*80)
    print("Example 1: Damaged Motor Assessment")
    print("="*80)
    
    client = DamageAssessmentClient()
    
    # Note: You would provide actual image path here
    # For demo, we'll simulate the response
    print("\n📸 Uploading motor image for assessment...")
    print("Image: damaged_motor.jpg")
    print("Part Type: motor_servo")
    
    # In real usage:
    # result = await client.upload_image(
    #     image_path="path/to/damaged_motor.jpg",
    #     part_type="motor_servo",
    #     robot_model="RoboArm-X1"
    # )
    
    # Simulated response
    result = {
        "assessment_id": "ASSESS-20261016120000-1234",
        "part_info": {
            "part_name": "Servo Motor",
            "part_number": "MOTOR_SERVO",
            "category": "Actuator",
            "confidence": 0.95
        },
        "overall_severity": "moderate",
        "condition": "fair",
        "recommendation": "repair",
        "repair_cost": {
            "total_cost": 58.75,
            "parts_cost": 45.00,
            "labor_cost": 75.00 * 0.8,
            "estimated_time_hours": 0.8
        },
        "replacement_cost": {
            "total_cost": 195.50,
            "parts_cost": 150.00,
            "estimated_time_hours": 0.5
        },
        "recommended_action": "🟢 RECOMMEND REPAIR: Cost $58.75, Time: 0.8h (Save $136.75 vs replacement)",
        "damage_regions": [
            {
                "damage_type": "crack",
                "severity": "moderate",
                "area_percentage": 3.5,
                "confidence": 0.75,
                "description": "Crack detected in Servo Motor"
            }
        ],
        "safety_concerns": [
            "⚠️ Servo Motor is a critical safety component",
            "✓ No immediate safety concerns identified"
        ],
        "estimated_remaining_life": "2-6 months with regular inspection"
    }
    
    print(f"\n✅ Assessment completed: {result['assessment_id']}")
    print(f"\n🔧 Part: {result['part_info']['part_name']}")
    print(f"   Category: {result['part_info']['category']}")
    print(f"   Severity: {result['overall_severity'].upper()}")
    print(f"   Condition: {result['condition'].upper()}")
    
    print(f"\n💰 Cost Analysis:")
    if result['repair_cost']:
        print(f"   Repair: ${result['repair_cost']['total_cost']:.2f} ({result['repair_cost']['estimated_time_hours']}h)")
    print(f"   Replace: ${result['replacement_cost']['total_cost']:.2f} ({result['replacement_cost']['estimated_time_hours']}h)")
    
    if result['repair_cost']:
        savings = result['replacement_cost']['total_cost'] - result['repair_cost']['total_cost']
        print(f"   💵 Savings with repair: ${savings:.2f}")
    
    print(f"\n📋 Recommendation:")
    print(f"   {result['recommended_action']}")
    
    print(f"\n⏱️ Estimated Remaining Life:")
    print(f"   {result['estimated_remaining_life']}")
    
    print(f"\n⚠️ Safety Concerns:")
    for concern in result['safety_concerns']:
        print(f"   {concern}")
    
    return result


# ============================================================================
# Example 2: Assess Damaged Camera Module
# ============================================================================


async def example_camera_assessment():
    """Assess a damaged camera module."""
    print("\n" + "="*80)
    print("Example 2: Damaged Camera Module Assessment")
    print("="*80)
    
    print("\n📸 Uploading camera image...")
    
    # Simulated severe damage requiring replacement
    result = {
        "assessment_id": "ASSESS-20261016120100-5678",
        "part_info": {
            "part_name": "Camera Module",
            "part_number": "CAMERA_MODULE",
            "category": "Sensor",
            "confidence": 0.92
        },
        "overall_severity": "severe",
        "condition": "poor",
        "recommendation": "replace",
        "repair_cost": {
            "total_cost": 95.50,
            "estimated_time_hours": 1.5
        },
        "replacement_cost": {
            "total_cost": 265.00,
            "parts_cost": 200.00,
            "estimated_time_hours": 0.5
        },
        "recommended_action": "🟠 RECOMMEND REPLACEMENT: Cost $265.00, Time: 0.5h",
        "damage_regions": [
            {
                "damage_type": "crack",
                "severity": "severe",
                "area_percentage": 15.2
            },
            {
                "damage_type": "broken",
                "severity": "severe",
                "area_percentage": 8.3
            }
        ],
        "safety_concerns": [
            "⚠️ Camera Module is a critical safety component",
            "⚠️ Severe damage may cause operational failure"
        ],
        "estimated_remaining_life": "Less than 1 month - immediate action required"
    }
    
    print(f"\n✅ Assessment completed")
    print(f"\n🔧 Part: {result['part_info']['part_name']}")
    print(f"   Severity: {result['overall_severity'].upper()} ❌")
    print(f"   Damage Regions: {len(result['damage_regions'])}")
    
    print(f"\n📋 Recommendation:")
    print(f"   {result['recommended_action']}")
    
    print(f"\n⚠️ Safety Concerns:")
    for concern in result['safety_concerns']:
        print(f"   {concern}")
    
    return result


# ============================================================================
# Example 3: Get Available Parts
# ============================================================================


async def example_list_parts():
    """List all available parts for assessment."""
    print("\n" + "="*80)
    print("Example 3: List Available Parts")
    print("="*80)
    
    client = DamageAssessmentClient()
    
    # In real usage:
    # result = await client.get_available_parts()
    
    # Simulated response
    result = {
        "parts": [
            {
                "id": "motor_servo",
                "name": "Servo Motor",
                "category": "Actuator",
                "avg_repair_cost": 45.00,
                "avg_replacement_cost": 150.00,
                "is_critical": True,
                "repairable": True
            },
            {
                "id": "camera_module",
                "name": "Camera Module",
                "category": "Sensor",
                "avg_repair_cost": 60.00,
                "avg_replacement_cost": 200.00,
                "is_critical": True,
                "repairable": True
            },
            {
                "id": "wheel_assembly",
                "name": "Wheel Assembly",
                "category": "Mobility",
                "avg_repair_cost": 25.00,
                "avg_replacement_cost": 80.00,
                "is_critical": False,
                "repairable": True
            },
            {
                "id": "battery_pack",
                "name": "Battery Pack",
                "category": "Power",
                "avg_repair_cost": 0.00,
                "avg_replacement_cost": 300.00,
                "is_critical": True,
                "repairable": False
            }
        ]
    }
    
    print(f"\n📚 Available Parts ({len(result['parts'])} total):\n")
    
    for part in result['parts']:
        critical = "🔴 CRITICAL" if part['is_critical'] else "⚪ Standard"
        repairable = "✅ Repairable" if part['repairable'] else "❌ Not Repairable"
        
        print(f"  {part['name']}")
        print(f"    ID: {part['id']}")
        print(f"    Category: {part['category']} | {critical} | {repairable}")
        print(f"    Repair Cost: ${part['avg_repair_cost']:.2f}")
        print(f"    Replace Cost: ${part['avg_replacement_cost']:.2f}")
        print()
    
    return result


# ============================================================================
# Example 4: Get Cost Estimate
# ============================================================================


async def example_cost_estimate():
    """Get cost estimate for different severity levels."""
    print("\n" + "="*80)
    print("Example 4: Cost Estimates by Severity")
    print("="*80)
    
    client = DamageAssessmentClient()
    
    part_type = "motor_servo"
    severities = ["minor", "moderate", "severe", "critical"]
    
    print(f"\n💰 Cost Estimates for {part_type}:\n")
    
    for severity in severities:
        # In real usage:
        # estimate = await client.get_cost_estimate(part_type, severity)
        
        # Simulated estimates
        multipliers = {
            "minor": 0.8,
            "moderate": 1.0,
            "severe": 1.5,
            "critical": 2.0
        }
        
        mult = multipliers[severity]
        base_repair = 45.00 * mult
        labor = 75.00 * (0.5 + 0.3 * mult)
        
        estimate = {
            "severity": severity,
            "repair_cost": {
                "total": base_repair + labor + 15.00,
                "time_hours": 0.5 + 0.3 * mult
            } if severity != "critical" else None,
            "replacement_cost": {
                "total": 195.50,
                "time_hours": 0.5
            },
            "recommendation": "replace" if severity == "critical" else "repair"
        }
        
        print(f"  {severity.upper()}:")
        if estimate['repair_cost']:
            print(f"    Repair: ${estimate['repair_cost']['total']:.2f} ({estimate['repair_cost']['time_hours']:.1f}h)")
        else:
            print(f"    Repair: Not available")
        print(f"    Replace: ${estimate['replacement_cost']['total']:.2f} ({estimate['replacement_cost']['time_hours']:.1f}h)")
        print(f"    Recommended: {estimate['recommendation'].upper()}")
        print()
    
    return True


# ============================================================================
# Example 5: Batch Assessment
# ============================================================================


async def example_batch_assessment():
    """Assess multiple parts in batch."""
    print("\n" + "="*80)
    print("Example 5: Batch Assessment")
    print("="*80)
    
    # List of parts to assess
    parts_to_assess = [
        ("motor_1.jpg", "motor_servo", "RoboArm-X1"),
        ("wheel_damage.jpg", "wheel_assembly", "MobileBot-Pro"),
        ("camera_crack.jpg", "camera_module", "RoboArm-X1"),
    ]
    
    print(f"\n🔄 Assessing {len(parts_to_assess)} parts...\n")
    
    results_summary = []
    
    for image_path, part_type, robot_model in parts_to_assess:
        print(f"  📸 Assessing {image_path}...")
        
        # Simulated assessment
        result = {
            "image": image_path,
            "part_type": part_type,
            "severity": "moderate",
            "recommendation": "repair",
            "cost": 65.00
        }
        
        results_summary.append(result)
        print(f"     ✅ {part_type}: {result['severity']} → {result['recommendation']} (${result['cost']:.2f})")
    
    # Summary
    total_cost = sum(r['cost'] for r in results_summary)
    print(f"\n📊 Batch Summary:")
    print(f"   Total Parts: {len(results_summary)}")
    print(f"   Total Estimated Cost: ${total_cost:.2f}")
    print(f"   Average Cost per Part: ${total_cost / len(results_summary):.2f}")
    
    return results_summary


# ============================================================================
# Example 6: Real-Time Assessment
# ============================================================================


async def example_realtime_assessment():
    """Simulate real-time assessment workflow."""
    print("\n" + "="*80)
    print("Example 6: Real-Time Assessment Workflow")
    print("="*80)
    
    print("\n🤖 Robot Part Inspection Workflow:\n")
    
    steps = [
        "1️⃣ User uploads image via mobile app",
        "2️⃣ Image preprocessed and enhanced",
        "3️⃣ AI identifies part type (Servo Motor)",
        "4️⃣ Damage detection algorithm runs",
        "5️⃣ Severity assessment completed",
        "6️⃣ Cost calculation performed",
        "7️⃣ Recommendation generated",
        "8️⃣ Results sent to user",
    ]
    
    for step in steps:
        print(f"   {step}")
        await asyncio.sleep(0.5)  # Simulate processing time
    
    print("\n✅ Assessment Complete in 3.2 seconds")
    print("\n📱 User receives instant feedback:")
    print("   • Part: Servo Motor")
    print("   • Damage: Moderate crack")
    print("   • Recommendation: Repair")
    print("   • Cost: $58.75")
    print("   • Time: 0.8 hours")
    print("   • Decision: Proceed with repair ✓")
    
    return True


# ============================================================================
# Main Execution
# ============================================================================


async def run_all_examples():
    """Run all examples."""
    examples = [
        ("Motor Assessment", example_motor_assessment),
        ("Camera Assessment", example_camera_assessment),
        ("List Parts", example_list_parts),
        ("Cost Estimates", example_cost_estimate),
        ("Batch Assessment", example_batch_assessment),
        ("Real-Time Workflow", example_realtime_assessment),
    ]
    
    print("\n" + "="*80)
    print("Image-Based Damage Assessment Examples")
    print("="*80)
    print("\nThese examples demonstrate the damage assessment API capabilities.")
    print(f"API endpoint: {API_BASE_URL}")
    print("\nNote: These examples use simulated data for demonstration.")
    print("In production, upload actual robot part images.\n")
    
    for i, (name, example_func) in enumerate(examples, 1):
        try:
            await example_func()
            await asyncio.sleep(1)
        except Exception as e:
            print(f"\n❌ Example {i} ({name}) failed: {e}")
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
    print("\n💡 Next Steps:")
    print("   1. Open frontend/damage-assessment.html in browser")
    print("   2. Upload actual robot part images")
    print("   3. Review damage assessments and recommendations")
    print("   4. Make informed repair/replacement decisions")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_motor_assessment,
            example_camera_assessment,
            example_list_parts,
            example_cost_estimate,
            example_batch_assessment,
            example_realtime_assessment,
        ]
        
        if 1 <= example_num <= len(examples):
            asyncio.run(examples[example_num - 1]())
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        # Run all examples
        asyncio.run(run_all_examples())
