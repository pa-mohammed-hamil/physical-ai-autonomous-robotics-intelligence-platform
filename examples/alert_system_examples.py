"""
Alert System Examples

Demonstrates how to send robot alerts through multiple channels
for various scenarios: battery issues, component failures, security incidents, etc.
"""

import asyncio
from typing import Dict

import httpx


API_BASE_URL = "http://localhost:8000/api/v1/alerts"
API_KEY = "your-api-key-here"


class AlertClient:
    """Client for alert API."""
    
    def __init__(self, base_url: str = API_BASE_URL, api_key: str = API_KEY):
        """Initialize client."""
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    async def send_alert(
        self,
        alert_type: str,
        robot_id: str,
        severity: str = "warning",
        custom_message: str = None,
        metadata: dict = None,
    ) -> Dict:
        """Send alert."""
        request_data = {
            "alert_type": alert_type,
            "robot_id": robot_id,
            "severity": severity,
            "custom_message": custom_message,
            "metadata": metadata or {},
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/send",
                json=request_data,
                headers=self.headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def add_recipient(
        self,
        name: str,
        email: str = None,
        phone: str = None,
        whatsapp: str = None,
        preferred_channels: list = None,
        severity_threshold: str = "warning",
    ) -> Dict:
        """Add alert recipient."""
        request_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "whatsapp": whatsapp,
            "preferred_channels": preferred_channels or ["email"],
            "severity_threshold": severity_threshold,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/recipients",
                json=request_data,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_history(
        self,
        robot_id: str = None,
        severity: str = None,
        limit: int = 50,
    ) -> list:
        """Get alert history."""
        params = {"limit": limit}
        if robot_id:
            params["robot_id"] = robot_id
        if severity:
            params["severity"] = severity
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/history",
                params=params,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_statistics(self) -> Dict:
        """Get alert statistics."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/statistics",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()


# ============================================================================
# Example 1: Battery Low Alert
# ============================================================================


async def example_battery_low():
    """Send low battery alert."""
    print("\n" + "="*80)
    print("Example 1: Low Battery Alert")
    print("="*80)
    
    client = AlertClient()
    
    result = await client.send_alert(
        alert_type="battery_low",
        robot_id="ROBOT-001",
        severity="warning",
        metadata={"battery_percentage": 18, "estimated_time_remaining": "15 minutes"},
    )
    
    print(f"\n🔋 Alert Sent: {result['alert_id']}")
    print(f"   Type: {result['alert_type']}")
    print(f"   Severity: {result['severity']}")
    print(f"   Notifications Sent: {result['notifications_sent']}")
    print(f"   Notifications Failed: {result['notifications_failed']}")
    
    print("\n📬 Delivery Results:")
    for r in result['results']:
        status = "✅ Delivered" if r['success'] else "❌ Failed"
        print(f"   {r['channel']}: {status} to {r['recipient']}")
    
    return result


# ============================================================================
# Example 2: Component Health Critical
# ============================================================================


async def example_component_failure():
    """Send component health critical alert."""
    print("\n" + "="*80)
    print("Example 2: Component Health Critical (0%)")
    print("="*80)
    
    client = AlertClient()
    
    result = await client.send_alert(
        alert_type="health_critical",
        robot_id="ROBOT-002",
        severity="critical",
        custom_message="Motor servo health has reached 0%. Immediate replacement required.",
        metadata={
            "component": "motor_servo",
            "health_percentage": 0,
            "last_maintenance": "2024-08-15",
        },
    )
    
    print(f"\n⚠️ CRITICAL ALERT: {result['alert_id']}")
    print(f"   Robot: {result['robot_id']}")
    print(f"   Severity: {result['severity'].upper()}")
    print(f"   Notifications: {result['notifications_sent']} sent")
    
    return result


# ============================================================================
# Example 3: VPN/Internet Issue
# ============================================================================


async def example_network_issue():
    """Send network connectivity alert."""
    print("\n" + "="*80)
    print("Example 3: VPN Connection Failed")
    print("="*80)
    
    client = AlertClient()
    
    result = await client.send_alert(
        alert_type="vpn_failure",
        robot_id="ROBOT-003",
        severity="error",
        metadata={
            "vpn_service": "enterprise_vpn",
            "last_connection": "2024-10-15T10:30:00Z",
            "attempts": 5,
        },
    )
    
    print(f"\n🔒 VPN Alert: {result['alert_id']}")
    print(f"   Issue: VPN connection failed")
    print(f"   Action: Remove VPN service or restore connection")
    print(f"   Notifications: {result['notifications_sent']} sent")
    
    return result


# ============================================================================
# Example 4: Unauthorized Access (Security)
# ============================================================================


async def example_security_breach():
    """Send unauthorized access alert."""
    print("\n" + "="*80)
    print("Example 4: SECURITY - Unauthorized Access Attempt")
    print("="*80)
    
    client = AlertClient()
    
    result = await client.send_alert(
        alert_type="unauthorized_access",
        robot_id="ROBOT-001",
        severity="emergency",
        metadata={
            "ip_address": "192.168.1.100",
            "attempted_actions": ["motor_control", "camera_access"],
            "timestamp": "2024-10-16T12:30:45Z",
            "blocked": True,
        },
    )
    
    print(f"\n🚨 SECURITY ALERT: {result['alert_id']}")
    print(f"   Type: Unauthorized Access Attempt")
    print(f"   Severity: EMERGENCY")
    print(f"   Status: Robot is in lockdown mode")
    print(f"   Action Required: Owner verification required")
    print(f"   Notifications: {result['notifications_sent']} sent (ALL CHANNELS)")
    
    return result


# ============================================================================
# Example 5: Memory Warning
# ============================================================================


async def example_memory_warning():
    """Send memory usage warning."""
    print("\n" + "="*80)
    print("Example 5: Memory Usage Warning")
    print("="*80)
    
    client = AlertClient()
    
    result = await client.send_alert(
        alert_type="memory_warning",
        robot_id="ROBOT-004",
        severity="warning",
        metadata={
            "memory_usage_percent": 85,
            "total_memory_gb": 16,
            "used_memory_gb": 13.6,
            "top_processes": ["vision_pipeline", "navigation", "planning"],
        },
    )
    
    print(f"\n💾 Memory Alert: {result['alert_id']}")
    print(f"   Usage: 85%")
    print(f"   Action: Monitor and consider restart")
    print(f"   Notifications: {result['notifications_sent']} sent")
    
    return result


# ============================================================================
# Example 6: Add Recipients
# ============================================================================


async def example_add_recipients():
    """Add alert recipients."""
    print("\n" + "="*80)
    print("Example 6: Configure Alert Recipients")
    print("="*80)
    
    client = AlertClient()
    
    # Add primary contact (all channels)
    result1 = await client.add_recipient(
        name="John Doe (Owner)",
        email="john@example.com",
        phone="+1234567890",
        whatsapp="+1234567890",
        preferred_channels=["email", "sms", "whatsapp", "push"],
        severity_threshold="warning",
    )
    
    print(f"\n✅ Added: {result1['recipient']['name']}")
    print(f"   Channels: {', '.join(result1['recipient']['channels'])}")
    print(f"   Threshold: {result1['recipient']['severity_threshold']}")
    
    # Add maintenance team (email only)
    result2 = await client.add_recipient(
        name="Maintenance Team",
        email="maintenance@company.com",
        preferred_channels=["email"],
        severity_threshold="error",
    )
    
    print(f"\n✅ Added: {result2['recipient']['name']}")
    print(f"   Channels: {', '.join(result2['recipient']['channels'])}")
    print(f"   Threshold: {result2['recipient']['severity_threshold']}")
    
    # Add emergency contact (SMS only, critical alerts)
    result3 = await client.add_recipient(
        name="Emergency Contact",
        phone="+1987654321",
        preferred_channels=["sms"],
        severity_threshold="critical",
    )
    
    print(f"\n✅ Added: {result3['recipient']['name']}")
    print(f"   Channels: {', '.join(result3['recipient']['channels'])}")
    print(f"   Threshold: {result3['recipient']['severity_threshold']}")
    
    return [result1, result2, result3]


# ============================================================================
# Example 7: View Alert History
# ============================================================================


async def example_view_history():
    """View alert history."""
    print("\n" + "="*80)
    print("Example 7: Alert History")
    print("="*80)
    
    client = AlertClient()
    
    # Get all alerts
    all_alerts = await client.get_history(limit=10)
    
    print(f"\n📋 Recent Alerts ({len(all_alerts)} total):")
    for alert in all_alerts:
        print(f"\n   {alert['timestamp']}")
        print(f"   {alert['severity'].upper()}: {alert['title']}")
        print(f"   Robot: {alert['robot_id']}")
        print(f"   Alert ID: {alert['alert_id']}")
    
    # Get critical alerts only
    critical_alerts = await client.get_history(severity="critical", limit=5)
    
    print(f"\n🔴 Critical Alerts ({len(critical_alerts)} total):")
    for alert in critical_alerts:
        print(f"   - {alert['title']} ({alert['robot_id']})")
    
    return all_alerts


# ============================================================================
# Example 8: Alert Statistics
# ============================================================================


async def example_statistics():
    """View alert statistics."""
    print("\n" + "="*80)
    print("Example 8: Alert Statistics")
    print("="*80)
    
    client = AlertClient()
    
    stats = await client.get_statistics()
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Alerts: {stats['total_alerts']}")
    
    print(f"\n📈 By Severity:")
    for severity, count in stats['by_severity'].items():
        print(f"   {severity}: {count}")
    
    print(f"\n🏷️ By Type:")
    for alert_type, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {alert_type}: {count}")
    
    print(f"\n🤖 By Robot:")
    for robot_id, count in stats['by_robot'].items():
        print(f"   {robot_id}: {count} alerts")
    
    return stats


# ============================================================================
# Example 9: Multi-Alert Scenario
# ============================================================================


async def example_multi_alert_scenario():
    """Simulate multiple alerts in sequence."""
    print("\n" + "="*80)
    print("Example 9: Multi-Alert Scenario")
    print("="*80)
    
    client = AlertClient()
    
    print("\n🎬 Simulating robot operational scenario...")
    
    # 1. Battery low
    print("\n1️⃣ Battery getting low...")
    await client.send_alert(
        alert_type="battery_low",
        robot_id="ROBOT-005",
        severity="warning",
    )
    await asyncio.sleep(1)
    
    # 2. Network issue
    print("2️⃣ Network connectivity problem...")
    await client.send_alert(
        alert_type="network_issue",
        robot_id="ROBOT-005",
        severity="warning",
    )
    await asyncio.sleep(1)
    
    # 3. Battery critical
    print("3️⃣ Battery now critical!")
    await client.send_alert(
        alert_type="battery_critical",
        robot_id="ROBOT-005",
        severity="critical",
    )
    await asyncio.sleep(1)
    
    # 4. Emergency stop
    print("4️⃣ Emergency stop activated")
    result = await client.send_alert(
        alert_type="emergency_stop",
        robot_id="ROBOT-005",
        severity="emergency",
    )
    
    print(f"\n✅ Scenario complete: {result['notifications_sent']} notifications sent")
    
    return result


# ============================================================================
# Main Execution
# ============================================================================


async def run_all_examples():
    """Run all examples."""
    examples = [
        ("Battery Low Alert", example_battery_low),
        ("Component Health Critical", example_component_failure),
        ("VPN/Network Issue", example_network_issue),
        ("Security Breach", example_security_breach),
        ("Memory Warning", example_memory_warning),
        ("Add Recipients", example_add_recipients),
        ("View History", example_view_history),
        ("Statistics", example_statistics),
        ("Multi-Alert Scenario", example_multi_alert_scenario),
    ]
    
    print("\n" + "="*80)
    print("Robot Alert System Examples")
    print("="*80)
    print("\nDemonstrates multi-channel alerts for various robot issues:")
    print("- Battery warnings")
    print("- Component failures")
    print("- Network issues")
    print("- Security incidents")
    print("- System health")
    print(f"\nAPI: {API_BASE_URL}\n")
    
    for i, (name, example_func) in enumerate(examples, 1):
        try:
            await example_func()
            await asyncio.sleep(1)
        except Exception as e:
            print(f"\n❌ Example {i} ({name}) failed: {e}")
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Multi-channel notifications (Email, SMS, WhatsApp, Push)")
    print("   ✅ Severity-based routing")
    print("   ✅ Multiple alert types")
    print("   ✅ Recipient management")
    print("   ✅ Alert history and statistics")
    print("   ✅ Security incident handling")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_battery_low,
            example_component_failure,
            example_network_issue,
            example_security_breach,
            example_memory_warning,
            example_add_recipients,
            example_view_history,
            example_statistics,
            example_multi_alert_scenario,
        ]
        
        if 1 <= example_num <= len(examples):
            asyncio.run(examples[example_num - 1]())
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        asyncio.run(run_all_examples())
