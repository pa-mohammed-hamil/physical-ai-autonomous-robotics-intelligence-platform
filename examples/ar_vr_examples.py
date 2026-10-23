"""
AR/VR Integration Examples
Demonstrates various AR/VR use cases for immersive robot monitoring
"""

import asyncio
import json
from datetime import datetime
from app.vr.ar_vr_integration import (
    ar_vr_manager,
    DeviceType,
    ViewMode,
    InteractionMode
)


async def example_1_create_vr_session():
    """Example 1: Create VR session with headset"""
    print("\n=== Example 1: Create VR Session ===")
    
    # Create VR session
    session = await ar_vr_manager.create_session(
        user_id="operator_001",
        device_type=DeviceType.VR_HEADSET,
        view_mode=ViewMode.THIRD_PERSON,
        interaction_mode=InteractionMode.CONTROLLER
    )
    
    print(f"✓ VR Session created: {session.session_id}")
    print(f"  Device: {session.device_type.value}")
    print(f"  View Mode: {session.view_mode.value}")
    print(f"  Interaction: {session.interaction_mode.value}")
    print(f"  Frame Rate: {session.frame_rate} FPS")
    
    return session.session_id


async def example_2_create_ar_phone_session():
    """Example 2: Create AR session on mobile phone"""
    print("\n=== Example 2: Create AR Phone Session ===")
    
    # Create AR phone session
    session = await ar_vr_manager.create_session(
        user_id="technician_mobile_001",
        device_type=DeviceType.AR_PHONE,
        view_mode=ViewMode.SPATIAL_MAP,
        interaction_mode=InteractionMode.TOUCH
    )
    
    print(f"✓ AR Phone Session created: {session.session_id}")
    print(f"  Device: ARCore/ARKit Mobile")
    print(f"  View Mode: Spatial Mapping")
    print(f"  Interaction: Touch Gestures")
    
    return session.session_id


async def example_3_connect_multiple_robots():
    """Example 3: Connect multiple robots to VR session"""
    print("\n=== Example 3: Connect Multiple Robots ===")
    
    # Create session
    session = await ar_vr_manager.create_session(
        user_id="supervisor_001",
        device_type=DeviceType.VR_HEADSET,
        view_mode=ViewMode.GOD_MODE
    )
    
    # Connect multiple robots
    robot_ids = ["ROBOT_001", "ROBOT_002", "ROBOT_003", "ROBOT_004"]
    
    for robot_id in robot_ids:
        success = await ar_vr_manager.connect_robot(session.session_id, robot_id)
        print(f"✓ Connected {robot_id}: {success}")
    
    print(f"\nTotal robots connected: {len(session.connected_robots)}")
    print(f"Robots: {', '.join(session.connected_robots)}")
    
    return session.session_id


async def example_4_spatial_anchors():
    """Example 4: Create spatial anchors in AR space"""
    print("\n=== Example 4: Create Spatial Anchors ===")
    
    # Create AR session
    session = await ar_vr_manager.create_session(
        user_id="field_engineer_001",
        device_type=DeviceType.AR_GLASSES,
        view_mode=ViewMode.HOLOGRAPHIC
    )
    
    # Create spatial anchors for different purposes
    anchors = [
        {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "content_type": "robot",
            "content_data": {"robot_id": "ROBOT_001", "label": "Primary Robot"},
            "persistent": True
        },
        {
            "position": {"x": 5.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "content_type": "alert",
            "content_data": {"severity": "warning", "message": "Maintenance Required"},
            "persistent": True
        },
        {
            "position": {"x": 0.0, "y": 0.0, "z": 5.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "content_type": "metric",
            "content_data": {"metric": "efficiency", "value": 87.5},
            "persistent": False
        }
    ]
    
    for i, anchor_data in enumerate(anchors, 1):
        anchor = await ar_vr_manager.create_spatial_anchor(
            session_id=session.session_id,
            **anchor_data
        )
        print(f"✓ Anchor {i}: {anchor.content_type} at ({anchor.position['x']}, {anchor.position['y']}, {anchor.position['z']})")
    
    return session.session_id


async def example_5_holographic_overlays():
    """Example 5: Create holographic overlays for robots"""
    print("\n=== Example 5: Create Holographic Overlays ===")
    
    # Create AR session
    session = await ar_vr_manager.create_session(
        user_id="technician_ar_001",
        device_type=DeviceType.AR_GLASSES,
        view_mode=ViewMode.HOLOGRAPHIC
    )
    
    # Connect robot
    await ar_vr_manager.connect_robot(session.session_id, "ROBOT_001")
    
    # Create different overlay types
    overlays = [
        {
            "overlay_type": "health",
            "position": {"x": 0.0, "y": 2.0, "z": 0.0},
            "data": {
                "health_percentage": 92,
                "battery_level": 78,
                "temperature": 45.2,
                "status": "operational"
            }
        },
        {
            "overlay_type": "telemetry",
            "position": {"x": 1.0, "y": 1.5, "z": 0.0},
            "data": {
                "speed": 0.5,
                "direction": 45,
                "distance_traveled": 1250.3,
                "uptime": "24h 15m"
            }
        },
        {
            "overlay_type": "path",
            "position": {"x": 0.0, "y": 0.1, "z": 0.0},
            "data": {
                "waypoints": [
                    {"x": 0, "y": 0, "z": 0},
                    {"x": 5, "y": 0, "z": 5},
                    {"x": 10, "y": 0, "z": 0}
                ],
                "current_waypoint": 1
            }
        },
        {
            "overlay_type": "alert",
            "position": {"x": 0.0, "y": 2.5, "z": 0.0},
            "data": {
                "alert_type": "maintenance",
                "severity": "warning",
                "message": "Oil change recommended in 50 hours"
            }
        }
    ]
    
    for overlay_data in overlays:
        overlay = await ar_vr_manager.create_holographic_overlay(
            session_id=session.session_id,
            robot_id="ROBOT_001",
            **overlay_data
        )
        print(f"✓ Overlay: {overlay.overlay_type} at position ({overlay.position['x']}, {overlay.position['y']}, {overlay.position['z']})")
    
    return session.session_id


async def example_6_hand_gesture_control():
    """Example 6: Hand gesture control"""
    print("\n=== Example 6: Hand Gesture Control ===")
    
    # Create VR session with hand tracking
    session = await ar_vr_manager.create_session(
        user_id="operator_vr_001",
        device_type=DeviceType.VR_HEADSET,
        interaction_mode=InteractionMode.HAND_TRACKING
    )
    
    # Simulate hand gestures
    gestures = [
        {"gesture_type": "point", "hand": "right", "position": {"x": 1.0, "y": 1.5, "z": -2.0}},
        {"gesture_type": "grab", "hand": "right", "position": {"x": 0.5, "y": 1.2, "z": -1.5}},
        {"gesture_type": "pinch", "hand": "left", "position": {"x": -0.5, "y": 1.3, "z": -1.8}},
        {"gesture_type": "swipe", "hand": "right", "position": {"x": 2.0, "y": 1.4, "z": -2.5}},
        {"gesture_type": "wave", "hand": "both", "position": {"x": 0.0, "y": 1.6, "z": -2.0}}
    ]
    
    for gesture_data in gestures:
        gesture = await ar_vr_manager.process_hand_gesture(
            session_id=session.session_id,
            confidence=0.95,
            **gesture_data
        )
        print(f"✓ Gesture: {gesture.gesture_type} ({gesture.hand} hand) - Confidence: {gesture.confidence:.0%}")
    
    return session.session_id


async def example_7_haptic_feedback():
    """Example 7: Haptic feedback patterns"""
    print("\n=== Example 7: Haptic Feedback ===")
    
    # Create VR session
    session = await ar_vr_manager.create_session(
        user_id="operator_haptic_001",
        device_type=DeviceType.VR_HEADSET,
        interaction_mode=InteractionMode.CONTROLLER
    )
    
    # Trigger different haptic feedback patterns
    feedback_patterns = [
        {"pattern": "light_tap", "intensity": 0.3, "duration_ms": 50},
        {"pattern": "medium_tap", "intensity": 0.5, "duration_ms": 100},
        {"pattern": "heavy_tap", "intensity": 0.8, "duration_ms": 150},
        {"pattern": "pulse", "intensity": 0.6, "duration_ms": 200},
        {"pattern": "alert", "intensity": 1.0, "duration_ms": 300}
    ]
    
    for feedback in feedback_patterns:
        success = await ar_vr_manager.trigger_haptic_feedback(
            session_id=session.session_id,
            **feedback
        )
        print(f"✓ Haptic: {feedback['pattern']} (intensity: {feedback['intensity']}, duration: {feedback['duration_ms']}ms)")
    
    # Get feedback queue
    queued_feedback = await ar_vr_manager.get_haptic_feedback(session.session_id)
    print(f"\nQueued feedback events: {len(queued_feedback)}")
    
    return session.session_id


async def example_8_view_mode_switching():
    """Example 8: Switch between view modes"""
    print("\n=== Example 8: View Mode Switching ===")
    
    # Create session
    session = await ar_vr_manager.create_session(
        user_id="operator_multiview_001",
        device_type=DeviceType.VR_HEADSET
    )
    
    # Connect robot
    await ar_vr_manager.connect_robot(session.session_id, "ROBOT_001")
    
    # Switch through different view modes
    view_modes = [
        ViewMode.FIRST_PERSON,
        ViewMode.THIRD_PERSON,
        ViewMode.GOD_MODE,
        ViewMode.SPLIT_VIEW,
        ViewMode.TWIN_VIEW
    ]
    
    for view_mode in view_modes:
        success = await ar_vr_manager.update_view_mode(session.session_id, view_mode)
        print(f"✓ Switched to: {view_mode.value.replace('_', ' ').title()}")
        await asyncio.sleep(0.5)  # Simulate time between switches
    
    return session.session_id


async def example_9_vr_environments():
    """Example 9: VR environment selection"""
    print("\n=== Example 9: VR Environments ===")
    
    # List all environments
    environments = ar_vr_manager.list_vr_environments()
    
    print(f"Available VR Environments: {len(environments)}\n")
    
    for env in environments:
        print(f"Environment: {env.environment_id}")
        print(f"  Scene Type: {env.scene_type}")
        print(f"  Lighting: {env.lighting}")
        print(f"  Weather: {env.weather or 'N/A'}")
        print(f"  Skybox: {env.skybox}")
        print(f"  Scale Factor: {env.scale_factor}x")
        print()
    
    # Get specific environment
    warehouse_env = ar_vr_manager.get_vr_environment("warehouse_day")
    if warehouse_env:
        print(f"Selected: {warehouse_env.environment_id}")
        print(f"  Scene: {warehouse_env.scene_type}")
        print(f"  Ground Plane: {warehouse_env.ground_plane}")
        print(f"  Ambient Sound: {warehouse_env.ambient_sound}")


async def example_10_performance_monitoring():
    """Example 10: Performance monitoring and optimization"""
    print("\n=== Example 10: Performance Monitoring ===")
    
    # Create session
    session = await ar_vr_manager.create_session(
        user_id="operator_perf_001",
        device_type=DeviceType.VR_HEADSET
    )
    
    # Simulate performance metrics over time
    print("Simulating performance metrics...\n")
    
    for i in range(5):
        frame_time = 11.1 + (i * 2)  # Simulating degrading performance
        latency = 20 + (i * 5)
        
        await ar_vr_manager.update_performance_metrics(
            session_id=session.session_id,
            frame_time_ms=frame_time,
            latency_ms=latency
        )
        
        stats = await ar_vr_manager.get_performance_stats(session.session_id)
        
        print(f"Frame {i+1}:")
        print(f"  Target FPS: {stats['target_fps']}")
        print(f"  Avg Frame Time: {stats['avg_frame_time_ms']:.2f}ms")
        print(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"  Quality Level: {stats['quality_level']}")
        print()
        
        await asyncio.sleep(0.1)
    
    return session.session_id


async def example_11_multi_robot_god_mode():
    """Example 11: God mode with multiple robots"""
    print("\n=== Example 11: God Mode Fleet Monitoring ===")
    
    # Create god mode session
    session = await ar_vr_manager.create_session(
        user_id="fleet_manager_001",
        device_type=DeviceType.VR_HEADSET,
        view_mode=ViewMode.GOD_MODE
    )
    
    # Connect entire fleet
    fleet_robots = [f"ROBOT_{i:03d}" for i in range(1, 11)]
    
    print(f"Connecting {len(fleet_robots)} robots to fleet view...")
    
    for robot_id in fleet_robots:
        await ar_vr_manager.connect_robot(session.session_id, robot_id)
    
    # Create spatial anchor for each robot
    for i, robot_id in enumerate(fleet_robots):
        x = (i % 5) * 3.0
        z = (i // 5) * 3.0
        
        await ar_vr_manager.create_spatial_anchor(
            session_id=session.session_id,
            position={"x": x, "y": 0, "z": z},
            rotation={"w": 1, "x": 0, "y": 0, "z": 0},
            content_type="robot",
            content_data={"robot_id": robot_id, "status": "operational"},
            persistent=True
        )
    
    print(f"✓ Fleet view configured with {len(fleet_robots)} robots")
    print(f"✓ Spatial anchors created")
    print(f"✓ God mode active - Full fleet visibility")
    
    # Get render data
    render_data = await ar_vr_manager.get_session_render_data(session.session_id)
    print(f"\nRender Data:")
    print(f"  Connected Robots: {len(render_data['connected_robots'])}")
    print(f"  Spatial Anchors: {len(render_data['spatial_anchors'])}")
    print(f"  View Mode: {render_data['view_mode']}")
    
    return session.session_id


async def example_12_ar_maintenance_workflow():
    """Example 12: AR-guided maintenance workflow"""
    print("\n=== Example 12: AR-Guided Maintenance ===")
    
    # Create AR glasses session
    session = await ar_vr_manager.create_session(
        user_id="maintenance_tech_001",
        device_type=DeviceType.AR_GLASSES,
        view_mode=ViewMode.HOLOGRAPHIC,
        interaction_mode=InteractionMode.VOICE
    )
    
    # Connect robot for maintenance
    robot_id = "ROBOT_001"
    await ar_vr_manager.connect_robot(session.session_id, robot_id)
    
    print(f"✓ Maintenance session started for {robot_id}")
    
    # Create maintenance-specific overlays
    maintenance_steps = [
        {
            "overlay_type": "instruction",
            "position": {"x": 0, "y": 2, "z": 0},
            "data": {
                "step": 1,
                "title": "Inspect Drive Motors",
                "instructions": "Check for unusual wear or damage",
                "status": "pending"
            }
        },
        {
            "overlay_type": "instruction",
            "position": {"x": 0, "y": 1.8, "z": 0},
            "data": {
                "step": 2,
                "title": "Lubricate Joints",
                "instructions": "Apply lubricant to all joint mechanisms",
                "status": "pending"
            }
        },
        {
            "overlay_type": "diagnostic",
            "position": {"x": 1, "y": 1.5, "z": 0},
            "data": {
                "component": "drive_motor",
                "health": 87,
                "temperature": 45.2,
                "vibration": "normal",
                "recommendation": "Continue monitoring"
            }
        }
    ]
    
    for step in maintenance_steps:
        overlay = await ar_vr_manager.create_holographic_overlay(
            session_id=session.session_id,
            robot_id=robot_id,
            **step
        )
        print(f"✓ Created overlay: {step['overlay_type']} - {step['data'].get('title', 'Diagnostic Info')}")
    
    # Create spatial markers for components
    component_markers = [
        {"position": {"x": 0.5, "y": 0.5, "z": 0}, "component": "Left Drive Motor"},
        {"position": {"x": -0.5, "y": 0.5, "z": 0}, "component": "Right Drive Motor"},
        {"position": {"x": 0, "y": 1.0, "z": 0.3}, "component": "Main Controller"},
    ]
    
    for marker in component_markers:
        await ar_vr_manager.create_spatial_anchor(
            session_id=session.session_id,
            position=marker["position"],
            rotation={"w": 1, "x": 0, "y": 0, "z": 0},
            content_type="component_marker",
            content_data={"component_name": marker["component"]},
            persistent=False
        )
        print(f"✓ Component marker: {marker['component']}")
    
    print("\n✓ AR maintenance workflow ready")
    print("  Voice commands enabled")
    print("  Holographic instructions displayed")
    print("  Component markers placed")
    
    return session.session_id


async def main():
    """Run all examples"""
    print("=" * 60)
    print("AR/VR Integration Examples")
    print("Physical AI Autonomous Robotics Intelligence Platform")
    print("=" * 60)
    
    examples = [
        example_1_create_vr_session,
        example_2_create_ar_phone_session,
        example_3_connect_multiple_robots,
        example_4_spatial_anchors,
        example_5_holographic_overlays,
        example_6_hand_gesture_control,
        example_7_haptic_feedback,
        example_8_view_mode_switching,
        example_9_vr_environments,
        example_10_performance_monitoring,
        example_11_multi_robot_god_mode,
        example_12_ar_maintenance_workflow
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            await example()
            print(f"\n{'='*60}\n")
            
            # Small delay between examples
            if i < len(examples):
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error in example {i}: {e}")
    
    # Cleanup
    print("\n=== Cleanup ===")
    active_sessions = ar_vr_manager.get_all_active_sessions()
    for session in active_sessions:
        await ar_vr_manager.close_session(session.session_id)
        print(f"✓ Closed session: {session.session_id}")
    
    print("\n✓ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
