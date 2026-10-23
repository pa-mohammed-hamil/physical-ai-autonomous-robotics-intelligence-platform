# AR/VR Integration Guide

## Overview

The AR/VR Integration system provides immersive monitoring experiences for the Physical AI Autonomous Robotics Intelligence Platform. It enables operators, technicians, and supervisors to interact with robots in virtual reality, augmented reality, and mixed reality environments.

## Features

### 🥽 Device Support

- **VR Headsets**: Meta Quest, Valve Index, HTC Vive, PlayStation VR
- **AR Phones**: ARCore (Android), ARKit (iOS)
- **AR Glasses**: Microsoft HoloLens, Magic Leap, Nreal
- **Mixed Reality**: Windows Mixed Reality devices
- **WebXR**: Browser-based VR/AR (Chrome, Edge, Firefox Reality)

### 👁️ View Modes

1. **First Person** - Experience from robot's perspective
2. **Third Person** - External view of robot
3. **God Mode** - Overview of entire robot fleet
4. **Split View** - Multiple robots simultaneously
5. **Spatial Map** - AR spatial mapping overlay
6. **Holographic** - AR holographic projections
7. **Twin View** - Digital twin overlay

### 🎮 Interaction Methods

- **Hand Tracking** - Natural hand gestures
- **Controller** - VR controller input
- **Gaze** - Eye tracking and gaze control
- **Voice** - Voice commands
- **Gesture** - Body gestures
- **Touch** - Touchscreen (mobile AR)

### ✨ Advanced Features

#### Spatial Anchors
- Persistent AR content placement
- Waypoint markers
- Alert indicators
- Metric displays
- Component labels

#### Holographic Overlays
- Real-time health metrics
- Telemetry data visualization
- Path projection
- Alert notifications
- Diagnostic information

#### Haptic Feedback
- Light tap (notifications)
- Medium tap (selections)
- Heavy tap (warnings)
- Pulse (ongoing process)
- Alert (urgent attention)

#### Performance Optimization
- Adaptive quality levels (low, medium, high, ultra)
- Frame rate targeting (90 FPS for VR, 60 FPS for AR)
- Latency monitoring
- Automatic quality adjustment

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AR/VR Frontend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   WebXR      │  │   Three.js   │  │   WebSocket  │     │
│  │   Browser    │  │   Renderer   │  │   Streaming  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ WebSocket (30 FPS)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              AR/VR Integration Manager                │   │
│  │  • Session Management                                 │   │
│  │  • Spatial Anchor Tracking                           │   │
│  │  • Holographic Overlay System                        │   │
│  │  • Hand Gesture Recognition                          │   │
│  │  • Haptic Feedback Queue                             │   │
│  │  • Performance Monitoring                            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Robot Control Layer                        │
│  • Real-time telemetry streaming                            │
│  • Command execution                                         │
│  • State synchronization                                     │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### Session Management

#### Create Session
```http
POST /api/v1/ar-vr/session
Content-Type: application/json

{
  "user_id": "operator_001",
  "device_type": "vr_headset",
  "view_mode": "third_person",
  "interaction_mode": "controller"
}
```

**Response:**
```json
{
  "session_id": "arvr_operator_001_1234567890",
  "user_id": "operator_001",
  "device_type": "vr_headset",
  "view_mode": "third_person",
  "interaction_mode": "controller",
  "active": true,
  "start_time": "2024-01-15T10:30:00Z",
  "websocket_url": "ws://localhost:8000/api/v1/ar-vr/session/arvr_operator_001_1234567890/stream"
}
```

#### Get Session
```http
GET /api/v1/ar-vr/session/{session_id}
```

#### List Sessions
```http
GET /api/v1/ar-vr/sessions?user_id=operator_001&active_only=true
```

#### Close Session
```http
DELETE /api/v1/ar-vr/session/{session_id}
```

### Robot Connection

#### Connect Robot
```http
POST /api/v1/ar-vr/session/{session_id}/robot
Content-Type: application/json

{
  "robot_id": "ROBOT_001"
}
```

#### Disconnect Robot
```http
DELETE /api/v1/ar-vr/session/{session_id}/robot/{robot_id}
```

### View Control

#### Update View Mode
```http
PUT /api/v1/ar-vr/session/{session_id}/view-mode
Content-Type: application/json

{
  "view_mode": "god_mode"
}
```

### Spatial Anchors

#### Create Spatial Anchor
```http
POST /api/v1/ar-vr/session/{session_id}/spatial-anchor
Content-Type: application/json

{
  "position": {"x": 5.0, "y": 0.0, "z": 3.0},
  "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
  "content_type": "alert",
  "content_data": {
    "severity": "warning",
    "message": "Low battery"
  },
  "persistent": true
}
```

#### Get Spatial Anchors
```http
GET /api/v1/ar-vr/session/{session_id}/spatial-anchors
```

### Holographic Overlays

#### Create Holographic Overlay
```http
POST /api/v1/ar-vr/session/{session_id}/holographic-overlay
Content-Type: application/json

{
  "robot_id": "ROBOT_001",
  "overlay_type": "health",
  "position": {"x": 0.0, "y": 2.0, "z": 0.0},
  "data": {
    "health_percentage": 92,
    "battery_level": 78,
    "temperature": 45.2
  },
  "anchor_to_robot": true
}
```

**Overlay Types:**
- `health` - Health and status metrics
- `telemetry` - Movement and sensor data
- `path` - Navigation path projection
- `alert` - Alert notifications
- `diagnostic` - Diagnostic information
- `instruction` - Maintenance instructions

#### Get Holographic Overlays
```http
GET /api/v1/ar-vr/session/{session_id}/holographic-overlays
```

### User Interaction

#### Update User Pose
```http
PUT /api/v1/ar-vr/session/{session_id}/user-pose
Content-Type: application/json

{
  "position": {"x": 0.0, "y": 1.6, "z": 3.0},
  "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
}
```

#### Process Hand Gesture
```http
POST /api/v1/ar-vr/session/{session_id}/gesture
Content-Type: application/json

{
  "gesture_type": "point",
  "hand": "right",
  "position": {"x": 1.0, "y": 1.5, "z": -2.0},
  "confidence": 0.95
}
```

**Gesture Types:**
- `point` - Point to select
- `grab` - Grab to move
- `pinch` - Pinch to scale
- `swipe` - Swipe to navigate
- `wave` - Wave to activate
- `thumbs_up` - Confirm action
- `thumbs_down` - Cancel action

### Haptic Feedback

#### Trigger Haptic Feedback
```http
POST /api/v1/ar-vr/session/{session_id}/haptic-feedback
Content-Type: application/json

{
  "pattern": "medium_tap",
  "intensity": 0.7,
  "duration_ms": 150
}
```

**Haptic Patterns:**
- `light_tap` - Subtle notification (50ms)
- `medium_tap` - Standard feedback (100ms)
- `heavy_tap` - Strong confirmation (150ms)
- `pulse` - Rhythmic feedback (200ms)
- `alert` - Urgent attention (300ms)

#### Get Haptic Feedback Queue
```http
GET /api/v1/ar-vr/session/{session_id}/haptic-feedback
```

### VR Environments

#### List Environments
```http
GET /api/v1/ar-vr/environments
```

**Response:**
```json
{
  "environments": [
    {
      "environment_id": "warehouse_day",
      "scene_type": "warehouse",
      "lighting": "day",
      "weather": null,
      "ground_plane": true,
      "skybox": "industrial",
      "ambient_sound": true,
      "scale_factor": 1.0
    }
  ],
  "total": 4
}
```

#### Get Environment
```http
GET /api/v1/ar-vr/environments/{environment_id}
```

### Performance Monitoring

#### Update Performance Metrics
```http
POST /api/v1/ar-vr/session/{session_id}/performance
Content-Type: application/json

{
  "frame_time_ms": 11.1,
  "latency_ms": 25.0
}
```

#### Get Performance Stats
```http
GET /api/v1/ar-vr/session/{session_id}/performance
```

**Response:**
```json
{
  "session_id": "arvr_operator_001_1234567890",
  "current_fps": 90,
  "target_fps": 90,
  "avg_frame_time_ms": 11.1,
  "min_frame_time_ms": 9.5,
  "max_frame_time_ms": 13.2,
  "avg_latency_ms": 25.0,
  "quality_level": "high",
  "connected_robots": 3,
  "spatial_anchors": 5,
  "overlays": 8
}
```

### WebSocket Streaming

#### Connect to Stream
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ar-vr/session/{session_id}/stream');

ws.onopen = () => {
  console.log('Connected to AR/VR stream');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'init') {
    // Initial session data
    handleInitData(message.data);
  } else if (message.type === 'update') {
    // Real-time updates
    handleUpdate(message.render_data);
    handleHapticFeedback(message.haptic_feedback);
  }
};
```

#### Send Messages
```javascript
// Update user pose
ws.send(JSON.stringify({
  type: 'pose_update',
  position: {x: 0, y: 1.6, z: 3},
  rotation: {w: 1, x: 0, y: 0, z: 0}
}));

// Send gesture
ws.send(JSON.stringify({
  type: 'gesture',
  gesture_type: 'point',
  hand: 'right',
  position: {x: 1, y: 1.5, z: -2},
  confidence: 0.95
}));

// Send performance metrics
ws.send(JSON.stringify({
  type: 'performance',
  frame_time_ms: 11.1,
  latency_ms: 25.0
}));
```

## Use Cases

### 1. Fleet Monitoring (God Mode)

**Scenario**: Supervisor monitors entire robot fleet from VR control room

```python
# Create god mode session
session = await ar_vr_manager.create_session(
    user_id="supervisor_001",
    device_type=DeviceType.VR_HEADSET,
    view_mode=ViewMode.GOD_MODE
)

# Connect all robots
for robot_id in robot_fleet:
    await ar_vr_manager.connect_robot(session.session_id, robot_id)

# Create spatial anchors for fleet layout
for robot in robots:
    await ar_vr_manager.create_spatial_anchor(
        session_id=session.session_id,
        position=robot.position,
        content_type="robot",
        content_data={"robot_id": robot.id, "status": robot.status}
    )
```

**Benefits:**
- Simultaneous monitoring of 10+ robots
- Real-time fleet status visualization
- Spatial awareness of robot positions
- Quick identification of issues

### 2. AR-Guided Maintenance

**Scenario**: Technician performs maintenance with AR glasses

```python
# Create AR maintenance session
session = await ar_vr_manager.create_session(
    user_id="tech_001",
    device_type=DeviceType.AR_GLASSES,
    view_mode=ViewMode.HOLOGRAPHIC,
    interaction_mode=InteractionMode.VOICE
)

# Connect robot
await ar_vr_manager.connect_robot(session.session_id, "ROBOT_001")

# Create step-by-step instructions
steps = [
    "Inspect drive motors",
    "Lubricate joints",
    "Check battery connections",
    "Test sensors"
]

for i, step in enumerate(steps):
    await ar_vr_manager.create_holographic_overlay(
        session_id=session.session_id,
        robot_id="ROBOT_001",
        overlay_type="instruction",
        position={"x": 0, "y": 2 - (i * 0.2), "z": 0},
        data={"step": i+1, "instruction": step}
    )
```

**Benefits:**
- Hands-free operation
- Step-by-step visual guidance
- Component highlighting
- Real-time diagnostic data

### 3. Remote Telepresence

**Scenario**: Operator controls robot remotely through VR

```python
# Create first-person VR session
session = await ar_vr_manager.create_session(
    user_id="operator_001",
    device_type=DeviceType.VR_HEADSET,
    view_mode=ViewMode.FIRST_PERSON,
    interaction_mode=InteractionMode.HAND_TRACKING
)

# Connect robot
await ar_vr_manager.connect_robot(session.session_id, "ROBOT_001")

# Process hand gestures for control
gesture = await ar_vr_manager.process_hand_gesture(
    session_id=session.session_id,
    gesture_type="point",
    hand="right",
    position=target_position
)

# Provide haptic feedback
await ar_vr_manager.trigger_haptic_feedback(
    session_id=session.session_id,
    pattern="light_tap",
    intensity=0.5
)
```

**Benefits:**
- Immersive robot control
- Natural hand gesture interface
- Haptic feedback for touch sensation
- First-person perspective

### 4. Training Simulation

**Scenario**: New operators train in safe VR environment

```python
# Create training session
session = await ar_vr_manager.create_session(
    user_id="trainee_001",
    device_type=DeviceType.VR_HEADSET,
    view_mode=ViewMode.THIRD_PERSON
)

# Load training environment
environment = ar_vr_manager.get_vr_environment("warehouse_day")

# Connect training robots
training_robots = ["SIM_ROBOT_001", "SIM_ROBOT_002"]
for robot_id in training_robots:
    await ar_vr_manager.connect_robot(session.session_id, robot_id)

# Create training objectives
objectives = [
    {"task": "Navigate to waypoint", "location": {"x": 10, "y": 0, "z": 5}},
    {"task": "Pick up object", "object": "box_001"},
    {"task": "Return to start", "location": {"x": 0, "y": 0, "z": 0}}
]
```

**Benefits:**
- Risk-free training
- Repeatable scenarios
- Performance tracking
- Immediate feedback

### 5. Collaborative Debugging

**Scenario**: Multiple experts collaborate in shared VR space

```python
# Create sessions for multiple experts
expert_sessions = []

for expert_id in ["expert_001", "expert_002", "expert_003"]:
    session = await ar_vr_manager.create_session(
        user_id=expert_id,
        device_type=DeviceType.VR_HEADSET,
        view_mode=ViewMode.SPLIT_VIEW
    )
    
    # Connect same robot
    await ar_vr_manager.connect_robot(session.session_id, "ROBOT_FAULT_001")
    
    # Create shared spatial anchors
    await ar_vr_manager.create_spatial_anchor(
        session_id=session.session_id,
        position={"x": 0, "y": 2, "z": 0},
        content_type="shared_whiteboard",
        content_data={"session_id": session.session_id}
    )
    
    expert_sessions.append(session)
```

**Benefits:**
- Real-time collaboration
- Shared visual context
- Multiple perspectives
- Expert knowledge pooling

## Best Practices

### Performance Optimization

1. **Target Frame Rates**
   - VR: 90 FPS minimum (Quest 2: 90/120 Hz)
   - AR Phone: 60 FPS
   - AR Glasses: 60 FPS
   - WebXR: 60-90 FPS

2. **Quality Levels**
   - **Ultra**: Full resolution, maximum effects, 4K textures
   - **High**: High resolution, standard effects, 2K textures
   - **Medium**: Medium resolution, reduced effects, 1K textures
   - **Low**: Low resolution, minimal effects, 512px textures

3. **Adaptive Quality**
   ```python
   # System automatically adjusts quality based on performance
   if avg_frame_time > target_frame_time * 1.5:
       # Reduce quality level
       session.quality_level = downgrade_quality(session.quality_level)
   elif avg_frame_time < target_frame_time * 0.8:
       # Increase quality level
       session.quality_level = upgrade_quality(session.quality_level)
   ```

### Latency Management

1. **Target Latency**: <20ms motion-to-photon
2. **Network Optimization**: WebSocket with binary protocol
3. **Prediction**: Client-side motion prediction
4. **Buffering**: Minimal buffering for real-time control

### User Comfort

1. **Motion Sickness Prevention**
   - Maintain consistent frame rate
   - Smooth camera movements
   - Fixed reference points
   - Gradual transitions

2. **Eye Strain Reduction**
   - Proper IPD (Inter-Pupillary Distance) settings
   - Comfortable brightness levels
   - High-contrast UI elements
   - Regular break reminders

3. **Ergonomics**
   - Comfortable viewing distances
   - Natural hand positions
   - Clear gesture recognition
   - Audio feedback for actions

### Security

1. **Session Management**
   - Unique session IDs
   - Automatic timeout (30 minutes idle)
   - Secure WebSocket (WSS)
   - User authentication required

2. **Data Privacy**
   - Encrypted communication
   - No persistent user tracking
   - Spatial data anonymization
   - GDPR compliance

3. **Access Control**
   - Role-based permissions
   - Robot access restrictions
   - Environment access control
   - Audit logging

## Troubleshooting

### Common Issues

#### Low Frame Rate
**Symptoms**: Stuttering, lag, poor responsiveness
**Solutions**:
- Reduce quality level manually
- Disconnect unnecessary robots
- Remove non-essential overlays
- Check network latency

#### High Latency
**Symptoms**: Delayed responses, desynced visuals
**Solutions**:
- Switch to wired connection
- Reduce spatial anchor count
- Disable non-critical overlays
- Check server load

#### Hand Tracking Not Working
**Symptoms**: Gestures not recognized
**Solutions**:
- Ensure good lighting
- Clean device cameras
- Recalibrate hand tracking
- Switch to controller mode

#### Spatial Anchors Drifting
**Symptoms**: AR content moves unexpectedly
**Solutions**:
- Recalibrate AR device
- Improve environment lighting
- Add more tracking markers
- Reset spatial mapping

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `ARVR_001` | Session not found | Create new session |
| `ARVR_002` | WebSocket connection failed | Check network connectivity |
| `ARVR_003` | Robot connection failed | Verify robot ID and availability |
| `ARVR_004` | Insufficient performance | Reduce quality level |
| `ARVR_005` | Device not supported | Use compatible device |
| `ARVR_006` | Gesture recognition failed | Recalibrate or switch input method |
| `ARVR_007` | Spatial anchor limit reached | Remove old anchors |
| `ARVR_008` | Haptic feedback not available | Check device capabilities |

## Hardware Requirements

### VR Headsets

**Meta Quest 2/3**
- Resolution: 1832x1920 per eye
- Refresh Rate: 90/120 Hz
- Hand Tracking: Yes
- Controllers: Touch Controllers

**Valve Index**
- Resolution: 1440x1600 per eye
- Refresh Rate: 120/144 Hz
- Hand Tracking: With Leap Motion
- Controllers: Index Controllers

### AR Devices

**Microsoft HoloLens 2**
- Resolution: 2K per eye
- Field of View: 52°
- Hand Tracking: Yes
- Voice Control: Yes

**Mobile AR (ARCore/ARKit)**
- Android 7.0+ (ARCore)
- iOS 11.0+ (ARKit)
- 6-axis IMU
- Rear camera

### System Requirements

**PC (for PC VR)**
- CPU: Intel i5-4590 / AMD Ryzen 5 1500X or better
- GPU: NVIDIA GTX 1060 / AMD RX 480 or better
- RAM: 8GB minimum, 16GB recommended
- USB: 3.0 port for headset
- OS: Windows 10/11, Linux (limited support)

**Network**
- Bandwidth: 10 Mbps minimum
- Latency: <50ms recommended
- Connection: WiFi 6 or Ethernet recommended

## Future Enhancements

- [ ] Multi-user shared VR spaces
- [ ] AI-powered gesture recognition
- [ ] Advanced haptic suits integration
- [ ] Eye tracking for gaze interaction
- [ ] Brain-computer interface support
- [ ] Photorealistic rendering
- [ ] 8K video streaming
- [ ] Volumetric video capture
- [ ] Neural rendering for digital twins
- [ ] 5G edge computing integration

## Support

For issues or questions:
- Email: support@physical-ai-platform.com
- Docs: https://docs.physical-ai-platform.com/ar-vr
- Community: https://community.physical-ai-platform.com

## License

Copyright © 2024 Physical AI Platform. All rights reserved.
