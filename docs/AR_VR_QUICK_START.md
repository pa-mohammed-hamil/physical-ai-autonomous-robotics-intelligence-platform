# AR/VR Integration - Quick Start Guide

## 5-Minute Setup

### Step 1: Open the Frontend
```bash
# Open in browser
http://localhost:8000/frontend/ar-vr-viewer.html
```

### Step 2: Initialize Session
1. Select your device type (VR Headset, AR Phone, AR Glasses, etc.)
2. Choose interaction mode (Controller, Hand Tracking, Voice, etc.)
3. Click **"Initialize Session"**

### Step 3: Connect Robots
1. Enter robot ID (e.g., `ROBOT_001`)
2. Click **"Connect Robot"**
3. Repeat for multiple robots

### Step 4: Start Monitoring
- Use view mode controls to switch perspectives
- Create spatial anchors by clicking the pin icon
- Toggle holographic overlays with the sparkle icon

## Quick API Examples

### Python

```python
import asyncio
from app.vr.ar_vr_integration import ar_vr_manager, DeviceType, ViewMode

async def quick_start():
    # 1. Create session
    session = await ar_vr_manager.create_session(
        user_id="operator_001",
        device_type=DeviceType.VR_HEADSET,
        view_mode=ViewMode.THIRD_PERSON
    )
    
    # 2. Connect robot
    await ar_vr_manager.connect_robot(session.session_id, "ROBOT_001")
    
    # 3. Create holographic overlay
    await ar_vr_manager.create_holographic_overlay(
        session_id=session.session_id,
        robot_id="ROBOT_001",
        overlay_type="health",
        position={"x": 0, "y": 2, "z": 0},
        data={"health": 92, "battery": 78}
    )
    
    print(f"Session ready: {session.session_id}")

asyncio.run(quick_start())
```

### JavaScript

```javascript
// 1. Create session
const response = await fetch('http://localhost:8000/api/v1/ar-vr/session', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        user_id: 'operator_001',
        device_type: 'vr_headset',
        view_mode: 'third_person',
        interaction_mode: 'controller'
    })
});

const {session_id, websocket_url} = await response.json();

// 2. Connect WebSocket
const ws = new WebSocket(websocket_url);
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

// 3. Connect robot
await fetch(`http://localhost:8000/api/v1/ar-vr/session/${session_id}/robot`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({robot_id: 'ROBOT_001'})
});
```

### cURL

```bash
# 1. Create session
curl -X POST http://localhost:8000/api/v1/ar-vr/session \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "operator_001",
    "device_type": "vr_headset",
    "view_mode": "third_person",
    "interaction_mode": "controller"
  }'

# 2. Connect robot (use session_id from above)
curl -X POST http://localhost:8000/api/v1/ar-vr/session/arvr_operator_001_1234567890/robot \
  -H "Content-Type: application/json" \
  -d '{"robot_id": "ROBOT_001"}'

# 3. Get session info
curl http://localhost:8000/api/v1/ar-vr/session/arvr_operator_001_1234567890
```

## Common Use Cases

### 1. Single Robot Monitoring
```python
session = await ar_vr_manager.create_session(
    user_id="tech_001",
    device_type=DeviceType.AR_PHONE,
    view_mode=ViewMode.THIRD_PERSON
)
await ar_vr_manager.connect_robot(session.session_id, "ROBOT_001")
```

### 2. Fleet Monitoring (God Mode)
```python
session = await ar_vr_manager.create_session(
    user_id="supervisor_001",
    device_type=DeviceType.VR_HEADSET,
    view_mode=ViewMode.GOD_MODE
)

# Connect entire fleet
for robot_id in ["ROBOT_001", "ROBOT_002", "ROBOT_003"]:
    await ar_vr_manager.connect_robot(session.session_id, robot_id)
```

### 3. AR-Guided Maintenance
```python
session = await ar_vr_manager.create_session(
    user_id="maintenance_tech_001",
    device_type=DeviceType.AR_GLASSES,
    view_mode=ViewMode.HOLOGRAPHIC,
    interaction_mode=InteractionMode.VOICE
)

await ar_vr_manager.connect_robot(session.session_id, "ROBOT_001")

# Create maintenance overlays
await ar_vr_manager.create_holographic_overlay(
    session_id=session.session_id,
    robot_id="ROBOT_001",
    overlay_type="instruction",
    position={"x": 0, "y": 2, "z": 0},
    data={"step": 1, "instruction": "Inspect drive motors"}
)
```

### 4. Hand Gesture Control
```python
session = await ar_vr_manager.create_session(
    user_id="operator_001",
    device_type=DeviceType.VR_HEADSET,
    interaction_mode=InteractionMode.HAND_TRACKING
)

# Process gestures
await ar_vr_manager.process_hand_gesture(
    session_id=session.session_id,
    gesture_type="point",
    hand="right",
    position={"x": 1, "y": 1.5, "z": -2},
    confidence=0.95
)
```

### 5. Spatial Anchors for Waypoints
```python
# Create waypoint markers
waypoints = [
    {"x": 0, "y": 0, "z": 0},
    {"x": 5, "y": 0, "z": 5},
    {"x": 10, "y": 0, "z": 0}
]

for i, pos in enumerate(waypoints):
    await ar_vr_manager.create_spatial_anchor(
        session_id=session.session_id,
        position=pos,
        rotation={"w": 1, "x": 0, "y": 0, "z": 0},
        content_type="waypoint",
        content_data={"number": i+1},
        persistent=True
    )
```

## View Modes Explained

| Mode | Description | Best For |
|------|-------------|----------|
| **First Person** | See through robot's eyes | Remote operation, telepresence |
| **Third Person** | External view of robot | Monitoring, training |
| **God Mode** | Bird's eye view of all robots | Fleet management, supervision |
| **Split View** | Multiple robots side-by-side | Comparison, coordination |
| **Spatial Map** | AR overlay on real world | On-site maintenance, AR navigation |
| **Holographic** | 3D projections in AR | AR instructions, data visualization |
| **Twin View** | Digital twin overlay | Predictive maintenance, simulation |

## Interaction Methods

| Method | Devices | Use Cases |
|--------|---------|-----------|
| **Controller** | VR headsets | Precise control, gaming-style interaction |
| **Hand Tracking** | Quest 2/3, HoloLens | Natural gestures, hands-free operation |
| **Gaze** | Most VR/AR | Simple selection, accessibility |
| **Voice** | HoloLens, mobile | Hands-free commands, multitasking |
| **Gesture** | HoloLens, Leap Motion | Air tap, pinch, grab |
| **Touch** | AR phones/tablets | Mobile AR interaction |

## Keyboard Shortcuts (WebXR)

| Key | Action |
|-----|--------|
| `1` | First Person View |
| `2` | Third Person View |
| `3` | God Mode |
| `Space` | Create Spatial Anchor |
| `H` | Toggle Holographic Overlays |
| `G` | Toggle Gesture Guide |
| `R` | Reset View |
| `Esc` | Close Session |

## Troubleshooting

### Issue: Can't create session
**Solution**: Ensure FastAPI backend is running on port 8000

### Issue: WebSocket connection fails
**Solution**: Check WebSocket URL in browser console, verify backend is running

### Issue: Robots not appearing
**Solution**: Verify robot IDs are correct, check robot connection status in API

### Issue: Poor performance
**Solution**: Reduce quality level, disconnect unused robots, remove old spatial anchors

### Issue: Hand tracking not working
**Solution**: Ensure good lighting, clean device cameras, recalibrate tracking

## Performance Tips

1. **Target 90 FPS for VR** - Essential for comfort
2. **Keep latency <20ms** - Critical for real-time control
3. **Limit spatial anchors** - Maximum 20-30 for good performance
4. **Reduce overlay complexity** - Simple overlays render faster
5. **Use adaptive quality** - System adjusts automatically

## Next Steps

1. **Read Full Guide**: [AR_VR_INTEGRATION_GUIDE.md](./AR_VR_INTEGRATION_GUIDE.md)
2. **Run Examples**: `python examples/ar_vr_examples.py`
3. **Explore API**: http://localhost:8000/docs#/AR%2FVR%20Integration
4. **Test WebXR**: Use Chrome/Edge with WebXR extensions

## Support Resources

- **Full Documentation**: [AR_VR_INTEGRATION_GUIDE.md](./AR_VR_INTEGRATION_GUIDE.md)
- **API Reference**: http://localhost:8000/docs
- **Examples**: [examples/ar_vr_examples.py](../examples/ar_vr_examples.py)
- **Frontend**: [frontend/ar-vr-viewer.html](../frontend/ar-vr-viewer.html)

## WebXR Browser Support

| Browser | VR | AR | Notes |
|---------|----|----|-------|
| Chrome 79+ | ✅ | ✅ | Best WebXR support |
| Edge 79+ | ✅ | ✅ | Chromium-based |
| Firefox 98+ | ✅ | ⚠️ | Limited AR |
| Safari | ❌ | ⚠️ | QuickLook AR only |

## Hardware Recommendations

**Budget Setup ($300-500)**
- Meta Quest 2
- WiFi 6 router
- Mid-range PC (optional for PCVR)

**Professional Setup ($1000-2000)**
- Meta Quest 3 or Valve Index
- Microsoft HoloLens 2
- High-end PC with RTX 3070+
- Dedicated 5GHz WiFi network

**Enterprise Setup ($5000+)**
- Multiple VR headsets
- Microsoft HoloLens 2 fleet
- Dedicated server infrastructure
- Haptic gloves/suits
- Omni-directional treadmill

---

**Ready to start?** Open `ar-vr-viewer.html` and click "Initialize Session"!
