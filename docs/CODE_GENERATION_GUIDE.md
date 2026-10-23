# AI-Powered Code Generation Guide

## Overview

The AI-Powered Code Generation feature enables developers to generate robot control scripts from natural language descriptions. This feature leverages large language models (LLMs) to translate high-level requirements into production-ready code for various robotics frameworks.

## Table of Contents

1. [Features](#features)
2. [Supported Languages & Frameworks](#supported-languages--frameworks)
3. [Getting Started](#getting-started)
4. [API Reference](#api-reference)
5. [Code Templates](#code-templates)
6. [Safety Levels](#safety-levels)
7. [Best Practices](#best-practices)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Features

### Core Capabilities

- **Natural Language to Code**: Describe what you want in plain English, get working code
- **Multi-Language Support**: Generate Python, C++, JavaScript, and YAML configurations
- **Multi-Framework Support**: ROS2, MoveIt2, Nav2, PyBullet, Gazebo, Isaac Sim
- **Template-Based Generation**: Start from pre-built templates for common tasks
- **Automatic Validation**: Syntax checking and error detection
- **Security Analysis**: Detect potentially unsafe code patterns
- **Documentation Generation**: Auto-generate comments and docstrings
- **Test Generation**: Create unit tests for generated code
- **Code Explanation**: Understand existing code with AI-powered analysis
- **Code Refactoring**: Improve existing code quality automatically
- **Complexity Analysis**: Get readability and maintainability scores

### Safety Features

- **Multi-Level Safety**: Simulation-only, supervised, and production modes
- **Security Scanning**: Detect exec(), eval(), and other dangerous patterns
- **Best Practice Enforcement**: Follow ROS, PEP8, and robotics standards
- **Error Handling**: Automatic exception handling and logging
- **Input Validation**: Prevent injection and malformed requests

---

## Supported Languages & Frameworks

### Programming Languages

| Language | Support Level | Use Cases |
|----------|--------------|-----------|
| **Python** | Full | ROS2 nodes, AI/ML pipelines, rapid prototyping |
| **C++** | Partial | Performance-critical code, real-time control |
| **JavaScript** | Partial | Web dashboards, visualization tools |
| **YAML** | Full | Launch files, configuration files |

### Robotics Frameworks

| Framework | Description | Key Features |
|-----------|-------------|--------------|
| **ROS2** | Robot Operating System 2 | Pub/sub, services, actions, lifecycle nodes |
| **MoveIt2** | Motion planning framework | Trajectory planning, collision checking, IK/FK |
| **Nav2** | Navigation framework | Path planning, obstacle avoidance, localization |
| **PyBullet** | Physics simulation | Dynamics simulation, sensor simulation |
| **Gazebo** | Robot simulation | 3D visualization, sensor plugins, world modeling |
| **Isaac Sim** | NVIDIA simulation | Photorealistic rendering, GPU acceleration |

---

## Getting Started

### Prerequisites

```bash
# Install required packages
pip install httpx asyncio

# Set up environment variables
export ROBOTICS_API_URL="http://localhost:8000"
export ROBOTICS_API_KEY="your-api-key"
```

### Basic Usage

```python
import httpx

API_URL = "http://localhost:8000/api/v1/copilot/code/generate"
API_KEY = "your-api-key"

# Prepare request
request_data = {
    "description": "Create a ROS2 node that publishes velocity commands",
    "language": "python",
    "framework": "ros2",
    "template": "ros2_publisher",
    "include_tests": True,
    "include_documentation": True
}

# Generate code
response = httpx.post(
    API_URL,
    json=request_data,
    headers={"Authorization": f"Bearer {API_KEY}"},
    timeout=60.0
)

result = response.json()
print(result['code'])
```

### Quick Start with Examples

```bash
# Run example code generation
cd examples
python code_generation_examples.py

# Run specific example
python code_generation_examples.py 1  # ROS2 publisher
python code_generation_examples.py 2  # Navigation controller
```

---

## API Reference

### Generate Code

**Endpoint:** `POST /api/v1/copilot/code/generate`

Generate robot control code from natural language description.

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `description` | string | Yes | - | Natural language description (10-2000 chars) |
| `language` | string | No | "python" | Programming language (python, cpp, javascript, yaml) |
| `framework` | string | No | "ros2" | Robotics framework (ros2, moveit2, nav2, etc.) |
| `template` | string | No | null | Code template to use as base |
| `style` | string | No | "ros" | Code style (pep8, google, ros, compact, verbose) |
| `safety_level` | string | No | "simulation_only" | Safety level (simulation_only, supervised, production) |
| `include_tests` | boolean | No | false | Generate unit tests |
| `include_documentation` | boolean | No | true | Generate documentation |
| `include_error_handling` | boolean | No | true | Include error handling |
| `constraints` | array | No | null | Additional requirements |
| `robot_config` | object | No | null | Robot-specific configuration |

#### Response

```json
{
  "code": "# Generated code...",
  "language": "python",
  "framework": "ros2",
  "explanation": "This code creates a ROS2 publisher node...",
  "warnings": [],
  "dependencies": ["rclpy", "geometry_msgs"],
  "usage_example": "# Example usage...",
  "test_code": "# Unit tests...",
  "documentation": "# Documentation...",
  "safety_notes": ["⚠️ Test in simulation first"],
  "complexity_score": 3.5,
  "estimated_lines": 45,
  "metadata": {
    "template": "ros2_publisher",
    "style": "ros",
    "security_issues": []
  }
}
```

#### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/copilot/code/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Create a ROS2 node that subscribes to camera images and detects faces",
    "language": "python",
    "framework": "ros2",
    "template": "sensor_processor",
    "include_tests": true,
    "constraints": [
      "Must handle missing camera gracefully",
      "Should publish detection results"
    ],
    "robot_config": {
      "camera_topic": "/camera/image_raw",
      "detection_topic": "/face_detections"
    }
  }'
```

---

### Validate Code

**Endpoint:** `POST /api/v1/copilot/code/validate`

Validate code for syntax, security, and style issues.

#### Request

```json
{
  "code": "import rclpy\n...",
  "language": "python"
}
```

#### Response

```json
{
  "is_valid": true,
  "syntax_errors": [],
  "security_issues": ["Potentially unsafe pattern: exec()"],
  "style_violations": [],
  "warnings": ["Code contains TODO comments"],
  "suggestions": ["Consider adding type hints"]
}
```

---

### Explain Code

**Endpoint:** `POST /api/v1/copilot/code/explain`

Get AI-powered explanation of existing code.

#### Request

```json
{
  "code": "class MyNode(Node):\n...",
  "language": "python",
  "detail_level": "standard"
}
```

#### Response

```json
{
  "explanation": "This code defines a ROS2 node that...",
  "key_components": [
    {"name": "MyNode", "type": "class", "purpose": "..."}
  ],
  "dependencies": ["rclpy"],
  "suggestions": ["Add logging", "Handle shutdown gracefully"]
}
```

---

### Refactor Code

**Endpoint:** `POST /api/v1/copilot/code/refactor`

Improve code quality through AI-powered refactoring.

#### Request

```json
{
  "code": "def process():\n  ...",
  "language": "python",
  "goals": [
    "improve performance",
    "add error handling",
    "enhance readability"
  ]
}
```

#### Response

```json
{
  "refactored_code": "def process() -> None:\n  ...",
  "changes_made": [
    "Added type hints",
    "Extracted helper functions",
    "Added error handling"
  ],
  "explanation": "Code refactored to improve maintainability...",
  "before_complexity": 6.5,
  "after_complexity": 4.2
}
```

---

### Get Templates

**Endpoint:** `GET /api/v1/copilot/code/templates`

List available code templates.

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `framework` | string | No | Filter by framework |

#### Response

```json
{
  "templates": [
    {
      "id": "ros2_publisher",
      "name": "ROS2 Publisher",
      "description": "ROS2 publisher node with timer callback",
      "framework": "ros2",
      "language": "python"
    }
  ]
}
```

---

## Code Templates

Pre-built templates for common robotics tasks.

### ROS2 Templates

| Template ID | Description | Use Case |
|-------------|-------------|----------|
| `ros2_node` | Basic ROS2 node | Starting point for any ROS2 node |
| `ros2_publisher` | Publisher with timer | Periodic data publishing |
| `ros2_subscriber` | Subscriber with callback | React to incoming messages |
| `ros2_service` | Service server | Request-response pattern |
| `ros2_action_server` | Action server | Long-running tasks with feedback |
| `ros2_action_client` | Action client | Goal management and cancellation |
| `launch_file` | Python launch file | Multi-node orchestration |

### MoveIt2 Templates

| Template ID | Description | Use Case |
|-------------|-------------|----------|
| `moveit_planning` | Motion planning pipeline | Trajectory planning and execution |
| `grasp_planner` | Grasp planning | Object manipulation |
| `trajectory_controller` | Joint control | Low-level motor control |

### Nav2 Templates

| Template ID | Description | Use Case |
|-------------|-------------|----------|
| `nav2_waypoint_follower` | Waypoint navigation | Sequential navigation |

### General Templates

| Template ID | Description | Use Case |
|-------------|-------------|----------|
| `sensor_processor` | Sensor data processing | Camera, LiDAR, IMU processing |
| `behavior_tree` | Behavior tree | Complex task orchestration |

---

## Safety Levels

### Simulation Only

```python
{
    "safety_level": "simulation_only"
}
```

**Use When:**
- Developing new algorithms
- Testing unvalidated code
- Experimenting with robot behavior

**Safety Features:**
- Clear warnings about simulation-only status
- Encourages thorough testing
- No hardware safety checks included

**Safety Notes:**
- ⚠️ Never deploy directly to hardware
- ⚠️ Test extensively in simulation first
- ⚠️ Validate physics accuracy

---

### Supervised

```python
{
    "safety_level": "supervised"
}
```

**Use When:**
- Initial hardware testing
- Demonstration environments
- Research labs

**Safety Features:**
- Emergency stop reminders
- Human supervision requirements
- Basic collision detection

**Safety Notes:**
- ⚠️ Requires human operator present
- ⚠️ Keep emergency stop accessible
- ⚠️ Monitor robot continuously
- ⚠️ Test in controlled environment

---

### Production

```python
{
    "safety_level": "production"
}
```

**Use When:**
- Deployed applications
- Autonomous operation
- Industrial settings

**Safety Features:**
- Comprehensive error handling
- Hardware safety checks
- Compliance with safety standards
- Robust failure recovery
- Extensive logging

**Safety Notes:**
- ✅ Production-grade safety features
- ✅ ISO 10218 compliance considerations
- ⚠️ Still requires safety certification
- ⚠️ Regular safety audits needed

---

## Best Practices

### Writing Effective Descriptions

#### ✅ Good Descriptions

```python
# Specific, actionable, includes constraints
"Create a ROS2 node that subscribes to /scan topic (LaserScan messages), 
detects obstacles within 0.5 meters, and publishes Twist messages to /cmd_vel 
to avoid collisions. Use a 10Hz update rate."
```

```python
# Clear goal with technical details
"Implement a MoveIt2 pick-and-place pipeline that:
1. Detects objects using camera
2. Plans collision-free grasp
3. Executes pick motion
4. Moves to drop-off location
5. Releases object
Include error handling for failed grasps."
```

#### ❌ Poor Descriptions

```python
# Too vague
"Make a robot move"

# Too short
"ROS2 node"

# Unclear requirements
"Do something with the camera"
```

### Description Guidelines

1. **Be Specific**: Mention exact topics, message types, frequencies
2. **Include Context**: What problem does this solve?
3. **Specify Constraints**: Performance, safety, resource limits
4. **Mention Error Cases**: What should happen when things fail?
5. **Provide Examples**: Sample inputs/outputs if applicable

### Optimal Parameters

```python
# For production code
{
    "language": "python",
    "framework": "ros2",
    "style": "ros",
    "safety_level": "production",
    "include_tests": True,
    "include_documentation": True,
    "include_error_handling": True
}
```

```python
# For rapid prototyping
{
    "language": "python",
    "template": "ros2_node",
    "style": "compact",
    "safety_level": "simulation_only",
    "include_tests": False,
    "include_documentation": False
}
```

### Code Review Checklist

Before deploying generated code:

- [ ] Review all safety-critical sections
- [ ] Validate against your robot's specifications
- [ ] Test in simulation extensively
- [ ] Run generated unit tests
- [ ] Check dependency versions
- [ ] Verify error handling paths
- [ ] Review security warnings
- [ ] Add robot-specific calibration
- [ ] Document any modifications
- [ ] Perform integration testing

---

## Examples

### Example 1: Simple ROS2 Publisher

```python
request = {
    "description": "Create a ROS2 node that publishes 'Hello Robot' string messages to /chatter topic at 1Hz",
    "language": "python",
    "framework": "ros2",
    "template": "ros2_publisher"
}
```

**Generated Code:**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ChatterPublisher(Node):
    def __init__(self):
        super().__init__('chatter_publisher')
        self.publisher_ = self.create_publisher(String, '/chatter', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Chatter publisher initialized')
        
    def timer_callback(self):
        msg = String()
        msg.data = 'Hello Robot'
        self.publisher_.publish(msg)
        self.get_logger().debug('Published: Hello Robot')

def main(args=None):
    rclpy.init(args=args)
    node = ChatterPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

### Example 2: Obstacle Avoidance

```python
request = {
    "description": "Create obstacle avoidance node that reads LaserScan from /scan, "
                   "stops if obstacle within 0.3m, slows down if within 0.6m, "
                   "otherwise moves forward at 0.5 m/s. Publish to /cmd_vel.",
    "language": "python",
    "framework": "ros2",
    "safety_level": "supervised",
    "include_error_handling": True
}
```

---

### Example 3: MoveIt2 Motion Planning

```python
request = {
    "description": "Create MoveIt2 node that moves robot arm to predefined poses. "
                   "Should accept target pose via service call, plan trajectory "
                   "with collision checking, and execute motion. Include success "
                   "feedback and error recovery.",
    "framework": "moveit2",
    "template": "moveit_planning",
    "safety_level": "supervised",
    "constraints": [
        "Maximum velocity: 0.5 rad/s",
        "Planning timeout: 5 seconds",
        "Must check for collisions"
    ]
}
```

---

### Example 4: Camera Processing

```python
request = {
    "description": "Process camera images to detect ArUco markers. Subscribe to "
                   "/camera/image_raw, detect markers, publish marker poses to "
                   "/aruco_poses, and visualization markers to /aruco_markers. "
                   "Handle camera disconnection gracefully.",
    "language": "python",
    "framework": "ros2",
    "template": "sensor_processor",
    "robot_config": {
        "camera_topic": "/camera/image_raw",
        "marker_size": 0.15,
        "marker_dict": "DICT_4X4_50"
    }
}
```

---

## Troubleshooting

### Common Issues

#### Generated Code Has Syntax Errors

**Problem:** Code fails validation with syntax errors.

**Solutions:**
1. Enable validation: Code is auto-fixed in most cases
2. Check description clarity: Ambiguous descriptions cause errors
3. Retry generation: LLMs are probabilistic
4. Use templates: More reliable starting point

#### Code Doesn't Match Requirements

**Problem:** Generated code doesn't do what was requested.

**Solutions:**
1. Make description more specific
2. Add constraints and examples
3. Specify exact topic names and types
4. Use appropriate template
5. Try refactoring endpoint with goals

#### Security Warnings

**Problem:** Generated code has security issues.

**Solutions:**
1. Review flagged patterns
2. Replace unsafe code (exec, eval)
3. Use environment variables for credentials
4. Enable security checking in production

#### High Complexity Score

**Problem:** Generated code is too complex (>7.0).

**Solutions:**
1. Break into smaller components
2. Use refactoring endpoint
3. Simplify requirements
4. Use helper functions

#### Missing Dependencies

**Problem:** Code imports unavailable packages.

**Solutions:**
1. Check `dependencies` field in response
2. Install missing packages: `pip install <package>`
3. Specify robot_config with available libraries
4. Use constraints to limit dependencies

---

## Advanced Features

### Batch Generation

Generate multiple related files:

```python
async def generate_package():
    tasks = [
        generate_code(description="Publisher node"),
        generate_code(description="Subscriber node"),
        generate_code(description="Launch file"),
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Iterative Refinement

Improve generated code:

```python
# Step 1: Generate initial code
result1 = await generate_code(description="Basic navigation")

# Step 2: Refactor for better quality
result2 = await refactor_code(
    code=result1['code'],
    goals=["add error handling", "improve performance"]
)

# Step 3: Add tests
result3 = await generate_code(
    description=f"Unit tests for: {result1['explanation']}",
    template="pytest_tests"
)
```

### Custom Templates

Create custom templates for your robots:

```python
custom_robot_config = {
    "robot_type": "mobile_manipulator",
    "arm_dof": 7,
    "base_type": "differential_drive",
    "sensors": ["lidar", "camera", "imu"],
    "namespaces": {
        "arm": "/arm",
        "base": "/base"
    }
}

result = await generate_code(
    description="Full robot control stack",
    robot_config=custom_robot_config
)
```

---

## Performance Tips

1. **Use Templates**: 2-3x faster than free-form generation
2. **Disable Tests**: Skip tests during prototyping for speed
3. **Cache Results**: Store generated code to avoid regeneration
4. **Batch Requests**: Generate multiple files concurrently
5. **Simplify Descriptions**: Shorter descriptions = faster generation

---

## API Rate Limits

| Tier | Requests/Min | Requests/Day | Max Code Length |
|------|--------------|--------------|-----------------|
| Free | 10 | 100 | 2000 lines |
| Standard | 60 | 1000 | 5000 lines |
| Premium | 300 | Unlimited | 10000 lines |

---

## FAQ

**Q: Can I use generated code in commercial products?**  
A: Yes, generated code is yours to use commercially. Review our license terms.

**Q: Is generated code guaranteed to be correct?**  
A: No. Always review, test, and validate generated code before deployment.

**Q: What happens if generation fails?**  
A: API returns error with details. Common fixes: simplify description, try template, retry.

**Q: Can I customize code style?**  
A: Yes, use `style` parameter (pep8, google, ros, compact, verbose).

**Q: Does it support my custom robot?**  
A: Yes, provide robot specs in `robot_config` parameter.

**Q: How accurate is complexity scoring?**  
A: Heuristic-based estimation. Use as relative comparison, not absolute measure.

---

## Support

- **Documentation**: `/docs`
- **API Reference**: `http://localhost:8000/docs`
- **Examples**: `/examples/code_generation_examples.py`
- **Issues**: GitHub Issues
- **Community**: Discord / Forums

---

## Changelog

### v2.7.0 (Current)
- ✨ Added code generation from natural language
- ✨ Multi-framework support (ROS2, MoveIt2, Nav2)
- ✨ Automatic validation and security checking
- ✨ Test generation
- ✨ Code explanation and refactoring
- ✨ Template library

### Future Releases
- 🔜 C++ code generation (full support)
- 🔜 Behavior tree generation
- 🔜 URDF/SDF file generation
- 🔜 Configuration file generation
- 🔜 Custom template upload

---

## License

See main LICENSE file for details.

---

**Last Updated:** October 2026  
**API Version:** v1  
**Document Version:** 2.7.0
