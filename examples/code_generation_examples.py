"""
Code Generation Examples

Demonstrates how to use the AI-powered code generation API to create
robot control scripts from natural language descriptions.
"""

import asyncio
import json
from typing import Dict

import httpx


# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "your-api-key-here"  # Replace with actual API key


class CodeGenerationClient:
    """Client for code generation API."""
    
    def __init__(self, base_url: str = API_BASE_URL, api_key: str = API_KEY):
        """Initialize client."""
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    async def generate_code(
        self,
        description: str,
        language: str = "python",
        framework: str = "ros2",
        template: str = None,
        include_tests: bool = False,
        **kwargs
    ) -> Dict:
        """Generate code from natural language description."""
        request_data = {
            "description": description,
            "language": language,
            "framework": framework,
            "template": template,
            "include_tests": include_tests,
            **kwargs
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/copilot/code/generate",
                json=request_data,
                headers=self.headers,
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def validate_code(self, code: str, language: str = "python") -> Dict:
        """Validate code."""
        request_data = {
            "code": code,
            "language": language,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/copilot/code/validate",
                json=request_data,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
    
    async def explain_code(
        self,
        code: str,
        language: str = "python",
        detail_level: str = "standard"
    ) -> Dict:
        """Get explanation of code."""
        request_data = {
            "code": code,
            "language": language,
            "detail_level": detail_level,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/copilot/code/explain",
                json=request_data,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
    
    async def refactor_code(
        self,
        code: str,
        goals: list,
        language: str = "python"
    ) -> Dict:
        """Refactor code."""
        request_data = {
            "code": code,
            "language": language,
            "goals": goals,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/copilot/code/refactor",
                json=request_data,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_templates(self, framework: str = None) -> Dict:
        """Get available templates."""
        params = {"framework": framework} if framework else {}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/copilot/code/templates",
                params=params,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()


# ============================================================================
# Example 1: Generate ROS2 Publisher Node
# ============================================================================


async def example_ros2_publisher():
    """Generate a ROS2 publisher node."""
    print("\n" + "="*80)
    print("Example 1: Generate ROS2 Publisher Node")
    print("="*80)
    
    client = CodeGenerationClient()
    
    result = await client.generate_code(
        description="Create a ROS2 node that publishes Twist messages to cmd_vel topic at 10Hz. "
                   "The node should publish forward velocity of 0.5 m/s when active.",
        language="python",
        framework="ros2",
        template="ros2_publisher",
        include_tests=True,
        include_documentation=True,
    )
    
    print(f"\n✅ Generated Code ({result['estimated_lines']} lines):")
    print(result['code'])
    
    print(f"\n📋 Dependencies: {', '.join(result['dependencies'])}")
    print(f"\n📊 Complexity Score: {result['complexity_score']:.2f}/10")
    
    if result['safety_notes']:
        print(f"\n⚠️ Safety Notes:")
        for note in result['safety_notes']:
            print(f"  - {note}")
    
    if result['usage_example']:
        print(f"\n📖 Usage Example:")
        print(result['usage_example'])
    
    return result


# ============================================================================
# Example 2: Generate Navigation Controller
# ============================================================================


async def example_navigation_controller():
    """Generate a navigation controller using Nav2."""
    print("\n" + "="*80)
    print("Example 2: Generate Navigation Controller")
    print("="*80)
    
    client = CodeGenerationClient()
    
    result = await client.generate_code(
        description="Create a ROS2 node that uses Nav2 to navigate through a list of waypoints. "
                   "The node should accept waypoints via a service call and navigate to each "
                   "waypoint in sequence. Include error handling and recovery behaviors.",
        language="python",
        framework="nav2",
        template="nav2_waypoint_follower",
        safety_level="supervised",
        include_error_handling=True,
        constraints=[
            "Must handle navigation failures gracefully",
            "Should provide progress feedback",
            "Include timeout handling",
        ],
    )
    
    print(f"\n✅ Generated Code:")
    print(result['code'])
    
    print(f"\n📚 Documentation:")
    print(result['documentation'])
    
    return result


# ============================================================================
# Example 3: Generate Object Detection Pipeline
# ============================================================================


async def example_object_detection():
    """Generate an object detection pipeline."""
    print("\n" + "="*80)
    print("Example 3: Generate Object Detection Pipeline")
    print("="*80)
    
    client = CodeGenerationClient()
    
    result = await client.generate_code(
        description="Create a ROS2 node that subscribes to camera images, performs object "
                   "detection using a YOLO model, and publishes detected objects with "
                   "bounding boxes. Include visualization markers.",
        language="python",
        framework="ros2",
        template="sensor_processor",
        include_tests=False,
        robot_config={
            "camera_topic": "/camera/image_raw",
            "detection_topic": "/detections",
            "model_path": "/models/yolov8n.pt",
        },
    )
    
    print(f"\n✅ Generated Code:")
    print(result['code'])
    
    return result


# ============================================================================
# Example 4: Generate Grasp Planning Code
# ============================================================================


async def example_grasp_planning():
    """Generate grasp planning code."""
    print("\n" + "="*80)
    print("Example 4: Generate Grasp Planning Code")
    print("="*80)
    
    client = CodeGenerationClient()
    
    result = await client.generate_code(
        description="Create a MoveIt2 node that plans and executes grasps. "
                   "The node should take object pose as input, generate grasp candidates, "
                   "select the best grasp based on reachability and collision checking, "
                   "and execute the grasp motion.",
        language="python",
        framework="moveit2",
        template="grasp_planner",
        safety_level="supervised",
        include_error_handling=True,
        constraints=[
            "Must check for collisions",
            "Should validate joint limits",
            "Include pre-grasp and post-grasp poses",
        ],
    )
    
    print(f"\n✅ Generated Code:")
    print(result['code'])
    
    if result['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    return result


# ============================================================================
# Example 5: Generate Launch File
# ============================================================================


async def example_launch_file():
    """Generate a ROS2 launch file."""
    print("\n" + "="*80)
    print("Example 5: Generate ROS2 Launch File")
    print("="*80)
    
    client = CodeGenerationClient()
    
    result = await client.generate_code(
        description="Create a ROS2 Python launch file that starts a robot with: "
                   "1) Robot state publisher "
                   "2) Nav2 stack "
                   "3) MoveIt2 "
                   "4) RViz with custom config "
                   "Include parameters for each node.",
        language="python",
        framework="ros2",
        template="launch_file",
        robot_config={
            "robot_name": "my_robot",
            "use_sim_time": True,
            "namespace": "/robot1",
        },
    )
    
    print(f"\n✅ Generated Launch File:")
    print(result['code'])
    
    return result


# ============================================================================
# Example 6: Validate Existing Code
# ============================================================================


async def example_code_validation():
    """Validate existing code."""
    print("\n" + "="*80)
    print("Example 6: Validate Existing Code")
    print("="*80)
    
    client = CodeGenerationClient()
    
    # Sample code to validate (intentionally has issues)
    code_to_validate = """
import rclpy
from geometry_msgs.msg import Twist

def main():
    rclpy.init()
    node = rclpy.create_node('test_node')
    pub = node.create_publisher(Twist, 'cmd_vel', 10)
    
    msg = Twist()
    msg.linear.x = 1.0  # Fast speed without safety check!
    
    pub.publish(msg)
    exec("print('unsafe code')")  # Security issue!
"""
    
    result = await client.validate_code(code_to_validate, language="python")
    
    print(f"\n✅ Validation Result:")
    print(f"Is Valid: {result['is_valid']}")
    
    if result['syntax_errors']:
        print(f"\n❌ Syntax Errors:")
        for error in result['syntax_errors']:
            print(f"  - {error}")
    
    if result['security_issues']:
        print(f"\n🔒 Security Issues:")
        for issue in result['security_issues']:
            print(f"  - {issue}")
    
    if result['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    return result


# ============================================================================
# Example 7: Explain Existing Code
# ============================================================================


async def example_code_explanation():
    """Get explanation of existing code."""
    print("\n" + "="*80)
    print("Example 7: Explain Existing Code")
    print("="*80)
    
    client = CodeGenerationClient()
    
    # Sample code to explain
    code_to_explain = """
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
    def scan_callback(self, msg):
        min_distance = min(msg.ranges)
        
        twist = Twist()
        if min_distance < 0.5:
            twist.linear.x = 0.0
            twist.angular.z = 0.5
        else:
            twist.linear.x = 0.3
            twist.angular.z = 0.0
            
        self.publisher.publish(twist)
"""
    
    result = await client.explain_code(
        code_to_explain,
        language="python",
        detail_level="detailed",
    )
    
    print(f"\n📖 Explanation:")
    print(result['explanation'])
    
    print(f"\n📦 Dependencies: {', '.join(result['dependencies'])}")
    
    return result


# ============================================================================
# Example 8: Refactor Code
# ============================================================================


async def example_code_refactoring():
    """Refactor existing code."""
    print("\n" + "="*80)
    print("Example 8: Refactor Code")
    print("="*80)
    
    client = CodeGenerationClient()
    
    # Code to refactor
    original_code = """
import rclpy
from rclpy.node import Node

def main():
    rclpy.init()
    node = Node('test')
    # Bad: hardcoded values, no error handling, poor structure
    pub = node.create_publisher(Twist, 'cmd_vel', 10)
    msg = Twist()
    msg.linear.x = 0.5
    pub.publish(msg)
    rclpy.spin(node)
"""
    
    result = await client.refactor_code(
        original_code,
        goals=[
            "improve error handling",
            "add configuration parameters",
            "enhance code structure",
            "add documentation",
        ],
        language="python",
    )
    
    print(f"\n✅ Refactored Code:")
    print(result['refactored_code'])
    
    print(f"\n📊 Complexity Improvement:")
    print(f"  Before: {result['before_complexity']:.2f}/10")
    print(f"  After:  {result['after_complexity']:.2f}/10")
    print(f"  Change: {result['before_complexity'] - result['after_complexity']:.2f}")
    
    print(f"\n📝 Changes Made:")
    for change in result['changes_made']:
        print(f"  - {change}")
    
    return result


# ============================================================================
# Example 9: Get Available Templates
# ============================================================================


async def example_list_templates():
    """List available code templates."""
    print("\n" + "="*80)
    print("Example 9: Available Code Templates")
    print("="*80)
    
    client = CodeGenerationClient()
    
    result = await client.get_templates()
    
    print(f"\n📚 Available Templates ({len(result['templates'])} total):")
    for template in result['templates']:
        print(f"\n  {template['name']}")
        print(f"    ID: {template['id']}")
        print(f"    Framework: {template['framework']}")
        print(f"    Description: {template['description']}")
    
    return result


# ============================================================================
# Example 10: Batch Code Generation
# ============================================================================


async def example_batch_generation():
    """Generate multiple code files in batch."""
    print("\n" + "="*80)
    print("Example 10: Batch Code Generation")
    print("="*80)
    
    client = CodeGenerationClient()
    
    # Define multiple code generation tasks
    tasks = [
        {
            "name": "publisher_node.py",
            "description": "ROS2 node that publishes robot state",
            "template": "ros2_publisher",
        },
        {
            "name": "subscriber_node.py",
            "description": "ROS2 node that subscribes to sensor data",
            "template": "ros2_subscriber",
        },
        {
            "name": "service_node.py",
            "description": "ROS2 service for robot control",
            "template": "ros2_service",
        },
    ]
    
    # Generate all in parallel
    results = await asyncio.gather(*[
        client.generate_code(
            description=task["description"],
            template=task["template"],
        )
        for task in tasks
    ])
    
    print(f"\n✅ Generated {len(results)} files:")
    for task, result in zip(tasks, results):
        print(f"\n  📄 {task['name']} ({result['estimated_lines']} lines)")
        print(f"     Complexity: {result['complexity_score']:.2f}/10")
    
    return results


# ============================================================================
# Main Execution
# ============================================================================


async def run_all_examples():
    """Run all examples."""
    examples = [
        ("ROS2 Publisher", example_ros2_publisher),
        ("Navigation Controller", example_navigation_controller),
        ("Object Detection", example_object_detection),
        ("Grasp Planning", example_grasp_planning),
        ("Launch File", example_launch_file),
        ("Code Validation", example_code_validation),
        ("Code Explanation", example_code_explanation),
        ("Code Refactoring", example_code_refactoring),
        ("List Templates", example_list_templates),
        ("Batch Generation", example_batch_generation),
    ]
    
    print("\n" + "="*80)
    print("AI-Powered Code Generation Examples")
    print("="*80)
    print("\nThese examples demonstrate the code generation API capabilities.")
    print("Make sure the API server is running at:", API_BASE_URL)
    print("\n")
    
    for i, (name, example_func) in enumerate(examples, 1):
        try:
            await example_func()
        except Exception as e:
            print(f"\n❌ Example {i} ({name}) failed: {e}")
        
        # Pause between examples
        await asyncio.sleep(1)
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    # Run a specific example or all examples
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_ros2_publisher,
            example_navigation_controller,
            example_object_detection,
            example_grasp_planning,
            example_launch_file,
            example_code_validation,
            example_code_explanation,
            example_code_refactoring,
            example_list_templates,
            example_batch_generation,
        ]
        
        if 1 <= example_num <= len(examples):
            asyncio.run(examples[example_num - 1]())
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        # Run all examples
        asyncio.run(run_all_examples())
