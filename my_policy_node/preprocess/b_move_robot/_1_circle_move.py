#!/usr/bin/env python3
#
# _1_circle_move.py
# Standalone script to make robot trace a circle in XY plane.
#
# WHAT THIS DOES:
#   1. Connects to ROS2
#   2. Publishes pose commands to /aic_controller/pose_commands
#   3. Robot moves in a circular path on XY plane (draws a circle)
#
# HOW TO RUN:
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_1_circle_move.py
#

import rclpy
from rclpy.node import Node
import math
import time

from geometry_msgs.msg import Pose, Point, Quaternion
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode
from std_msgs.msg import Header
import numpy as np


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert euler angles to quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cy * cp * cr + sy * sp * sr
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr
    return q


class CircleMover(Node):
    """
    Node that makes the robot trace a circle in XY plane.
    """

    def __init__(self):
        super().__init__("circle_mover")

        self.pose_pub = self.create_publisher(
            MotionUpdate,
            "/aic_controller/pose_commands",
            10
        )

        # Client to change target mode
        self.change_mode_client = self.create_client(
            ChangeTargetMode,
            "/aic_controller/change_target_mode"
        )

        self.get_logger().info("CircleMover: initialized")
        self.get_logger().info("  Publishing to: /aic_controller/pose_commands")

    def set_cartesian_mode(self):
        """Set controller to CARTESIAN mode."""
        if not self.change_mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("ChangeTargetMode service not available")
            return False

        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_CARTESIAN

        future = self.change_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        result = future.result()
        if result and result.success:
            self.get_logger().info("Controller set to CARTESIAN mode")
            return True
        else:
            self.get_logger().error("Failed to set CARTESIAN mode")
            return False

    def create_motion_update(self, pose: Pose) -> MotionUpdate:
        """Create a MotionUpdate message with default parameters."""
        return MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten(),
            target_damping=np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten(),
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

    def move_to_pose(self, pose: Pose, wait_sec: float = 1.5):
        """Publish a pose command and wait."""
        motion_update = self.create_motion_update(pose)
        self.pose_pub.publish(motion_update)
        self.get_logger().info(
            f"  Moved to: x={pose.position.x:.3f} y={pose.position.y:.3f} z={pose.position.z:.3f}"
        )
        time.sleep(wait_sec)

    def trace_circle(
        self,
        center_x: float = -0.2,
        center_y: float = 0.0,
        radius: float = 0.1,
        height: float = 0.3,
        angle_step_deg: float = 30.0,
        direction: int = 1,  # 1 = counter-clockwise, -1 = clockwise
    ):
        """
        Move robot in a circle on XY plane.

        Args:
            center_x: X center of circle
            center_y: Y center of circle
            radius: Circle radius in meters
            height: Z height (constant)
            angle_step_deg: Degrees per step
            direction: 1 for CCW, -1 for CW
        """
        # Set controller to CARTESIAN mode first
        if not self.set_cartesian_mode():
            self.get_logger().error("Cannot proceed without CARTESIAN mode")
            return

        angle_step_rad = math.radians(angle_step_deg)
        total_steps = int(360.0 / angle_step_deg)

        # Fixed orientation: pointing down
        orientation = euler_to_quaternion(roll=math.pi, pitch=0.0, yaw=0.0)

        self.get_logger().info(f"CircleMover: tracing circle")
        self.get_logger().info(f"  Center: ({center_x}, {center_y})")
        self.get_logger().info(f"  Radius: {radius}m")
        self.get_logger().info(f"  Height: {height}m")
        self.get_logger().info(f"  Steps: {total_steps}")
        self.get_logger().info(f"  Direction: {'CCW' if direction > 0 else 'CW'}")

        # Move in a circle
        for step in range(total_steps + 1):  # +1 to complete the circle
            angle = direction * angle_step_rad * step

            # Circle parametric equations
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            pose = Pose(
                position=Point(x=x, y=y, z=height),
                orientation=orientation,
            )

            self.get_logger().info(f"Step {step}/{total_steps}: angle={math.degrees(angle):.0f}° pos=({x:.3f}, {y:.3f})")
            self.move_to_pose(pose, wait_sec=1.5)

        self.get_logger().info("CircleMover: circle complete!")


def main(args=None):
    rclpy.init(args=args)

    node = CircleMover()

    try:
        # Trace a circle in XY plane
        node.trace_circle(
            center_x=-0.5,
            center_y=0.4,
            radius=0.2,
            height=0.3,
            angle_step_deg=30.0,
            direction=-1,      # -1 = clockwise, 1 = counter-clockwise
        )
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()