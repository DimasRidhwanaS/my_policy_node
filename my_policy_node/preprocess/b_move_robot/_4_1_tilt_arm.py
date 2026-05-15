#!/usr/bin/env python3
#
# _4_1_tilt_arm.py
# Orient end effector perpendicular to the XY plane (straight down).
# Port tilt is fixed/horizontal — command TCP to point straight down.
#
# WHAT THIS DOES:
#   1. Reads current TCP pose from /observations
#   2. Keeps current XYZ position unchanged
#   3. Sets orientation to straight down (roll=π, pitch=0) preserving current yaw
#   4. Sends one position command and waits to settle
#
# ASSUMPTIONS:
#   - Port is already centered and wrist-aligned (_2_ and _3_ done first)
#   - Port face is horizontal (parallel to ground) — tilt is fixed
#
# HOW TO RUN:
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_4_1_tilt_arm.py
#

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import numpy as np
import math
import time

from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from aic_model_interfaces.msg import Observation
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


def quaternion_to_euler(q: Quaternion):
    """Return (roll, pitch, yaw) in radians from a geometry_msgs Quaternion."""
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert euler angles to geometry_msgs Quaternion."""
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


class TiltArm(Node):

    def __init__(self):
        super().__init__("tilt_arm")

        self.current_tcp_pose = None

        self.create_subscription(Observation, "/observations", self._obs_callback, 10)

        self.pose_pub = self.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10
        )
        self.change_mode_client = self.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )

        while self.pose_pub.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscriber to '/aic_controller/pose_commands'...")
            time.sleep(0.5)

        while not self.change_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service '/aic_controller/change_target_mode'...")
            time.sleep(0.5)

        # How long to wait after sending the command
        self.settle_sec = 2.0

    def _obs_callback(self, msg: Observation):
        self.current_tcp_pose = msg.controller_state.tcp_pose

    def set_cartesian_mode(self):
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_CARTESIAN
        self.get_logger().info("Setting control mode to CARTESIAN...")
        future = self.change_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result and result.success:
            self.get_logger().info("Controller set to CARTESIAN mode")
            time.sleep(0.5)
            return True
        self.get_logger().error("Failed to set CARTESIAN mode")
        return False

    def send_pose_command(self, pose: Pose):
        msg = MotionUpdate(
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
        self.pose_pub.publish(msg)

    def run(self):
        if not self.set_cartesian_mode():
            self.get_logger().error("Cannot proceed without CARTESIAN mode")
            return False

        # Use a dedicated executor to avoid conflicts with other nodes/threads
        executor = SingleThreadedExecutor()
        executor.add_node(self)

        # Wait for first observation
        self.get_logger().info("Waiting for TCP pose from /observations...")
        deadline = time.time() + 10.0
        while self.current_tcp_pose is None and time.time() < deadline:
            executor.spin_once(timeout_sec=0.1)

        if self.current_tcp_pose is None:
            self.get_logger().error("No observation received — aborting")
            return False

        tcp = self.current_tcp_pose
        roll, pitch, yaw = quaternion_to_euler(tcp.orientation)

        self.get_logger().info(
            f"Current TCP: pos=({tcp.position.x:.3f}, {tcp.position.y:.3f}, {tcp.position.z:.3f}) "
            f"rpy=({math.degrees(roll):.1f}°, {math.degrees(pitch):.1f}°, {math.degrees(yaw):.1f}°)"
        )

        # Connector roll offset relative to current roll for perpendicular insertion
        CONNECTOR_ROLL_OFFSET_DEG = -21.0
        target_roll = roll + math.radians(CONNECTOR_ROLL_OFFSET_DEG)

        # Build target orientation: adjusted roll, pitch=0, keep yaw
        target_orientation = euler_to_quaternion(roll=target_roll, pitch=0.0, yaw=yaw)

        target_pose = Pose()
        target_pose.position = tcp.position   # keep current XYZ
        target_pose.orientation = target_orientation

        self.get_logger().info(
            f"Current roll={math.degrees(roll):.1f}° → target roll={math.degrees(target_roll):.1f}° "
            f"(+{CONNECTOR_ROLL_OFFSET_DEG}°), pitch=0°, yaw={math.degrees(yaw):.1f}°"
        )

        self.send_pose_command(target_pose)
        time.sleep(self.settle_sec)

        self.get_logger().info("✓ Tilt corrected — TCP now perpendicular to ground")
        return True


def main():
    rclpy.init()
    node = TiltArm()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
