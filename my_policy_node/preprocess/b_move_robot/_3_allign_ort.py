#!/usr/bin/env python3
#
# _3_allign_ort.py
# Align end effector orientation with port orientation.
# Rotates wrist_3_joint (joint 6) to match the port's angle.
#
# WHAT THIS DOES:
#   1. Subscribes to /port_detection topic for port orientation
#   2. Computes angular velocity from orientation error
#   3. Sends JointMotionUpdate with velocity on wrist_3_joint only
#   4. All other joints stay at zero velocity
#
# ASSUMPTIONS:
#   - Port is already centered in camera (use _2_move_to_port.py first)
#   - Detection topic is being published (by _1_camera_stream.py)
#
# HOW TO RUN:
#   Terminal 1: camera stream
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sfp
#
#   Terminal 2: orientation alignment
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_3_allign_ort.py --port sfp
#

import rclpy
from rclpy.node import Node
import numpy as np
import json
import threading
import argparse
import time

from std_msgs.msg import String
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


class PIDController:
    """Simple PID controller for single axis."""

    def __init__(self, kp, ki, kd, max_output, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, error, current_time):
        """Compute PID output for given error."""
        if self.prev_time is None:
            dt = 0.01
        else:
            dt = current_time - self.prev_time

        # Proportional
        p_term = self.kp * error

        # Integral (with anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        # Derivative
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative

        self.prev_error = error
        self.prev_time = current_time

        output = p_term + i_term + d_term
        return np.clip(output, -self.max_output, self.max_output)

    def reset(self):
        """Reset PID state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None


class AlignOrientation(Node):
    """
    Orientation alignment using angular velocity.
    Rotates end effector around Z axis to match port orientation.
    """

    def __init__(self, port_type: str = "sfp"):
        super().__init__("align_orientation")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.get_logger().info(f"AlignOrientation: looking for {self.target_class}")

        # Detection state
        self.latest_detection = None
        self.detection_lock = threading.Lock()

        # Subscribers
        self.create_subscription(String, "/port_detection", self.detection_callback, 10)

        # Publisher for joint velocity commands
        self.motion_pub = self.create_publisher(
            JointMotionUpdate, "/aic_controller/joint_commands", 10
        )

        # Client to change target mode
        self.change_mode_client = self.create_client(
            ChangeTargetMode,
            "/aic_controller/change_target_mode"
        )

        # Wait for publisher and service
        while self.motion_pub.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscriber to '/aic_controller/joint_commands'...")
            time.sleep(0.5)

        while not self.change_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service '/aic_controller/change_target_mode'...")
            time.sleep(0.5)

        # Control parameters
        self.angle_threshold_rad = np.deg2rad(2)  # 2 degrees - when to stop
        self.max_angular_vel = 0.5  # rad/s - max angular velocity

        # Orientation offset: angle between camera view and end effector
        # If port appears rotated 90° in camera but end effector should be at 0°,
        # set this offset to rotate the reference frame.
        # Positive = rotate CCW, Negative = rotate CW
        self.orientation_offset = 0.0  # radians - tune this based on camera mounting

        # Sign for rotation direction
        # +1.0: positive orientation error → positive ωz (CCW)
        # -1.0: positive orientation error → negative ωz (CW)
        # Tune based on camera mounting and coordinate conventions
        self.sign_error_to_omega = -1.0  # May need to flip based on setup

        # PID controller for orientation
        self.pid_orientation = PIDController(
            kp=0.5,          # Start conservative
            ki=0.01,         # Small integral for steady-state error
            kd=0.02,         # Small derivative for damping
            max_output=self.max_angular_vel,
            integral_limit=0.5
        )

        # Timer for control loop at 100Hz
        self.timer = self.create_timer(1.0/100.0, self.control_loop)

        self.get_logger().info("AlignOrientation: initialized")
        self.get_logger().info(f"  PID: kp=0.5, ki=0.01, kd=0.02")
        self.get_logger().info(f"  Angle threshold: {np.rad2deg(self.angle_threshold_rad):.1f}° (~2°)")
        self.get_logger().info(f"  Max angular velocity: {self.max_angular_vel} rad/s")
        self.get_logger().info(f"  Orientation offset: {np.rad2deg(self.orientation_offset):.1f}°")
        self.get_logger().info("  Subscribing to: /port_detection")
        self.get_logger().info("  Publishing to: /aic_controller/joint_commands (wrist_3 only)")

    def detection_callback(self, msg):
        """Receive detection from camera stream."""
        try:
            detection = json.loads(msg.data)
            with self.detection_lock:
                self.latest_detection = detection
        except Exception as e:
            self.get_logger().error(f"Detection parse error: {e}")

    def set_joint_mode(self):
        """Set controller to JOINT mode."""
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_JOINT

        self.get_logger().info("Setting control mode to JOINT...")
        future = self.change_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result and result.success:
            self.get_logger().info("Controller set to JOINT mode")
            time.sleep(0.5)
            return True
        else:
            self.get_logger().error("Failed to set JOINT mode")
            return False

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def send_wrist3_velocity(self, vel):
        """Send velocity on wrist_3_joint only; all other joints zero."""
        msg = JointMotionUpdate()
        msg.target_state.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, vel]
        msg.target_stiffness = [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
        msg.target_damping = [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        self.motion_pub.publish(msg)

    def control_loop(self):
        """Main control loop - called at 100Hz."""
        with self.detection_lock:
            detection = self.latest_detection

        current_time = time.time()

        if detection is None or not detection.get("detected", False):
            # No detection - stop and reset PID
            self.pid_orientation.reset()
            self.send_wrist3_velocity(0.0)
            return

        # Filter by port type
        port_type = detection.get("port_type", "")
        if port_type != self.target_class:
            self.get_logger().debug(f"Ignoring {port_type}, looking for {self.target_class}")
            self.send_wrist3_velocity(0.0)
            return

        # Get orientation from detection
        orientation_rad = detection.get("orientation", 0.0)
        confidence = detection.get("confidence", 0.0)

        # Apply offset (camera mounting correction)
        target_orientation = self.normalize_angle(orientation_rad + self.orientation_offset)

        # Error: how much to rotate to align
        # We want to rotate so that orientation becomes 0 (aligned)
        error_rad = self.normalize_angle(target_orientation)

        # Check if aligned
        if abs(error_rad) < self.angle_threshold_rad:
            self.get_logger().info(f"✓ Orientation aligned! error={np.rad2deg(error_rad):.1f}°")
            self.pid_orientation.reset()
            self.send_wrist3_velocity(0.0)
            return

        # PID control for orientation
        wrist3_vel = self.sign_error_to_omega * self.pid_orientation.compute(error_rad, current_time)

        # Log status
        self.get_logger().info(
            f"ORIENTATION: angle={np.rad2deg(orientation_rad):.1f}° | "
            f"error={np.rad2deg(error_rad):+.1f}° | "
            f"wrist3_vel={wrist3_vel:+.3f} rad/s | "
            f"conf={confidence:.2f}"
        )

        # Rotate wrist_3_joint only
        self.send_wrist3_velocity(wrist3_vel)

    def run(self):
        """Set up and run."""
        if not self.set_joint_mode():
            self.get_logger().error("Cannot proceed without JOINT mode")
            return False

        self.get_logger().info("Waiting for detection...")

        start = time.time()
        while self.latest_detection is None and (time.time() - start) < 10:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.latest_detection is None:
            self.get_logger().error("No detection received")
            return False

        self.get_logger().info("Starting orientation alignment... Press Ctrl+C to stop")
        rclpy.spin(self)
        return True


def main():
    parser = argparse.ArgumentParser(description="Align end effector orientation with port")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp", help="Port type to align with (sfp or sc)")
    args = parser.parse_args()

    rclpy.init()
    node = AlignOrientation(port_type=args.port)

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted - stopping")
        node.send_wrist3_velocity(0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()