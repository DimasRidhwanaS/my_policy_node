#!/usr/bin/env python3
#
# _2_move_to_port.py
# Center port in camera frame using Cartesian velocity control.
# Controller handles inverse kinematics automatically.
# End effector stays perpendicular (angular velocity = 0).
#
# WHAT THIS DOES:
#   1. Subscribes to /port_detection topic for port position
#   2. Computes desired Cartesian velocity from pixel error
#   3. Sends MotionUpdate with velocity (controller handles IK)
#   4. End effector stays perpendicular (angular velocity = 0)
#
# ASSUMPTIONS:
#   - Port is already visible in camera
#   - Detection topic is being published (by _1_camera_stream.py)
#   - Controller is in CARTESIAN mode
#
# HOW TO RUN:
#   Terminal 1: camera stream
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sc
#
#   Terminal 2: movement
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_2_move_to_port.py --port sc
#

import rclpy
from rclpy.node import Node
import numpy as np
import json
import threading
import argparse
import time

from std_msgs.msg import String
from geometry_msgs.msg import Twist, Wrench, Vector3
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


class PIDController:
    """Simple PID controller for single axis."""

    def __init__(self, kp, ki, kd, max_output, integral_limit=50.0):
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
        # Time delta
        if self.prev_time is None:
            dt = 0.01  # Assume 100Hz on first call
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

        # Store for next iteration
        self.prev_error = error
        self.prev_time = current_time

        # Total output (clamped)
        output = p_term + i_term + d_term
        return np.clip(output, -self.max_output, self.max_output)

    def reset(self):
        """Reset PID state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None


class MoveToPort(Node):
    """
    Cartesian velocity control for centering port using PID.
    Controller handles inverse kinematics automatically.
    End effector stays perpendicular (angular velocity = 0).
    """

    def __init__(self, port_type: str = "sfp"):
        super().__init__("move_to_port")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.get_logger().info(f"MoveToPort: looking for {self.target_class}")

        # Detection state
        self.latest_detection = None
        self.detection_lock = threading.Lock()

        # Subscribers
        self.create_subscription(String, "/port_detection", self.detection_callback, 10)

        # Publisher for Cartesian velocity commands
        self.motion_pub = self.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10
        )

        # Client to change target mode
        self.change_mode_client = self.create_client(
            ChangeTargetMode,
            "/aic_controller/change_target_mode"
        )

        # Wait for publisher and service
        while self.motion_pub.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscriber to '/aic_controller/pose_commands'...")
            time.sleep(0.5)

        while not self.change_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service '/aic_controller/change_target_mode'...")
            time.sleep(0.5)

        # Control parameters
        self.center_threshold_px = 15      # pixels - when to stop
        self.max_linear_vel = 0.1          # m/s - max linear velocity

        # Direction signs (tune these based on camera mounting)
        # Image: error_x = horizontal (left/right), error_y = vertical (above/below)
        # Base: +X forward, +Y left, +Z up
        self.sign_x_to_vy = -1.0  # horizontal error → Y velocity
        self.sign_y_to_vx = 1.0  # vertical error → X velocity

        # PID controllers for X and Y axes
        # Start with P-only (no I, no D) for basic tuning
        # Increase kp to make movement visible
        self.pid_x = PIDController(
            kp=0.001,       # Proportional - larger for visible movement
            ki=0.0,         # Integral - DISABLED for now
            kd=0.0,         # Derivative - DISABLED for now
            max_output=self.max_linear_vel,
            integral_limit=10.0  # Small limit for safety
        )
        self.pid_y = PIDController(
            kp=0.001,
            ki=0.0,
            kd=0.0,
            max_output=self.max_linear_vel,
            integral_limit=10.0
        )

        # Frame ID for velocity commands
        self.frame_id = "base_link"

        # Timer for control loop at 100Hz
        self.timer = self.create_timer(1.0/100.0, self.control_loop)

        self.get_logger().info("MoveToPort: initialized with PID control")
        self.get_logger().info("  PID: kp=0.0008, ki=0.00005, kd=0.0002")
        self.get_logger().info("  Subscribing to: /port_detection")
        self.get_logger().info("  Publishing to: /aic_controller/pose_commands")

        self.get_logger().info("MoveToPort: initialized with Cartesian velocity control")
        self.get_logger().info("  Subscribing to: /port_detection")
        self.get_logger().info("  Publishing to: /aic_controller/pose_commands")

    def detection_callback(self, msg):
        """Receive detection from camera stream."""
        try:
            detection = json.loads(msg.data)
            with self.detection_lock:
                self.latest_detection = detection
        except Exception as e:
            self.get_logger().error(f"Detection parse error: {e}")

    def set_cartesian_mode(self):
        """Set controller to CARTESIAN mode."""
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
        else:
            self.get_logger().error("Failed to set CARTESIAN mode")
            return False

    def send_cartesian_velocity(self, vx, vy, vz):
        """Send Cartesian velocity command."""
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = vz
        twist.angular.x = 0.0  # Keep orientation fixed (perpendicular)
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.velocity = twist
        msg.target_stiffness = np.diag([85.0, 85.0, 85.0, 85.0, 85.0, 85.0]).flatten()
        msg.target_damping = np.diag([75.0, 75.0, 75.0, 75.0, 75.0, 75.0]).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0)
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        self.motion_pub.publish(msg)

    def control_loop(self):
        """Main control loop - called at 100Hz with PID control."""
        with self.detection_lock:
            detection = self.latest_detection

        current_time = time.time()

        if detection is None or not detection.get("detected", False):
            # No detection - stop and reset PID
            self.pid_x.reset()
            self.pid_y.reset()
            self.send_cartesian_velocity(0.0, 0.0, 0.0)
            return

        # Filter by port type
        port_type = detection.get("port_type", "")
        if port_type != self.target_class:
            self.get_logger().debug(f"Ignoring {port_type}, looking for {self.target_class}")
            self.send_cartesian_velocity(0.0, 0.0, 0.0)
            return

        # Get detection data
        cx = detection["cx"]
        cy = detection["cy"]
        img_w = detection["img_width"]
        img_h = detection["img_height"]
        confidence = detection["confidence"]

        # Calculate image center and errors
        img_cx = img_w / 2
        img_cy = img_h / 2
        error_x_px = cx - img_cx  # positive = port right of center
        error_y_px = cy - img_cy  # positive = port below center

        # Check if centered
        if abs(error_x_px) < self.center_threshold_px and abs(error_y_px) < self.center_threshold_px:
            self.get_logger().info(f"✓ Port centered!")
            self.pid_x.reset()
            self.pid_y.reset()
            self.send_cartesian_velocity(0.0, 0.0, 0.0)
            return

        # PID control for X axis (forward/backward) based on error_y_px
        pid_x_output = self.pid_x.compute(error_x_px, current_time)
        pid_y_output = self.pid_y.compute(error_y_px, current_time)
        vx = self.sign_y_to_vx * pid_x_output
        vy = self.sign_x_to_vy * pid_y_output
        vz = 0.0  # No Z movement - stays at current height

        # Debug X axis only (pid_y is disabled)
        self.get_logger().info(
            f"X_AXIS: err_x={error_x_px:+.0f}px | pid_out_x={pid_x_output:+.4f} | vx={vx:+.4f} | vy={vy:+.4f}"
        )

        # Send Cartesian velocity (controller handles IK)
        self.send_cartesian_velocity(vx, vy, vz)

    def run(self):
        """Set up and run."""
        if not self.set_cartesian_mode():
            self.get_logger().error("Cannot proceed without CARTESIAN mode")
            return False

        self.get_logger().info("Waiting for detection...")

        start = time.time()
        while self.latest_detection is None and (time.time() - start) < 10:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.latest_detection is None:
            self.get_logger().error("No detection received")
            return False

        self.get_logger().info("Starting Cartesian velocity centering... Press Ctrl+C to stop")
        rclpy.spin(self)
        return True


def main():
    parser = argparse.ArgumentParser(description="Center port in camera using Cartesian velocity control")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp", help="Port type to center on (sfp or sc)")
    args = parser.parse_args()

    rclpy.init()
    node = MoveToPort(port_type=args.port)

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted - stopping")
        node.send_cartesian_velocity(0.0, 0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()