#!/usr/bin/env python3
#
# _5_1_micro_alignment.py
# Position port so CONNECTOR TIP (fixed on robot) aligns with PORT TIP.
# The connector is visible in the camera at a fixed position.
#
# WHAT THIS DOES:
#   1. Subscribes to /port_detection topic for port position
#   2. Computes target: connector position + port tip offset
#   3. Moves robot so port tip aligns with connector tip
#   4. Uses Cartesian velocity control (controller handles IK)
#
# CONNECTOR POSITION:
#   The connector (attached to robot) appears in camera at a fixed position.
#   For example: (img_w/2, img_h × 0.37) — horizontally centered, 37% from top.
#   This is the TARGET position where the PORT TIP should be.
#
# PORT TIP OFFSET:
#   The port tip is offset from the port center (from YOLO detection).
#   For SFP: tip is above center (negative Y offset in image).
#   For SC: different geometry, needs calibration.
#
# ASSUMPTIONS:
#   - Port is already centered (use _2_move_to_port.py first)
#   - Port orientation is aligned (use _3_2_align_port_edge.py first)
#   - Detection topic is being published (by _1_camera_stream.py)
#   - Controller is in CARTESIAN mode
#
# HOW TO RUN:
#   Terminal 1: camera stream
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sfp --camera center
#
#   Terminal 2: micro alignment
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_5_1_micro_alignment.py --port sfp --camera center
#

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import numpy as np
import json
import threading
import argparse
import time

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Wrench, Vector3
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


# Connector position ratio in camera image.
# The connector is fixed on the robot and visible in camera.
# Connector is BELOW center (larger y = lower in image).
# 1/6 from middle to bottom: 0.5 + 0.5/6 ≈ 0.58
CONNECTOR_POSITION = {
    "center": {
        "x_ratio": 0.5,    # Horizontally centered
        "y_ratio": 0.605,   # Below center (adjust for your setup)
    },
    "left": {
        "x_ratio": 0.5,
        "y_ratio": 0.58,
    },
    "right": {
        "x_ratio": 0.5,
        "y_ratio": 0.58,
    },
}

# Port tip offset from port center (as ratio of bbox dimensions).
# Positive Y = downward in image, Negative Y = upward.
# For SFP: tip is above center (negative Y).
PORT_TIP_OFFSET = {
    "sfp": {
        "x_ratio": 0.0,     # Tip is horizontally centered
        "y_ratio": -0.2,   # Tip is 20% above center (adjust for actual geometry)
    },
    "sc": {
        "x_ratio": 0.0,
        "y_ratio": 0.0,    # Needs calibration
    },
}


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
        if self.prev_time is None:
            dt = 0.01
        else:
            dt = current_time - self.prev_time

        p_term = self.kp * error

        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

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


class MicroAlignment(Node):
    """
    Micro alignment: position port so CONNECTOR TIP aligns with PORT TIP.

    Target position = connector position in image
    Port tip = port center + port tip offset
    """

    def __init__(self, port_type: str = "sfp", camera: str = "center"):
        super().__init__("micro_alignment")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.camera = camera.lower()

        if self.camera not in ["center", "left", "right"]:
            self.get_logger().warn(f"Invalid camera '{self.camera}', defaulting to 'center'")
            self.camera = "center"

        # Get connector position for this camera
        if self.camera not in CONNECTOR_POSITION:
            self.get_logger().error(f"Unknown camera: {self.camera}")
            raise ValueError(f"Unknown camera: {self.camera}")

        self.connector_x_ratio = CONNECTOR_POSITION[self.camera]["x_ratio"]
        self.connector_y_ratio = CONNECTOR_POSITION[self.camera]["y_ratio"]

        # Get port tip offset for this port type
        if self.port_type not in PORT_TIP_OFFSET:
            self.get_logger().error(f"Unknown port type: {self.port_type}")
            raise ValueError(f"Unknown port type: {self.port_type}")

        self.tip_x_ratio = PORT_TIP_OFFSET[self.port_type]["x_ratio"]
        self.tip_y_ratio = PORT_TIP_OFFSET[self.port_type]["y_ratio"]

        self.get_logger().info(f"MicroAlignment: {self.target_class} on {self.camera.upper()} camera")
        self.get_logger().info(f"  Connector position: ({self.connector_x_ratio:.2f} × width, {self.connector_y_ratio:.2f} × height)")
        self.get_logger().info(f"  Port tip offset: ({self.tip_x_ratio:.2f} × bbox_w, {self.tip_y_ratio:.2f} × bbox_h)")

        # Detection state
        self.latest_detection = None
        self.detection_lock = threading.Lock()
        self.done = False

        # Wrist_3 tracking for camera-relative velocity compensation
        self.current_wrist3 = None

        # Subscribers
        self.create_subscription(String, "/port_detection", self.detection_callback, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

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
        self.center_threshold_px = 10      # pixels - tighter threshold for micro alignment
        self.max_linear_vel = 0.02         # m/s - slower for precision

        # Direction signs (same as _2_move_to_port)
        self.sign_x_to_vy = 1.0
        self.sign_y_to_vx = 1.0

        # PID controllers for X and Y axes
        self.pid_x = PIDController(
            kp=0.0008,      # Lower gain for precision
            ki=0.0,
            kd=0.0,
            max_output=self.max_linear_vel,
            integral_limit=10.0
        )
        self.pid_y = PIDController(
            kp=0.0008,
            ki=0.0,
            kd=0.0,
            max_output=self.max_linear_vel,
            integral_limit=10.0
        )

        # Frame ID for velocity commands
        self.frame_id = "base_link"

        # Timer for control loop at 100Hz
        self.timer = self.create_timer(1.0/100.0, self.control_loop)

        self.get_logger().info("MicroAlignment: initialized")
        self.get_logger().info(f"  Threshold: {self.center_threshold_px}px")
        self.get_logger().info(f"  PID: kp=0.0008, ki=0.0, kd=0.0")

    def _joint_state_cb(self, msg: JointState):
        pos_map = dict(zip(msg.name, msg.position))
        w3 = pos_map.get("wrist_3_joint")
        if w3 is None:
            return
        self.current_wrist3 = w3

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
        twist.angular.x = 0.0
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
        """
        Main control loop.

        Goal: Move robot so PORT TIP aligns with CONNECTOR TIP.

        Calculation:
          - Connector position (fixed in image): (img_w × connector_x_ratio, img_h × connector_y_ratio)
          - Port center (from detection): (cx, cy)
          - Port tip position: (cx + tip_x_ratio × bbox_w, cy + tip_y_ratio × bbox_h)
          - Target for port tip: connector position
          - Error: port_tip - connector_position
        """
        with self.detection_lock:
            detection = self.latest_detection

        current_time = time.time()

        if detection is None or not detection.get("detected", False):
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

        # Filter by camera
        detection_camera = detection.get("camera", "center")
        if detection_camera != self.camera:
            self.get_logger().debug(f"Ignoring {detection_camera} camera, looking for {self.camera}")
            self.send_cartesian_velocity(0.0, 0.0, 0.0)
            return

        # Get detection data
        cx = detection["cx"]
        cy = detection["cy"]
        img_w = detection["img_width"]
        img_h = detection["img_height"]

        # Get bounding box dimensions for tip offset calculation
        bbox = detection.get("bbox", None)
        if bbox is None:
            self.get_logger().warn("No bbox in detection")
            self.send_cartesian_velocity(0.0, 0.0, 0.0)
            return

        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Calculate positions
        # Connector position (fixed in image)
        connector_x = img_w * self.connector_x_ratio
        connector_y = img_h * self.connector_y_ratio

        # Port tip position (offset from port center)
        port_tip_x = cx + self.tip_x_ratio * bbox_w
        port_tip_y = cy + self.tip_y_ratio * bbox_h

        # Error: port tip should be at connector position
        error_x_px = port_tip_x - connector_x  # positive = port tip right of connector
        error_y_px = port_tip_y - connector_y  # positive = port tip below connector

        # Check if aligned
        if abs(error_x_px) < self.center_threshold_px and abs(error_y_px) < self.center_threshold_px:
            self.get_logger().info(f"✓ Aligned! Error: ({error_x_px:+.0f}px, {error_y_px:+.0f}px)")
            self.pid_x.reset()
            self.pid_y.reset()
            self.send_cartesian_velocity(0.0, 0.0, 0.0)
            self.done = True
            return

        # PID control
        pid_x_output = self.pid_x.compute(error_x_px, current_time)
        pid_y_output = self.pid_y.compute(error_y_px, current_time)
        vx = self.sign_y_to_vx * pid_y_output
        vy = self.sign_x_to_vy * pid_x_output
        vz = 0.0

        # Compensate for wrist rotation
        if self.current_wrist3 is not None:
            angle = -self.current_wrist3
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            vx_new = cos_a * vx - sin_a * vy
            vy_new = sin_a * vx + cos_a * vy
            vx, vy = vx_new, vy_new

        self.get_logger().info(
            f"Port tip ({port_tip_x:.0f}, {port_tip_y:.0f}) → Connector ({connector_x:.0f}, {connector_y:.0f}) | "
            f"Error: ({error_x_px:+.0f}px, {error_y_px:+.0f}px) | "
            f"v: ({vx:+.4f}, {vy:+.4f})"
        )

        self.send_cartesian_velocity(vx, vy, vz)

    def run(self):
        """Set up and run."""
        if not self.set_cartesian_mode():
            self.get_logger().error("Cannot proceed without CARTESIAN mode")
            return False

        # Use a dedicated executor to avoid conflicts with other nodes/threads
        executor = SingleThreadedExecutor()
        executor.add_node(self)

        self.get_logger().info("Waiting for detection...")

        start = time.time()
        while self.latest_detection is None and (time.time() - start) < 10:
            executor.spin_once(timeout_sec=0.1)

        if self.latest_detection is None:
            self.get_logger().error("No detection received")
            return False

        self.get_logger().info("Starting micro alignment...")
        self.get_logger().info(f"  Moving port tip to connector position at ({self.connector_x_ratio:.2f}×W, {self.connector_y_ratio:.2f}×H)")

        while rclpy.ok() and not self.done:
            executor.spin_once(timeout_sec=0.1)
        return self.done


def main():
    parser = argparse.ArgumentParser(description="Micro alignment - align connector tip with port tip")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp", help="Port type")
    parser.add_argument("--camera", choices=["center", "left", "right"], default="center", help="Camera to use")
    args = parser.parse_args()

    rclpy.init()
    node = MicroAlignment(port_type=args.port, camera=args.camera)

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