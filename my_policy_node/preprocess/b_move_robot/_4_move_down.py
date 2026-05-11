#!/usr/bin/env python3
#
# _4_move_down.py
# Move end effector down vertically along Z axis.
# Used after centering and orientation alignment to approach the port.
#
# WHAT THIS DOES:
#   1. Subscribes to /aic_controller/controller_state for current Z position
#   2. Sends Cartesian velocity (vz < 0) to move down
#   3. Stops when reaching target Z or when port fills certain image size
#
# ASSUMPTIONS:
#   - Port is already centered in camera (use _2_move_to_port.py)
#   - Orientation is aligned (use _3_allign_ort.py)
#   - Controller is in CARTESIAN mode
#
# HOW TO RUN:
#   Terminal 1: camera stream
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sfp
#
#   Terminal 2: move down
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_4_move_down.py --port sfp --target-z 0.05
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
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode, ControllerState
from aic_control_interfaces.srv import ChangeTargetMode


class MoveDown(Node):
    """
    Move end effector down along Z axis using Cartesian velocity control.
    Stops at target Z position or when detection indicates close approach.
    """

    def __init__(self, port_type: str = "sfp", target_z: float = 0.05, use_detection: bool = True):
        super().__init__("move_down")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.target_z = target_z  # Target Z position in meters
        self.use_detection = use_detection  # Whether to use detection for stop condition

        self.get_logger().info(f"MoveDown: looking for {self.target_class}")
        self.get_logger().info(f"  Target Z: {self.target_z}m")

        # Detection state
        self.latest_detection = None
        self.detection_lock = threading.Lock()

        # Controller state (for current Z position)
        self.current_z = None
        self.controller_lock = threading.Lock()

        # Subscribers
        self.create_subscription(String, "/port_detection", self.detection_callback, 10)
        self.create_subscription(ControllerState, "/aic_controller/controller_state", self.controller_state_callback, 10)

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
        self.max_linear_vel = 0.05  # m/s - slow and steady for downward movement
        self.approach_vel = -0.02   # m/s - downward velocity (negative Z)
        self.z_threshold = 0.01     # meters - stop when within this of target
        self.port_size_threshold = 300  # pixels - stop when port is this wide (image-based stop)

        # Timer for control loop at 100Hz
        self.timer = self.create_timer(1.0/100.0, self.control_loop)

        # Throttle counter for logging
        self.log_counter = 0
        self.log_interval = 100  # Log every 100 iterations (1 second at 100Hz)

        self.get_logger().info("MoveDown: initialized")
        self.get_logger().info(f"  Approach velocity: {self.approach_vel} m/s")
        self.get_logger().info(f"  Z threshold: {self.z_threshold}m")
        self.get_logger().info(f"  Use detection stop: {self.use_detection}")
        if self.use_detection:
            self.get_logger().info(f"  Port size threshold: {self.port_size_threshold}px")

    def detection_callback(self, msg):
        """Receive detection from camera stream."""
        try:
            detection = json.loads(msg.data)
            with self.detection_lock:
                self.latest_detection = detection
        except Exception as e:
            self.get_logger().error(f"Detection parse error: {e}")

    def controller_state_callback(self, msg):
        """Receive controller state for current Z position."""
        with self.controller_lock:
            # Z position is typically the 3rd element of position
            if hasattr(msg, 'current_state') and hasattr(msg.current_state, 'position'):
                self.current_z = msg.current_state.position.z
            elif hasattr(msg, 'position'):
                self.current_z = msg.position.z
            else:
                # Try to extract from the message structure
                # ControllerState may have different field names
                pass

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
        msg.header.frame_id = "base_link"
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
        """Main control loop - move down until target reached."""
        # Check Z position stop condition
        with self.controller_lock:
            current_z = self.current_z

        if current_z is not None:
            z_error = current_z - self.target_z

            if abs(z_error) < self.z_threshold:
                self.get_logger().info(f"✓ Target Z reached: {current_z:.4f}m (target: {self.target_z}m)")
                self.send_cartesian_velocity(0.0, 0.0, 0.0)
                return

        # Check detection-based stop condition
        if self.use_detection:
            with self.detection_lock:
                detection = self.latest_detection

            if detection is not None and detection.get("detected", False):
                port_type = detection.get("port_type", "")
                if port_type == self.target_class:
                    # Get port size from detection
                    # Detection should include img_width, img_height
                    # We can estimate distance from port size in image
                    img_w = detection.get("img_width", 640)
                    img_h = detection.get("img_height", 480)
                    cx = detection.get("cx", 0)
                    cy = detection.get("cy", 0)

                    # Calculate port size as percentage of image
                    # When port is large in image, we're close
                    port_half_w = abs(cx - img_w/2) * 2  # Approximate width from center offset
                    port_half_h = abs(cy - img_h/2) * 2  # Approximate height from center offset
                    port_size = max(port_half_w, port_half_h)

                    # If port is very large in image, we're too close
                    if port_size > self.port_size_threshold:
                        self.get_logger().info(f"✓ Close approach detected: port size {port_size:.0f}px > {self.port_size_threshold}px")
                        self.send_cartesian_velocity(0.0, 0.0, 0.0)
                        return

                    self.get_logger().debug(f"Port size: {port_size:.0f}px, continuing descent")

        # Continue moving down
        vz = self.approach_vel
        self.log_counter += 1
        if self.log_counter >= self.log_interval:
            self.get_logger().info(f"Moving down: vz={vz:.3f}m/s")
            self.log_counter = 0
        self.send_cartesian_velocity(0.0, 0.0, vz)

    def run(self):
        """Set up and run."""
        if not self.set_cartesian_mode():
            self.get_logger().error("Cannot proceed without CARTESIAN mode")
            return False

        self.get_logger().info(f"Moving down to Z={self.target_z}m... Press Ctrl+C to stop")
        rclpy.spin(self)
        return True


def main():
    parser = argparse.ArgumentParser(description="Move end effector down along Z axis")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp", help="Port type (sfp or sc)")
    parser.add_argument("--target-z", type=float, default=0.05, help="Target Z position in meters (default: 0.05)")
    parser.add_argument("--no-detection", action="store_true", help="Disable detection-based stop (only use Z position)")
    args = parser.parse_args()

    rclpy.init()
    node = MoveDown(port_type=args.port, target_z=args.target_z, use_detection=not args.no_detection)

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