#!/usr/bin/env python3
#
# _3_1_align_port.py
# Align port orientation using keyhole/feature detection.
# Detects two key points on the port and rotates until they're horizontal.
#
# APPROACH:
#   1. Detect port mask from YOLO segmentation
#   2. Find the two keyholes/features in the mask using contour analysis
#   3. Compute the angle between keyhole centers
#   4. Rotate wrist_3_joint until keyholes are horizontally aligned
#
# WHY THIS APPROACH:
#   - More robust than PCA orientation (which can be noisy)
#   - Directly measures what matters: keyhole alignment for insertion
#   - Works for SFP ports (two side keyholes) and SC ports (center keyhole + reference)
#
# ASSUMPTIONS:
#   - Port is already centered in camera (use _2_move_to_port.py first)
#   - Detection topic is being published (by _1_camera_stream.py)
#   - Controller is in JOINT mode for wrist rotation
#
# HOW TO RUN:
#   Terminal 1: camera stream
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sfp
#
#   Terminal 2: alignment
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_3_1_align_port.py --port sfp
#

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import json
import threading
import argparse
import time

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


class AlignPortKeyholes(Node):
    """
    Align port by detecting keyholes/features and rotating until horizontal.
    """

    def __init__(self, port_type: str = "sfp"):
        super().__init__("align_port_keyholes")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.get_logger().info(f"AlignPortKeyholes: looking for {self.target_class}")

        # Detection state
        self.latest_detection = None
        self.detection_lock = threading.Lock()

        # Joint state
        self.current_joints = None
        self.joint_lock = threading.Lock()

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Subscribers
        self.create_subscription(String, "/port_detection", self.detection_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)

        # Publisher
        self.motion_pub = self.create_publisher(
            JointMotionUpdate, "/aic_controller/joint_commands", 10
        )

        # Service client
        self.change_mode_client = self.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )

        # Wait for connections
        while self.motion_pub.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscriber to '/aic_controller/joint_commands'...")
            time.sleep(0.5)

        while not self.change_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service '/aic_controller/change_target_mode'...")
            time.sleep(0.5)

        # Parameters
        self.angle_threshold_deg = 2.0          # Stop when |error| < 2°
        self.max_iterations = 20                 # Max correction attempts
        self.settle_time_sec = 1.0               # Wait after each command
        self.n_samples = 5                       # Readings to average per iteration

        # For SFP: keyholes are on left/right sides
        # For SC: keyhole is in center, use width as reference
        self.keyhole_detection_mode = "side" if port_type == "sfp" else "center"

        self.get_logger().info(f"  Port type: {port_type}")
        self.get_logger().info(f"  Detection mode: {self.keyhole_detection_mode}")
        self.get_logger().info(f"  Angle threshold: {self.angle_threshold_deg}°")
        self.get_logger().info(f"  Max iterations: {self.max_iterations}")

    def detection_callback(self, msg):
        try:
            detection = json.loads(msg.data)
            with self.detection_lock:
                self.latest_detection = detection
        except Exception as e:
            self.get_logger().error(f"Detection parse error: {e}")

    def joint_state_callback(self, msg: JointState):
        with self.joint_lock:
            self.current_joints = msg

    def set_joint_mode(self):
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
        self.get_logger().error("Failed to set JOINT mode")
        return False

    def get_current_joint_positions(self):
        """Return all 6 joint positions in order, or None."""
        with self.joint_lock:
            msg = self.current_joints
        if msg is None:
            return None
        pos_map = dict(zip(msg.name, msg.position))
        return [pos_map.get(name, 0.0) for name in self.joint_names]

    def find_keyholes_from_mask(self, mask_xy):
        """
        Find two key reference points on the port mask.
        These represent where the keyholes/features are located.

        For SFP ports: keyholes are typically on opposite sides.
        For SC ports: keyhole in center with reference points.

        Returns: (point1, point2) as (x, y) tuples representing key reference points.
                 The line between them should be horizontal when aligned.
        """
        if mask_xy is None or len(mask_xy) < 10:
            return None

        points = np.array(mask_xy, dtype=np.float32)

        # Get bounding box
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Find extreme points in each region
        # Left region (left 30%)
        left_region = points[points[:, 0] < center_x - (x_max - x_min) * 0.2]
        # Right region (right 30%)
        right_region = points[points[:, 0] > center_x + (x_max - x_min) * 0.2]
        # Top region (top 30%)
        top_region = points[points[:, 1] < center_y - (y_max - y_min) * 0.2]
        # Bottom region (bottom 30%)
        bottom_region = points[points[:, 1] > center_y + (y_max - y_min) * 0.2]

        # Find centroid of each region
        centroids = {}

        if len(left_region) > 3:
            centroids['left'] = np.mean(left_region, axis=0)
        if len(right_region) > 3:
            centroids['right'] = np.mean(right_region, axis=0)
        if len(top_region) > 3:
            centroids['top'] = np.mean(top_region, axis=0)
        if len(bottom_region) > 3:
            centroids['bottom'] = np.mean(bottom_region, axis=0)

        # Determine which two regions to use based on port shape
        width = x_max - x_min
        height = y_max - y_min

        # Prefer horizontal alignment (left/right) if width > height
        # Prefer vertical alignment (top/bottom) if height > width
        if width > height and 'left' in centroids and 'right' in centroids:
            # Use left-right keyholes (horizontal alignment)
            return tuple(centroids['left']), tuple(centroids['right'])
        elif height > width and 'top' in centroids and 'bottom' in centroids:
            # Use top-bottom keyholes (vertical alignment)
            return tuple(centroids['top']), tuple(centroids['bottom'])
        elif 'left' in centroids and 'right' in centroids:
            # Fallback to left-right
            return tuple(centroids['left']), tuple(centroids['right'])
        elif 'top' in centroids and 'bottom' in centroids:
            # Fallback to top-bottom
            return tuple(centroids['top']), tuple(centroids['bottom'])
        else:
            # Last resort: use extreme points
            leftmost_idx = np.argmin(points[:, 0])
            rightmost_idx = np.argmax(points[:, 0])
            return tuple(points[leftmost_idx]), tuple(points[rightmost_idx])

    def compute_alignment_angle(self, point1, point2):
        """
        Compute angle needed to align two keyhole points horizontally.

        Returns angle in radians. Positive = CW rotation needed to align.
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]

        # Angle of line connecting the two points
        # dy > 0 means point2 is below point1 (need to rotate CW to align)
        # dy < 0 means point2 is above point1 (need to rotate CCW to align)
        angle = np.arctan2(dy, dx)

        return angle

    def collect_alignment_readings(self, n: int, timeout_sec: float = 5.0):
        """
        Collect n readings of alignment angle from keyholes.
        Returns median angle in radians, or None on failure.
        """
        readings = []
        deadline = time.time() + timeout_sec

        while len(readings) < n:
            if time.time() > deadline:
                break

            rclpy.spin_once(self, timeout_sec=0.05)

            with self.detection_lock:
                det = self.latest_detection

            if det is None or not det.get("detected", False):
                continue

            if det.get("port_type", "") != self.target_class:
                continue

            # Try keyhole detection from polygon first
            polygon = det.get("polygon", None)
            if polygon is not None:
                keyholes = self.find_keyholes_from_mask(polygon)
                if keyholes is not None:
                    pt1, pt2 = keyholes
                    angle = self.compute_alignment_angle(pt1, pt2)
                    readings.append(angle)
                    self.get_logger().debug(
                        f"Keyholes: pt1=({pt1[0]:.0f},{pt1[1]:.0f}) pt2=({pt2[0]:.0f},{pt2[1]:.0f}) → angle={np.rad2deg(angle):.1f}°"
                    )
                    continue
                else:
                    self.get_logger().debug("Keyhole detection failed, using PCA fallback")

            # Fallback: use orientation from detection (PCA)
            orientation = det.get("orientation", 0.0)
            readings.append(orientation)
            self.get_logger().debug(f"Using PCA orientation: {np.rad2deg(orientation):.2f}°")

        if not readings:
            return None

        return float(np.median(readings))

    def send_wrist_position(self, wrist_3_angle: float):
        """Send position command for wrist_3_joint only."""
        positions = self.get_current_joint_positions()
        if positions is None:
            self.get_logger().error("No joint state received")
            return False

        # Keep all joints the same except wrist_3
        positions[5] = wrist_3_angle

        target_state = JointTrajectoryPoint()
        target_state.positions = positions
        target_state.velocities = [0.0] * 6
        target_state.accelerations = [0.0] * 6
        target_state.time_from_start.sec = int(self.settle_time_sec)
        target_state.time_from_start.nanosec = 0

        msg = JointMotionUpdate()
        msg.target_state = target_state
        msg.target_stiffness = [85.0] * 6
        msg.target_damping = [75.0] * 6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        msg.target_feedforward_torque = [0.0] * 6

        self.motion_pub.publish(msg)
        return True

    def run(self):
        """Main alignment loop."""
        if not self.set_joint_mode():
            self.get_logger().error("Cannot proceed without JOINT mode")
            return False

        # Wait for first detection
        self.get_logger().info("Waiting for port detection...")
        deadline = time.time() + 10.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self.detection_lock:
                det = self.latest_detection
            if det and det.get("detected") and det.get("port_type") == self.target_class:
                break
        else:
            self.get_logger().error("No detection received")
            return False

        self.get_logger().info("Starting alignment loop...")

        for iteration in range(self.max_iterations):
            # Collect orientation readings
            median_angle = self.collect_alignment_readings(self.n_samples, timeout_sec=3.0)

            if median_angle is None:
                self.get_logger().warn(f"Iter {iteration}: No valid readings")
                continue

            error_deg = np.rad2deg(median_angle)

            # Log detection method
            with self.detection_lock:
                det = self.latest_detection
            method = "keyholes" if (det and det.get("polygon")) else "PCA"
            self.get_logger().info(f"Iter {iteration}: Angle error = {error_deg:+.2f}° (method: {method})")

            # Check if aligned
            if abs(error_deg) < self.angle_threshold_deg:
                self.get_logger().info(f"✓ Port aligned! Final error: {error_deg:+.2f}°")
                return True

            # Get current wrist_3 position
            current_positions = self.get_current_joint_positions()
            if current_positions is None:
                self.get_logger().error("No joint state")
                return False

            current_wrist_3 = current_positions[5]

            # Compute correction
            # Sign: if image shows CCW rotation, we need CW correction
            correction = -median_angle * 0.5  # 50% of error (conservative)

            # Limit correction per step
            max_correction = np.deg2rad(10)  # Max 10° per step
            correction = np.clip(correction, -max_correction, max_correction)

            target_wrist_3 = current_wrist_3 + correction

            self.get_logger().info(
                f"  Rotating wrist_3: {np.rad2deg(current_wrist_3):.1f}° → {np.rad2deg(target_wrist_3):.1f}°"
            )

            # Send command
            if not self.send_wrist_position(target_wrist_3):
                self.get_logger().error("Failed to send position command")
                return False

            # Wait for settle
            time.sleep(self.settle_time_sec + 0.5)

        self.get_logger().warn(f"Max iterations ({self.max_iterations}) reached without convergence")
        return False


def main():
    parser = argparse.ArgumentParser(description="Align port using keyhole/feature detection")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp", help="Port type")
    args = parser.parse_args()

    rclpy.init()
    node = AlignPortKeyholes(port_type=args.port)

    try:
        success = node.run()
        if success:
            node.get_logger().info("Alignment complete!")
        else:
            node.get_logger().error("Alignment failed")
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()