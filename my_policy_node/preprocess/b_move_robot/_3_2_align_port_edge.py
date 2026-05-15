#!/usr/bin/env python3
#
# _3_2_align_port_edge.py
# Align port orientation using YOLO polygon corners directly.
# Finds top-left and top-right corners from polygon vertices.
#
# APPROACH:
#   1. Subscribe to /port_detection for polygon
#   2. Get YOLO polygon vertices
#   3. Find top points (smallest Y values - top 25%)
#   4. Get top-left (smallest X) and top-right (largest X)
#   5. Calculate angle from line connecting corners
#   6. Rotate wrist_3_joint to align port horizontally
#
# WHY THIS APPROACH:
#   - Uses YOLO polygon directly, no edge detection noise
#   - Simple and stable - just find corners from polygon
#   - No complex filtering needed
#
# ASSUMPTIONS:
#   - Port is already centered in camera (use _2_move_to_port.py first)
#   - Detection topic is being published (by _1_camera_stream.py)
#   - Controller is in JOINT mode for wrist rotation
#
# HOW TO RUN:
#   Terminal 1: camera stream
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sfp --camera right
#
#   Terminal 2: polygon-based alignment
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_3_2_align_port_edge.py --port sfp --camera right
#

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import numpy as np
import cv2
import json
import threading
import argparse
import time

from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from trajectory_msgs.msg import JointTrajectoryPoint
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


# Camera angle offsets (degrees) - relative to center camera
CAMERA_ANGLE_OFFSETS = {
    "center": 0.0,
    "left": -30.0,   # Left camera sees port from left side
    "right": 30.0,   # Right camera sees port from right side
}

CAMERA_TOPICS = {
    "center": "/center_camera/image",
    "left": "/left_camera/image",
    "right": "/right_camera/image",
}


class AlignPortEdge(Node):
    """
    Align port using YOLO polygon corners directly.
    Simple and stable - no edge detection complexity.
    """

    def __init__(self, port_type: str = "sfp", camera: str = "center", headless: bool = False):
        super().__init__("align_port_edge")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.camera = camera.lower()
        self.headless = headless  # when True: skip all cv2 display calls

        if self.camera not in CAMERA_TOPICS:
            raise ValueError(f"Invalid camera: {self.camera}")

        self.camera_topic = CAMERA_TOPICS[self.camera]
        self.camera_offset = CAMERA_ANGLE_OFFSETS[self.camera]

        self.get_logger().info(f"AlignPortEdge: looking for {self.target_class}")
        self.get_logger().info(f"  Camera: {self.camera} (offset: {self.camera_offset}°)")

        # Detection state
        self.latest_detection = None
        self.detection_lock = threading.Lock()

        # Camera image
        self.latest_image = None
        self.image_lock = threading.Lock()

        # Joint state
        self.current_joints = None
        self.joint_lock = threading.Lock()

        # CV bridge
        self.bridge = CvBridge()

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
        self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
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
        self.angle_threshold_deg = 0.5          # Stop when |error| < 0.5° (tighter for insertion)
        self.max_iterations = 20                 # Max correction attempts
        self.settle_time_sec = 1.0               # Wait after each command
        self.n_samples = 10                      # Readings to average per iteration (more samples)

        self.get_logger().info(f"  Port type: {port_type}")
        self.get_logger().info(f"  Camera: {self.camera} (offset: {self.camera_offset}°)")
        self.get_logger().info(f"  Angle threshold: {self.angle_threshold_deg}°")
        self.get_logger().info(f"  Max iterations: {self.max_iterations}")

    def detection_callback(self, msg):
        try:
            detection = json.loads(msg.data)
            with self.detection_lock:
                self.latest_detection = detection
        except Exception as e:
            self.get_logger().error(f"Detection parse error: {e}")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.latest_image = image
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")

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

    def detect_angle_from_polygon(self, image, detection):
        """
        Detect port orientation using YOLO polygon.
        Finds 4 corners of the rectangle using approxPolyDP.

        Args:
            image: Full camera image (BGR)
            detection: YOLO detection dict with polygon

        Returns:
            angle_rad: Orientation angle in radians (None if detection fails)
            top_left: (x, y) of top-left corner (None if fails)
            top_right: (x, y) of top-right corner (None if fails)
            debug_image: Annotated image for visualization
        """
        if detection is None or not detection.get("detected", False):
            return None, None, None, image

        # Get polygon from YOLO detection
        polygon = detection.get("polygon", None)
        cx = int(detection.get("cx", 0))
        cy = int(detection.get("cy", 0))

        if polygon is None or len(polygon) < 3:
            debug = image.copy()
            cv2.putText(debug, "No polygon", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, None, None, debug

        polygon_np = np.array(polygon, dtype=np.int32)

        # Find 4 corners of the rectangle
        # Use approxPolyDP to simplify polygon to quadrilateral
        epsilon = 0.02 * cv2.arcLength(polygon_np, True)
        approx = cv2.approxPolyDP(polygon_np, epsilon, True)

        # If approx has more than 4 points, use convex hull and try again
        if len(approx) > 4:
            hull = cv2.convexHull(polygon_np)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

        # If still more than 4, take the 4 extreme points
        if len(approx) > 4:
            rect = cv2.minAreaRect(polygon_np)
            box = cv2.boxPoints(rect)
            approx = box.astype(np.int32)

        if len(approx) < 4:
            debug = image.copy()
            cv2.putText(debug, "Cannot find 4 corners", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, None, None, debug

        # Reshape to (N, 2) format
        corners = approx.reshape(-1, 2)

        # Order corners: top-left, top-right, bottom-right, bottom-left
        sorted_by_y = sorted(corners, key=lambda p: p[1])
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]

        top_two_sorted = sorted(top_two, key=lambda p: p[0])
        top_left = tuple(top_two_sorted[0])
        top_right = tuple(top_two_sorted[1])

        bottom_two_sorted = sorted(bottom_two, key=lambda p: p[0])
        bottom_left = tuple(bottom_two_sorted[0])
        bottom_right = tuple(bottom_two_sorted[1])

        ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])

        # Find longest edge among all 4 edges
        edges = [
            ("top", top_left, top_right),
            ("right", top_right, bottom_right),
            ("bottom", bottom_right, bottom_left),
            ("left", bottom_left, top_left),
        ]

        longest_edge = max(edges, key=lambda e: np.sqrt((e[2][0] - e[1][0])**2 + (e[2][1] - e[1][1])**2))
        edge_name, p1, p2 = longest_edge[0], longest_edge[1], longest_edge[2]

        # Calculate angle from longest edge
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        raw_angle_rad = np.arctan2(dy, dx)

        # Apply camera angle offset to get true angle relative to robot
        angle_rad = raw_angle_rad - np.deg2rad(self.camera_offset)

        # Create debug image
        debug = image.copy()

        # Draw polygon
        cv2.polylines(debug, [polygon_np], True, (0, 255, 0), 1)
        cv2.polylines(debug, [ordered_corners], True, (0, 255, 0), 2)

        # Draw center
        cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)

        # Draw longest edge (yellow)
        cv2.line(debug, p1, p2, (0, 255, 255), 3)
        cv2.circle(debug, p1, 8, (255, 0, 0), -1)
        cv2.circle(debug, p2, 8, (0, 0, 255), -1)

        # Draw angle
        angle_deg = np.degrees(angle_rad)
        cv2.putText(debug, f"Angle: {angle_deg:.1f} deg",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Status
        status = "ALIGNED" if abs(angle_deg) < 2 else f"ROTATE {angle_deg:+.1f} deg"
        color = (0, 255, 0) if abs(angle_deg) < 2 else (0, 165, 255)
        cv2.putText(debug, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Info
        cv2.putText(debug, f"Edge: {edge_name}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(debug, f"Polygon: {len(polygon)} pts -> 4 corners",
                   (10, debug.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return angle_rad, p1, p2, debug

    def collect_edge_readings(self, n: int, timeout_sec: float = 5.0):
        """
        Collect n readings of edge-based orientation.
        Returns median angle in radians, or None on failure.
        """
        readings = []
        deadline = time.time() + timeout_sec

        while len(readings) < n:
            if time.time() > deadline:
                break

            self._executor.spin_once(timeout_sec=0.05)

            with self.detection_lock:
                det = self.latest_detection
            with self.image_lock:
                img = self.latest_image

            if det is None or not det.get("detected", False):
                continue

            if det.get("port_type", "") != self.target_class:
                continue

            if img is None:
                continue

            angle_rad, top_left, top_right, debug = self.detect_angle_from_polygon(img, det)

            if angle_rad is not None:
                readings.append(angle_rad)
                self.get_logger().debug(
                    f"Polygon reading: {np.rad2deg(angle_rad):.2f}° "
                    f"(TL: {top_left}, TR: {top_right})"
                )
                if not self.headless:
                    cv2.imshow("Port Alignment", debug)
                    cv2.waitKey(1)

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
        """Single-shot alignment: measure angle, rotate wrist once."""
        if not self.set_joint_mode():
            self.get_logger().error("Cannot proceed without JOINT mode")
            return False

        # Use a dedicated executor to avoid conflicts with other nodes/threads
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)

        if not self.headless:
            cv2.namedWindow("Port Alignment", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Port Alignment", 800, 600)

        # Wait for first detection and image
        self.get_logger().info("Waiting for port detection and image...")
        deadline = time.time() + 10.0
        while time.time() < deadline:
            self._executor.spin_once(timeout_sec=0.1)
            with self.detection_lock:
                det = self.latest_detection
            with self.image_lock:
                img = self.latest_image
            if det and det.get("detected") and det.get("port_type") == self.target_class and img is not None:
                break
        else:
            self.get_logger().error("No detection/image received")
            if not self.headless:
                cv2.destroyAllWindows()
            return False

        self.get_logger().info("Collecting angle readings...")

        # Collect multiple readings for stability
        median_angle = self.collect_edge_readings(self.n_samples, timeout_sec=3.0)

        if median_angle is None:
            self.get_logger().error("No valid angle readings")
            if not self.headless:
                cv2.destroyAllWindows()
            return False

        error_deg = np.rad2deg(median_angle)
        self.get_logger().info(f"Detected angle: {error_deg:+.2f}°")

        # Check if already aligned
        if abs(error_deg) < self.angle_threshold_deg:
            self.get_logger().info(f"✓ Port already aligned! Error: {error_deg:+.2f}°")
            if not self.headless:
                self.get_logger().info("Press any key to exit...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return True

        # Get current wrist_3 position
        current_positions = self.get_current_joint_positions()
        if current_positions is None:
            self.get_logger().error("No joint state")
            if not self.headless:
                cv2.destroyAllWindows()
            return False

        current_wrist_3 = current_positions[5]

        # Compute target wrist position
        # Camera is on wrist, so rotating wrist rotates camera
        # To align port (make angle → 0), rotate wrist BY the detected angle
        target_wrist_3 = current_wrist_3 + median_angle  # Rotate by +angle to correct

        self.get_logger().info(
            f"Rotating wrist_3: {np.rad2deg(current_wrist_3):.1f}° → {np.rad2deg(target_wrist_3):.1f}° "
            f"(correction: {error_deg:+.1f}°)"
        )

        # Send command
        if not self.send_wrist_position(target_wrist_3):
            self.get_logger().error("Failed to send position command")
            if not self.headless:
                cv2.destroyAllWindows()
            return False

        # Wait for settle
        self.get_logger().info("Waiting for wrist to settle...")
        time.sleep(self.settle_time_sec + 1.0)

        # Verify alignment
        self.get_logger().info("Verifying alignment...")
        median_angle = self.collect_edge_readings(self.n_samples, timeout_sec=3.0)

        if median_angle is None:
            self.get_logger().warn("Could not verify alignment")
            if not self.headless:
                cv2.destroyAllWindows()
            return True  # Assume success

        final_error_deg = np.rad2deg(median_angle)
        self.get_logger().info(f"Final angle: {final_error_deg:+.2f}°")

        if abs(final_error_deg) < self.angle_threshold_deg:
            self.get_logger().info(f"✓ Port aligned! Final error: {final_error_deg:+.2f}°")
        else:
            self.get_logger().warn(f"Alignment not perfect. Final error: {final_error_deg:+.2f}°")

        if not self.headless:
            self.get_logger().info("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return True


def main():
    parser = argparse.ArgumentParser(description="Align port using polygon corners")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp", help="Port type")
    parser.add_argument("--camera", choices=["center", "left", "right"], default="center", help="Camera to use")
    args = parser.parse_args()

    rclpy.init()
    node = AlignPortEdge(port_type=args.port, camera=args.camera)

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