#!/usr/bin/env python3
#
# _3_allign_ort.py
# Align end effector orientation with port orientation.
# Rotates wrist_3_joint (joint 6) to match the port's angle.
#
# APPROACH: single-shot position control
#   1. Collect N orientation readings from /port_detection, take median (denoise)
#   2. Read current wrist_3 position from /joint_states
#   3. Send one position command: target = current_wrist3 + sign * median_error
#   4. Wait to settle, re-check
#   5. Repeat until |error| < 2°
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
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


class AlignOrientation(Node):

    def __init__(self, port_type: str = "sfp"):
        super().__init__("align_orientation")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.get_logger().info(f"AlignOrientation: looking for {self.target_class}")

        self.latest_detection = None
        self.detection_lock = threading.Lock()

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

        self.create_subscription(String, "/port_detection", self.detection_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)

        self.motion_pub = self.create_publisher(
            JointMotionUpdate, "/aic_controller/joint_commands", 10
        )
        self.change_mode_client = self.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )

        while self.motion_pub.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscriber to '/aic_controller/joint_commands'...")
            time.sleep(0.5)

        while not self.change_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service '/aic_controller/change_target_mode'...")
            time.sleep(0.5)

        # Parameters
        self.angle_threshold_rad = np.deg2rad(2)  # stop when |error| < 2°
        self.n_samples = 10                        # frames to collect per measurement
        self.settle_sec = 1.5                      # wait after each position command
        self.max_iterations = 10                   # give up after this many corrections
        self.sign_correction = -1.0                # flip if robot rotates wrong way

        self.get_logger().info(f"  Threshold:   {np.rad2deg(self.angle_threshold_rad):.1f}°")
        self.get_logger().info(f"  Samples:     {self.n_samples}")
        self.get_logger().info(f"  Settle time: {self.settle_sec}s")
        self.get_logger().info(f"  Max iter:    {self.max_iterations}")

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

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_current_joint_positions(self):
        """Return all 6 joint positions in joint_names order, or None."""
        with self.joint_lock:
            msg = self.current_joints
        if msg is None:
            return None
        pos_map = dict(zip(msg.name, msg.position))
        return [pos_map.get(name, 0.0) for name in self.joint_names]

    def collect_orientation_samples(self, n: int, timeout_sec: float = 5.0):
        """Collect n valid orientation readings and return the median. None on timeout."""
        samples = []
        deadline = time.time() + timeout_sec

        while len(samples) < n:
            if time.time() > deadline:
                self.get_logger().warn(f"Timeout collecting samples (got {len(samples)}/{n})")
                break
            rclpy.spin_once(self, timeout_sec=0.05)
            with self.detection_lock:
                det = self.latest_detection
            if det is None or not det.get("detected", False):
                continue
            if det.get("port_type", "") != self.target_class:
                continue
            samples.append(det["orientation"])

        if not samples:
            return None
        return float(np.median(samples))

    def send_position_command(self, positions: list):
        target_state = JointTrajectoryPoint()
        target_state.positions = positions
        target_state.velocities = [0.0] * 6
        target_state.accelerations = [0.0] * 6
        target_state.time_from_start.sec = int(self.settle_sec)
        target_state.time_from_start.nanosec = 0

        msg = JointMotionUpdate()
        msg.target_state = target_state
        msg.target_stiffness = [85.0] * 6
        msg.target_damping = [75.0] * 6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        msg.target_feedforward_torque = [0.0] * 6

        self.motion_pub.publish(msg)

    def run(self):
        if not self.set_joint_mode():
            self.get_logger().error("Cannot proceed without JOINT mode")
            return False

        # Wait for first valid detection
        self.get_logger().info("Waiting for initial detection...")
        deadline = time.time() + 10.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self.detection_lock:
                det = self.latest_detection
            if det and det.get("detected") and det.get("port_type") == self.target_class:
                break
        else:
            self.get_logger().error("No detection received — aborting")
            return False

        for i in range(self.max_iterations):
            self.get_logger().info(f"--- Iteration {i + 1}/{self.max_iterations} ---")

            median_error = self.collect_orientation_samples(self.n_samples)
            if median_error is None:
                self.get_logger().error("Failed to collect orientation samples")
                return False

            median_error = self.normalize_angle(median_error)
            self.get_logger().info(f"  Median error: {np.rad2deg(median_error):+.2f}°")

            if abs(median_error) < self.angle_threshold_rad:
                self.get_logger().info(f"✓ Aligned! error={np.rad2deg(median_error):.2f}°")
                return True

            positions = self.get_current_joint_positions()
            if positions is None:
                self.get_logger().error("No joint state available")
                return False

            correction = self.sign_correction * median_error
            positions[5] += correction

            self.get_logger().info(
                f"  Correcting wrist_3 by {np.rad2deg(correction):+.2f}° "
                f"→ target {np.rad2deg(positions[5]):.2f}°"
            )

            self.send_position_command(positions)
            time.sleep(self.settle_sec)

        self.get_logger().warn("Max iterations reached without full alignment")
        return False


def main():
    parser = argparse.ArgumentParser(description="Align end effector orientation with port")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp")
    args = parser.parse_args()

    rclpy.init()
    node = AlignOrientation(port_type=args.port)

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
