#!/usr/bin/env python3
#
# _4_move_down.py
# Move down with continuous XY centering and force monitoring.
#
# WHAT THIS DOES:
#   1. Subscribe to /port_detection for XY centering
#   2. Subscribe to /observations for Z position and force
#   3. Move down while keeping port centered
#   4. Stop when: target Z reached, force spike, or port too large
#
# HOW TO RUN:
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_4_move_down.py --port sfp --target-z 0.05
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
from geometry_msgs.msg import Twist, Wrench, Vector3
from aic_model_interfaces.msg import Observation
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


class MoveDown(Node):
    """
    Move down with XY centering. Simple approach:
    - Constant downward velocity
    - Continuous XY correction from camera
    - Stop on Z threshold, force, or port size
    """

    def __init__(self, port_type: str = "sfp", target_z: float = 0.05):
        super().__init__("move_down")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.target_z = target_z

        self.get_logger().info(f"MoveDown: {self.target_class} → Z={self.target_z}m")

        # State
        self.latest_detection = None
        self.detection_lock = threading.Lock()

        self.current_z = None
        self.current_force = (0.0, 0.0, 0.0)
        self.robot_lock = threading.Lock()

        self.done = False

        # Subscribers
        self.create_subscription(String, "/port_detection", self._detection_cb, 10)
        self.create_subscription(Observation, "/observations", self._obs_cb, 10)

        # Publisher
        self.motion_pub = self.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 10)

        # Service client
        self.change_mode_client = self.create_client(ChangeTargetMode, "/aic_controller/change_target_mode")

        # Wait for connections
        while self.motion_pub.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscriber...")
            time.sleep(0.5)

        while not self.change_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")
            time.sleep(0.5)

        # Parameters
        self.vz = -0.02                    # m/s - downward velocity
        self.max_xy_vel = 0.02             # m/s - max XY correction
        self.xy_gain = 0.001               # pixel → velocity gain
        self.z_threshold = 0.005           # m - stop when this close to target
        self.force_z_threshold = 25.0      # N - Z force = contact
        self.force_xy_threshold = 20.0     # N - XY force = collision
        self.port_size_max = 400           # px - stop when port this large

        # Force smoothing
        self.smooth_fx = 0.0
        self.smooth_fy = 0.0
        self.smooth_fz = 0.0

        # Direction signs (tune if needed)
        self.sign_x_to_vy = -1.0
        self.sign_y_to_vx = 1.0

        # Timer at 100Hz
        self.timer = self.create_timer(1.0/100.0, self.control_loop)

        self.get_logger().info("MoveDown ready")

    def _detection_cb(self, msg):
        try:
            with self.detection_lock:
                self.latest_detection = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Parse error: {e}")

    def _obs_cb(self, msg: Observation):
        with self.robot_lock:
            self.current_z = msg.controller_state.tcp_pose.position.z
            f = msg.wrist_wrench.wrench.force
            self.current_force = (f.x, f.y, f.z)

            # Smooth force
            alpha = 0.3
            self.smooth_fx = alpha * f.x + (1-alpha) * self.smooth_fx
            self.smooth_fy = alpha * f.y + (1-alpha) * self.smooth_fy
            self.smooth_fz = alpha * f.z + (1-alpha) * self.smooth_fz

    def set_cartesian_mode(self):
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN

        future = self.change_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().success:
            self.get_logger().info("CARTESIAN mode set")
            time.sleep(0.5)
            return True
        return False

    def send_velocity(self, vx, vy, vz):
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
        msg.target_stiffness = np.diag([70.0, 70.0, 85.0, 85.0, 85.0, 85.0]).flatten()
        msg.target_damping = np.diag([60.0, 60.0, 75.0, 75.0, 75.0, 75.0]).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0)
        )
        msg.wrench_feedback_gains_at_tip = [0.0]*6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        self.motion_pub.publish(msg)

    def control_loop(self):
        """Main loop: move down."""
        if self.done:
            return

        # Get state
        with self.robot_lock:
            z = self.current_z
            fx, fy, fz = self.smooth_fx, self.smooth_fy, self.smooth_fz

        with self.detection_lock:
            det = self.latest_detection

        # Wait for data
        if z is None:
            return

        # ─── STOP CONDITIONS ───

        # 1. Target Z reached
        if z <= self.target_z + self.z_threshold:
            self.get_logger().info(f"✓ Target Z: {z:.4f}m ≤ {self.target_z}m")
            self.send_velocity(0, 0, 0)
            self.done = True
            return

        # 2. Z force spike (contact)
        if abs(fz) > self.force_z_threshold:
            self.get_logger().info(f"✓ Z contact: fz={fz:.1f}N")
            self.send_velocity(0, 0, 0)
            self.done = True
            return

        # 3. XY force spike (collision)
        if abs(fx) > self.force_xy_threshold or abs(fy) > self.force_xy_threshold:
            self.get_logger().warn(f"⚠ XY collision: fx={fx:.1f}N, fy={fy:.1f}N")
            self.send_velocity(0, 0, 0)
            self.done = True
            return

        # 4. Port size (too close)
        if det and det.get("detected") and det.get("port_type") == self.target_class:
            bbox = det.get("bbox", [0,0,0,0])
            if len(bbox) >= 4:
                port_size = max(bbox[2]-bbox[0], bbox[3]-bbox[1])
                if port_size > self.port_size_max:
                    self.get_logger().info(f"✓ Port size: {port_size}px")
                    self.send_velocity(0, 0, 0)
                    self.done = True
                    return

        # ─── SEND COMMAND: MOVE DOWN ───

        vz = self.vz  # constant downward velocity
        vx, vy = 0.0, 0.0  # no XY correction

        self.get_logger().info(f"z={z:.3f}m | vz={vz:+.3f} | fz={fz:.1f}N")
        self.send_velocity(vx, vy, vz)

    def run(self):
        if not self.set_cartesian_mode():
            return False

        # Use a dedicated executor to avoid conflicts with other nodes/threads
        executor = SingleThreadedExecutor()
        executor.add_node(self)

        # Wait for data
        self.get_logger().info("Waiting for data...")
        start = time.time()
        while (self.current_z is None) and (time.time() - start) < 10:
            executor.spin_once(timeout_sec=0.1)

        if self.current_z is None:
            self.get_logger().error("No Z data")
            return False

        self.get_logger().info(f"Starting Z: {self.current_z:.4f}m")

        while rclpy.ok() and not self.done:
            executor.spin_once(timeout_sec=0.1)

        return self.done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp")
    parser.add_argument("--target-z", type=float, default=0.05)
    args = parser.parse_args()

    rclpy.init()
    node = MoveDown(port_type=args.port, target_z=args.target_z)

    try:
        node.run()
    except KeyboardInterrupt:
        node.send_velocity(0, 0, 0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()