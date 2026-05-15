#!/usr/bin/env python3
#
# _5_enter_port.py
# Wiggle insertion: small wrist rotations to find the port slot orientation.
#
# PROBLEM:
#   Connector is already near the port but won't go in — slight rotational
#   misalignment is stopping it.
#
# STRATEGY:
#   Wiggle wrist_3 in small steps: 3 steps right, back to center, 3 steps
#   left, back to center. Repeat. After each step, check if Z dropped
#   (insertion happened). While wiggling, maintain gentle downward pressure
#   via feedforward force in CARTESIAN mode, then rotate via wrist_3 offset.
#
#   Actually simpler: JOINT mode, wiggle wrist_3 only, keep all other joints
#   fixed. Monitor /observations for Z position to detect insertion.
#
# STOP CONDITIONS:
#   - Z ≤ target_z         → inserted, success
#   - |Fz| > force_abort   → hard jam, abort
#   - max_cycles reached   → give up
#
# HOW TO RUN:
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 \
#       my_policy_node/my_policy_node/preprocess/b_move_robot/_5_enter_port.py \
#       --port sfp --target-z 0.05
#

import argparse
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from trajectory_msgs.msg import JointTrajectoryPoint

from aic_control_interfaces.msg import JointMotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from aic_model_interfaces.msg import Observation
from sensor_msgs.msg import JointState


# ── Tunable parameters ────────────────────────────────────────────────────────

STEP_DEG        = 0.8     # degrees per wrist step (very small)
STEPS_PER_SIDE  = 3       # steps in each direction before returning to center
SETTLE_S        = 0.4     # seconds to wait after each step
MAX_CYCLES      = 10      # max wiggle cycles before giving up
FORCE_ABORT_N   = 35.0    # N — abort if |Fz| exceeds this
FORCE_SMOOTH    = 0.3     # EMA smoothing factor for force

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
WRIST3_IDX = 5


class EnterPort(Node):

    def __init__(self, port_type: str, target_z: float):
        super().__init__("enter_port")

        self.port_type = port_type.lower()
        self.target_z  = target_z

        self.get_logger().info(
            f"EnterPort: port={self.port_type}  target_z={self.target_z:.3f} m"
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.current_z      = None
        self.current_joints = None
        self.smooth_fz      = 0.0
        self.lock           = threading.Lock()

        # ── ROS ───────────────────────────────────────────────────────────────
        self.create_subscription(Observation, "/observations", self._obs_cb, 10)
        self.create_subscription(JointState,  "/joint_states", self._joint_cb, 10)

        self.joint_pub = self.create_publisher(
            JointMotionUpdate, "/aic_controller/joint_commands", 10
        )
        self.mode_client = self.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )

        self.get_logger().info("Waiting for joint command subscriber...")
        while self.joint_pub.get_subscription_count() == 0:
            time.sleep(0.2)

        self.get_logger().info("Waiting for mode service...")
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            time.sleep(0.2)

        self.get_logger().info("EnterPort ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _obs_cb(self, msg: Observation):
        fz = msg.wrist_wrench.wrench.force.z
        with self.lock:
            self.current_z  = msg.controller_state.tcp_pose.position.z
            self.smooth_fz  = FORCE_SMOOTH * fz + (1.0 - FORCE_SMOOTH) * self.smooth_fz

    def _joint_cb(self, msg: JointState):
        with self.lock:
            self.current_joints = msg

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_joint_mode(self) -> bool:
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_JOINT
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().success:
            self.get_logger().info("JOINT mode set.")
            time.sleep(0.5)
            return True
        self.get_logger().error("Failed to set JOINT mode.")
        return False

    def _get_joint_positions(self):
        with self.lock:
            msg = self.current_joints
        if msg is None:
            return None
        pos_map = dict(zip(msg.name, msg.position))
        return [pos_map.get(n, 0.0) for n in JOINT_NAMES]

    def _send_wrist(self, positions: list, duration_s: float = 0.4):
        """Send joint command keeping all joints except wrist_3 unchanged."""
        pt = JointTrajectoryPoint()
        pt.positions       = positions
        pt.velocities      = [0.0] * 6
        pt.accelerations   = [0.0] * 6
        pt.time_from_start.sec     = int(duration_s)
        pt.time_from_start.nanosec = int((duration_s % 1) * 1e9)

        msg = JointMotionUpdate()
        msg.target_state        = pt
        msg.target_stiffness    = [85.0] * 6
        msg.target_damping      = [75.0] * 6
        msg.target_feedforward_torque = [0.0] * 6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        self.joint_pub.publish(msg)

    def _spin_for(self, seconds: float):
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._executor.spin_once(timeout_sec=0.05)

    def _check_insertion(self) -> bool:
        """True if Z reached target or force is too high (abort)."""
        with self.lock:
            z  = self.current_z
            fz = self.smooth_fz
        if z is None:
            return False
        if z <= self.target_z:
            self.get_logger().info(f"Inserted!  Z={z:.4f} m")
            return True
        if abs(fz) > FORCE_ABORT_N:
            self.get_logger().warn(f"Force abort: Fz={fz:.1f} N")
            return True   # caller checks final Z to decide success/fail
        return False

    # ── Main logic ────────────────────────────────────────────────────────────

    def run(self) -> bool:
        if not self._set_joint_mode():
            return False

        # Use a dedicated executor to avoid conflicts with other nodes/threads
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)

        # Wait for joint state and observation
        self.get_logger().info("Waiting for data...")
        deadline = time.time() + 10.0
        while time.time() < deadline:
            self._executor.spin_once(timeout_sec=0.1)
            with self.lock:
                ready = self.current_joints is not None and self.current_z is not None
            if ready:
                break
        else:
            self.get_logger().error("No joint state / observation received.")
            return False

        base_positions = self._get_joint_positions()
        if base_positions is None:
            return False

        base_wrist3 = base_positions[WRIST3_IDX]
        step_rad    = np.deg2rad(STEP_DEG)

        self.get_logger().info(
            f"Base wrist_3 = {np.rad2deg(base_wrist3):.2f}°  "
            f"step = {STEP_DEG:.1f}°  sides = {STEPS_PER_SIDE}  cycles = {MAX_CYCLES}"
        )

        for cycle in range(MAX_CYCLES):
            self.get_logger().info(f"── Cycle {cycle + 1}/{MAX_CYCLES} ──")

            # ── 3 steps RIGHT ──────────────────────────────────────────────
            for s in range(1, STEPS_PER_SIDE + 1):
                positions = list(base_positions)
                positions[WRIST3_IDX] = base_wrist3 + s * step_rad
                self.get_logger().info(
                    f"  Right step {s}: wrist_3 → {np.rad2deg(positions[WRIST3_IDX]):.2f}°"
                )
                self._send_wrist(positions)
                self._spin_for(SETTLE_S)
                if self._check_insertion():
                    with self.lock:
                        return self.current_z <= self.target_z

            # ── Back to center ─────────────────────────────────────────────
            self.get_logger().info("  Center")
            self._send_wrist(list(base_positions))
            self._spin_for(SETTLE_S)
            if self._check_insertion():
                with self.lock:
                    return self.current_z <= self.target_z

            # ── 3 steps LEFT ───────────────────────────────────────────────
            for s in range(1, STEPS_PER_SIDE + 1):
                positions = list(base_positions)
                positions[WRIST3_IDX] = base_wrist3 - s * step_rad
                self.get_logger().info(
                    f"  Left step {s}: wrist_3 → {np.rad2deg(positions[WRIST3_IDX]):.2f}°"
                )
                self._send_wrist(positions)
                self._spin_for(SETTLE_S)
                if self._check_insertion():
                    with self.lock:
                        return self.current_z <= self.target_z

            # ── Back to center ─────────────────────────────────────────────
            self.get_logger().info("  Center")
            self._send_wrist(list(base_positions))
            self._spin_for(SETTLE_S)
            if self._check_insertion():
                with self.lock:
                    return self.current_z <= self.target_z

        self.get_logger().warn(f"Max cycles ({MAX_CYCLES}) reached — insertion failed.")
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Wrist wiggle insertion")
    parser.add_argument("--port",     choices=["sfp", "sc"], default="sfp")
    parser.add_argument("--target-z", type=float, default=0.05,
                        help="Z (m) that confirms insertion")
    args = parser.parse_args()

    rclpy.init()
    node = EnterPort(port_type=args.port, target_z=args.target_z)

    try:
        ok = node.run()
        node.get_logger().info("Done." if ok else "Failed.")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
