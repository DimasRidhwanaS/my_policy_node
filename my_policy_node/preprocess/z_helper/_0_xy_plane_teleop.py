#!/usr/bin/env python3
#
# _0_xy_plane_teleop.py
# WASD teleop for XY plane movement + Z vertical + wrist_3 rotation.
# End effector stays perpendicular unless rotating with q/e.
#
# CONTROLS:
#   w → forward (+X)
#   s → backward (-X)
#   a → left (+Y)
#   d → right (-Y)
#   r → up (+Z)
#   f → down (-Z)
#   q → rotate CCW (+wrist)
#   e → rotate CW  (-wrist)
#   k → slow mode (0.02 m/s, 0.1 rad/s)
#   l → fast mode (0.08 m/s, 0.3 rad/s)
#   ESC → quit
#
# HOW TO RUN:
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/z_helper/_0_xy_plane_teleop.py
#

import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from pynput import keyboard
import numpy as np

from geometry_msgs.msg import Twist, Wrench, Vector3
from sensor_msgs.msg import JointState
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


SLOW_LINEAR_VEL  = 0.05  # m/s
FAST_LINEAR_VEL  = 0.2   # m/s
SLOW_ANGULAR_VEL = 0.2   # rad/s
FAST_ANGULAR_VEL = 0.6   # rad/s

KEY_MAPPINGS = {
    "w": (1, 0, 0),    # forward (+X)
    "s": (-1, 0, 0),   # backward (-X)
    "a": (0, 1, 0),    # left (+Y)
    "d": (0, -1, 0),   # right (-Y)
    "r": (0, 0, 1),    # up (+Z)
    "f": (0, 0, -1),   # down (-Z)
}

ROTATION_MAPPINGS = {
    "q": 1,    # rotate CCW (+angular Z)
    "e": -1,   # rotate CW  (-angular Z)
}


class XYPlaneTeleop(Node):
    """
    WASD teleop for XY plane movement.
    End effector stays perpendicular (angular velocity = 0).
    No Z movement - stays at current height.
    """

    def __init__(self):
        super().__init__("xy_plane_teleop")

        # Declare parameters
        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value

        # Publisher for Cartesian velocity commands
        self.motion_pub = self.create_publisher(
            MotionUpdate, f"/{self.controller_namespace}/pose_commands", 10
        )

        # Client to change target mode
        self.change_mode_client = self.create_client(
            ChangeTargetMode,
            f"/{self.controller_namespace}/change_target_mode"
        )

        # Wait for publisher and service
        while self.motion_pub.get_subscription_count() == 0:
            self.get_logger().info(f"Waiting for subscriber to '{self.controller_namespace}/pose_commands'...")
            time.sleep(0.5)

        while not self.change_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for service '{self.controller_namespace}/change_target_mode'...")
            time.sleep(0.5)

        # Track pressed keys
        self.active_keys = set()

        # Velocity settings
        self.linear_vel  = FAST_LINEAR_VEL
        self.angular_vel = FAST_ANGULAR_VEL

        # Frame ID for velocity commands
        self.frame_id = "base_link"

        # Wrist_3 tracking for camera-relative XY movement
        self.current_wrist3 = None
        self.ref_wrist3 = None
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.keyboard_listener.start()

        # Timer for sending commands at 25Hz
        self.timer = self.create_timer(0.04, self.send_velocity_command)

        self.get_logger().info("XY Plane Teleop initialized")
        self.get_logger().info("  Controls: w/a/s/d = XY, r/f = Z, q/e = rotate, k = slow, l = fast, ESC = quit")

    def _joint_state_cb(self, msg: JointState):
        pos_map = dict(zip(msg.name, msg.position))
        w3 = pos_map.get("wrist_3_joint")
        if w3 is None:
            return
        if self.ref_wrist3 is None:
            self.ref_wrist3 = w3
        self.current_wrist3 = w3

    def on_key_press(self, key):
        """Track pressed keys."""
        try:
            if hasattr(key, "char") and key.char is not None:
                self.active_keys.add(key.char.lower())
        except AttributeError:
            pass

    def on_key_release(self, key):
        """Remove released keys."""
        try:
            if hasattr(key, "char") and key.char is not None:
                k = key.char.lower()
                if k in self.active_keys:
                    self.active_keys.remove(k)
        except AttributeError:
            pass

        if key == keyboard.Key.esc:
            rclpy.shutdown()

    def send_velocity_command(self):
        """Send Cartesian velocity command."""
        vx = 0.0
        vy = 0.0
        vz = 0.0
        wz = 0.0

        for key in self.active_keys:
            if key in KEY_MAPPINGS:
                dx, dy, dz = KEY_MAPPINGS[key]
                vx += dx * self.linear_vel
                vy += dy * self.linear_vel
                vz += dz * self.linear_vel
            if key in ROTATION_MAPPINGS:
                wz += ROTATION_MAPPINGS[key] * self.angular_vel
            if key == "k":
                self.linear_vel  = SLOW_LINEAR_VEL
                self.angular_vel = SLOW_ANGULAR_VEL
            if key == "l":
                self.linear_vel  = FAST_LINEAR_VEL
                self.angular_vel = FAST_ANGULAR_VEL

        # Rotate XY to stay camera-relative regardless of wrist_3 rotation
        if self.current_wrist3 is not None and self.ref_wrist3 is not None:
            delta = self.current_wrist3 - self.ref_wrist3
            cos_d, sin_d = np.cos(delta), np.sin(delta)
            vx, vy = cos_d * vx - sin_d * vy, sin_d * vx + cos_d * vy

        # Build Twist message
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = vz
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = wz

        # Build MotionUpdate message
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

        # Log velocity if moving
        if abs(vx) > 0.001 or abs(vy) > 0.001 or abs(vz) > 0.001 or abs(wz) > 0.001:
            self.get_logger().info(
                f"v=({vx:.3f}, {vy:.3f}, {vz:.3f}) wz={wz:.3f}"
            )

    def set_cartesian_mode(self):
        """Set controller to CARTESIAN mode."""
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN

        self.get_logger().info("Setting control mode to CARTESIAN...")
        future = self.change_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result and result.success:
            self.get_logger().info("Controller set to CARTESIAN mode")
            time.sleep(0.5)
            return True
        else:
            self.get_logger().error("Failed to set CARTESIAN mode")
            return False

    def stop_keyboard_listener(self):
        """Stop keyboard listener."""
        if self.keyboard_listener:
            self.keyboard_listener.stop()


def main(args=None):
    print("""
    XY Plane Teleop
    ----------------------------------------------------
    Movement:
        w : forward (+X)
        s : backward (-X)
        a : left (+Y)
        d : right (-Y)
        r : up (+Z)
        f : down (-Z)

    Rotation (wrist):
        q : rotate CCW (+wrist)
        e : rotate CW  (-wrist)

    Speed:
        k : slow mode (0.02 m/s, 0.1 rad/s)
        l : fast mode (0.08 m/s, 0.3 rad/s)

        ESC : quit
    """)

    rclpy.init(args=args)
    node = XYPlaneTeleop()

    try:
        if not node.set_cartesian_mode():
            node.get_logger().error("Cannot proceed without CARTESIAN mode")
            return

        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_keyboard_listener()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)