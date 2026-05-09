#!/usr/bin/env python3
#
# _0_move_to_init.py
# Move robot to initial spawn position using joint control.
#
# WHAT THIS DOES:
#   1. Reads current joint positions from /joint_states
#   2. Prints them (so you can see "init" position)
#   3. Moves to specified joint positions
#
# HOW TO RUN:
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/b_move_robot/_0_move_to_init.py
#

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
import time

from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode


class MoveToInit(Node):
    """
    Move robot using joint control.
    """

    def __init__(self):
        super().__init__("move_to_init")

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Publisher for joint commands
        self.joint_pub = self.create_publisher(
            JointMotionUpdate,
            "/aic_controller/joint_commands",
            10
        )

        # Client to change target mode
        self.change_mode_client = self.create_client(
            ChangeTargetMode,
            "/aic_controller/change_target_mode"
        )

        # Subscribe to joint states to read current position
        self.current_joints = None
        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        self.get_logger().info("MoveToInit: initialized")

    def joint_state_callback(self, msg: JointState):
        """Store current joint positions."""
        self.current_joints = msg

    def get_current_positions(self):
        """Get current joint positions as a dict."""
        if self.current_joints is None:
            return None

        positions = {}
        for i, name in enumerate(self.current_joints.name):
            if name in self.joint_names:
                positions[name] = self.current_joints.position[i]
        return positions

    def print_current_positions(self):
        """Print current joint positions."""
        rclpy.spin_once(self, timeout_sec=1.0)

        if self.current_joints is None:
            self.get_logger().warn("No joint state received yet")
            return None

        self.get_logger().info("=== CURRENT JOINT POSITIONS ===")
        for i, name in enumerate(self.current_joints.name):
            pos = self.current_joints.position[i]
            self.get_logger().info(f"  {name}: {pos:.4f} rad ({pos * 57.3:.1f}°)")
        self.get_logger().info("")

        return self.get_current_positions()

    def move_to_joints(self, positions: list, wait_sec: float = 3.0):
        """
        Move robot to specified joint positions.

        Args:
            positions: List of 6 joint positions in radians
                       [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
            wait_sec: Time to wait after sending command
        """
        if len(positions) != 6:
            self.get_logger().error(f"Expected 6 joint positions, got {len(positions)}")
            return

        # Set controller to JOINT mode first
        self.set_joint_mode()

        # Create JointTrajectoryPoint for target_state
        target_state = JointTrajectoryPoint()
        target_state.positions = positions
        target_state.velocities = [0.0] * 6
        target_state.accelerations = [0.0] * 6
        target_state.time_from_start.sec = int(wait_sec)
        target_state.time_from_start.nanosec = 0

        # Create JointMotionUpdate
        msg = JointMotionUpdate()
        msg.target_state = target_state
        msg.target_stiffness = [100.0] * 6
        msg.target_damping = [10.0] * 6
        msg.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        )
        msg.target_feedforward_torque = [0.0] * 6

        self.joint_pub.publish(msg)

        self.get_logger().info("Moving to joint positions:")
        for i, (name, pos) in enumerate(zip(self.joint_names, positions)):
            self.get_logger().info(f"  {name}: {pos:.4f} rad ({pos * 57.3:.1f}°)")

        time.sleep(wait_sec)
        self.get_logger().info("Done!")

    def set_joint_mode(self):
        """Set controller to JOINT mode."""
        if not self.change_mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("ChangeTargetMode service not available")
            return False

        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_JOINT

        future = self.change_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        result = future.result()
        if result and result.success:
            self.get_logger().info("Controller set to JOINT mode")
            return True
        else:
            self.get_logger().error("Failed to set JOINT mode")
            return False

    def move_to_init(self):
        """
        Move to initial/spawn position.
        These values are from the simulator's spawn configuration.
        """
        # Spawn positions from simulator (joint_states)
        positions = [
            -0.1565,    # shoulder_pan_joint
            -1.3527,    # shoulder_lift_joint
            -1.6957,    # elbow_joint
            -1.6640,    # wrist_1_joint
            1.5709,     # wrist_2_joint
            1.4142,     # wrist_3_joint
        ]

        self.get_logger().info("Moving to init position...")
        self.move_to_joints(positions, wait_sec=3.0)


def main(args=None):
    rclpy.init(args=args)

    node = MoveToInit()

    try:
        # Wait for joint states
        node.get_logger().info("Waiting for joint states...")
        import time
        start = time.time()
        while node.current_joints is None and (time.time() - start) < 5.0:
            rclpy.spin_once(node, timeout_sec=0.1)

        # Print current position first
        current = node.print_current_positions()

        # Then move to init
        node.get_logger().info("")
        node.move_to_init()

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()