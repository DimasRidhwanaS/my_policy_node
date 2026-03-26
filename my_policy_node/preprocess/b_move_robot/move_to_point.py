#!/usr/bin/env python3
#
# move_to_point.py
# Simplified motion planner using Policy.set_pose_target()
# No manual mode switching needed — the base class handles everything.

import time
from geometry_msgs.msg import Point, Pose, Quaternion
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode

# ─────────────────────────────────────────────
# TEST MODE
# Set True to test without YOLO
# Edit HARDCODED_TARGET_POSE to a reachable position
# ─────────────────────────────────────────────
USE_HARDCODED_TARGET = True

HARDCODED_TARGET_POSE = Pose(
    position=Point(x=-0.4, y=0.2, z=0.25),
    orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),  # pointing down
)

# ─────────────────────────────────────────────
# TUNING PARAMETERS
# ─────────────────────────────────────────────
APPROACH_HEIGHT_OFFSET = 0.05   # meters above port to hover
PRE_INSERTION_OFFSET   = 0.03   # meters above port for fine alignment
MOVE_TIMEOUT_SEC       = 5.0    # seconds to wait after each move
MAX_INSERTION_FORCE    = 10.0   # Newtons — safety limit

# Home joint positions in radians (from sample_config.yaml)
HOME_JOINT_POSITIONS = [
    -0.1597,   # shoulder_pan_joint
    -1.3542,   # shoulder_lift_joint
    -1.6648,   # elbow_joint
    -1.6933,   # wrist_1_joint
     1.5710,   # wrist_2_joint
     1.4110,   # wrist_3_joint
]


class MoveToPoint:
    """
    Motion planner for cable insertion.
    Uses Policy.set_pose_target() — no manual mode switching needed.

    Example usage:
        planner = MoveToPoint(policy_instance)
        planner.go_to_home(move_robot)
        planner.approach(port_pose, move_robot)
        planner.align(port_pose, move_robot)
        success = planner.insert(port_pose, move_robot, get_observation)
        planner.retract(port_pose, move_robot)
    """

    def __init__(self, policy):
        # policy = the Policy instance (self in Attempt1)
        # gives access to set_pose_target(), get_logger(), sleep_for()
        self.policy = policy
        self.logger = policy.get_logger()

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def go_to_home(self, move_robot):
        """Move to safe home position using joint control."""
        self.logger.info("MoveToPoint: going to home position")

        msg = JointMotionUpdate()
        msg.target_state.positions    = HOME_JOINT_POSITIONS
        msg.target_stiffness          = [85.0] * 6
        msg.target_damping            = [75.0] * 6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        move_robot(joint_motion_update=msg)  # ← keyword arg!
        self._wait(3.0)
        self.logger.info("MoveToPoint: reached home position")

    def approach(self, port_pose: Pose, move_robot):
        """Coarse move to position above the port."""
        self.logger.info("MoveToPoint: approaching port")
        target = HARDCODED_TARGET_POSE if USE_HARDCODED_TARGET else port_pose


        approach_pose = Pose(
            position=Point(
                x=target.position.x,
                y=target.position.y,
                z=target.position.z + APPROACH_HEIGHT_OFFSET,
            ),
            orientation=target.orientation,
        )

        # set_pose_target handles all the MotionUpdate details
        self.policy.set_pose_target(move_robot=move_robot, pose=approach_pose)
        self._wait(MOVE_TIMEOUT_SEC)
        self.logger.info("MoveToPoint: approach complete")

    def align(self, port_pose: Pose, move_robot):
        """Fine alignment — directly in line with port."""
        self.logger.info("MoveToPoint: aligning to port")
        target = HARDCODED_TARGET_POSE if USE_HARDCODED_TARGET else port_pose

        align_pose = Pose(
            position=Point(
                x=target.position.x,
                y=target.position.y,
                z=target.position.z + PRE_INSERTION_OFFSET,
            ),
            orientation=target.orientation,
        )

        self.policy.set_pose_target(move_robot=move_robot, pose=align_pose)
        self._wait(MOVE_TIMEOUT_SEC)
        self.logger.info("MoveToPoint: alignment complete")

    def insert(self, port_pose: Pose, move_robot, get_observation) -> bool:
        """Slow controlled move into port while monitoring force."""
        self.logger.info("MoveToPoint: inserting")
        target = HARDCODED_TARGET_POSE if USE_HARDCODED_TARGET else port_pose

        insert_pose = Pose(
            position=Point(
                x=target.position.x,
                y=target.position.y,
                z=target.position.z,
            ),
            orientation=target.orientation,
        )

        self.policy.set_pose_target(move_robot=move_robot, pose=insert_pose)
        self._wait(MOVE_TIMEOUT_SEC)

        # Monitor force sensor
        observation = get_observation()
        if observation is not None:
            force = self._get_force_magnitude(observation)
            self.logger.info(f"MoveToPoint: insertion force = {force:.2f}N")
            if force > MAX_INSERTION_FORCE:
                self.logger.warn(f"MoveToPoint: too much force ({force:.1f}N)!")
                self.retract(port_pose, move_robot)
                return False

        self.logger.info("MoveToPoint: insertion complete")
        return True

    def retract(self, port_pose: Pose, move_robot):
        """Pull back from port."""
        self.logger.info("MoveToPoint: retracting")
        target = HARDCODED_TARGET_POSE if USE_HARDCODED_TARGET else port_pose

        retract_pose = Pose(
            position=Point(
                x=target.position.x,
                y=target.position.y,
                z=target.position.z + APPROACH_HEIGHT_OFFSET,
            ),
            orientation=target.orientation,
        )

        self.policy.set_pose_target(move_robot=move_robot, pose=retract_pose)
        self._wait(2.0)
        self.logger.info("MoveToPoint: retracted")

    # ─────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────

    def _wait(self, seconds: float):
        time.sleep(seconds)

    def _get_force_magnitude(self, observation) -> float:
        """Get total force from wrist wrench sensor."""
        try:
            wrench = observation.wrist_wrench.wrench
            fx = wrench.force.x
            fy = wrench.force.y
            fz = wrench.force.z
            import numpy as np
            return float(np.sqrt(fx**2 + fy**2 + fz**2))
        except Exception:
            return 0.0