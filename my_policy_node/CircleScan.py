#!/usr/bin/env python3
#
# CircleScan.py
# Moves the robot arm in a horizontal circle, driven by sim observation time.
# Mirrors the WaveArm pattern from aic_example_policies exactly.
#
# Movement: self.set_pose_target() from aic_model.policy  ← provided framework code
# Planner:  math.cos/sin to compute circular waypoints     ← ours
#
# HOW TO RUN:
#   Terminal 1 — sim:
#     pixi run ros2 launch aic_bringup aic_gz_bringup.launch.py \
#       ground_truth:=true spawn_task_board:=true spawn_cable:=true attach_cable_to_gripper:=true
#
#   Terminal 2 — policy:
#     pixi run bash -c "source install/setup.bash && ros2 run aic_model aic_model \
#       --ros-args -p use_sim_time:=true \
#       -p policy:=my_policy_node.CircleScan.CircleScan"
#
#   Terminal 3 — trigger task:
#     pixi run bash -c "source install/setup.bash && ros2 action send_goal /insert_cable \
#       aic_task_interfaces/action/InsertCable \
#       '{task: {id: \"test\", time_limit: 300}}'"
#
#   Terminal 4 — camera viewer:
#     pixi run bash -c "source install/setup.bash && python3 \
#       my_policy_node/my_policy_node/preprocess/z_helper/camera_viewer.py"

import math

from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

# ─────────────────────────────────────────────
# SCAN PARAMETERS
# ─────────────────────────────────────────────
CENTER_X    = -0.4    # circle center X in base_link (meters) — same area WaveArm sweeps
CENTER_Y    =  0.45   # circle center Y in base_link (meters)
RADIUS      =  0.15   # circle radius (meters)
HEIGHT      =  0.25   # fixed Z height (meters) — same as WaveArm
LOOP_SEC    =  8.0    # seconds for one full lap


class CircleScan(Policy):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("CircleScan: initialized")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:

        self.get_logger().info("CircleScan: starting")
        send_feedback("CircleScan: circling over task area")

        start_time = self.time_now()
        timeout    = Duration(seconds=float(task.time_limit))

        while (self.time_now() - start_time) < timeout:
            self.sleep_for(0.25)

            observation = get_observation()
            if observation is None:
                self.get_logger().info("CircleScan: waiting for observation")
                continue

            # Use sim timestamp from observation — same approach as WaveArm
            t = (
                observation.center_image.header.stamp.sec
                + observation.center_image.header.stamp.nanosec / 1e9
            )

            # Map time to a full circle [0, 2π]
            angle = 2.0 * math.pi * (t % LOOP_SEC) / LOOP_SEC

            x = CENTER_X + RADIUS * math.cos(angle)
            y = CENTER_Y + RADIUS * math.sin(angle)

            self.set_pose_target(
                move_robot=move_robot,
                pose=Pose(
                    position=Point(x=x, y=y, z=HEIGHT),
                    orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
                ),
            )

            self.get_logger().info(
                f"CircleScan: angle={math.degrees(angle):.0f}°  pos=({x:.3f}, {y:.3f}, {HEIGHT})"
            )

        self.get_logger().info("CircleScan: time limit reached")
        return True
