#!/usr/bin/env python3
#
# Attempt1.py
# Orchestrator policy.
#
# HOW TO RUN:
#   pixi run bash -c "source install/setup.bash && ros2 run aic_model aic_model \
#       --ros-args -p use_sim_time:=true \
#       -p policy:=my_policy_node.Attempt1.Attempt1"

import os
import yaml

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from my_policy_node.preprocess.a_image_processing.task_board_finder import TaskBoardFinder


# Load config once at module level
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "attempt1_config.yaml",
)
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(_CONFIG_PATH, "r") as f:
    _CONFIG = yaml.safe_load(f)


class Attempt1(Policy):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("Attempt1: initialized")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:

        self.get_logger().info("Attempt1: insert_cable called")
        self.get_logger().info(f"  task id:    {task.id}")
        self.get_logger().info(f"  plug type:  {task.plug_type}")
        self.get_logger().info(f"  port name:  {task.port_name}")
        self.get_logger().info(f"  target:     {task.target_module_name}")
        self.get_logger().info(f"  time limit: {task.time_limit}s")

        # ── PHASE 1: Find and center over taskboard ──────────
        send_feedback("Phase 1: finding taskboard")
        finder = TaskBoardFinder(
            policy=self,
            config=_CONFIG["task_board_finder"],
            move_robot=move_robot,
            get_observation=get_observation,
        )
        if not finder.search_and_center():
            self.get_logger().error("Attempt1: failed to find taskboard")
            send_feedback("Phase 1 failed: taskboard not found")
            return False
        send_feedback("Phase 1 complete: taskboard found")

        # TODO PHASE 2: port_finder.locate(task.plug_type)
        # TODO PHASE 3: edge_detector.align()
        # TODO PHASE 4: inserter.execute()

        self.get_logger().info("Attempt1: insert_cable done")
        return True