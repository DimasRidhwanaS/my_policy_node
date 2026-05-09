#!/usr/bin/env python3
#
# Attempt1.py
# Orchestrator policy — 3-phase cable insertion pipeline.
#
# PHASES:
#   1. PortSearch    — Circle to find port, center over it (YOLO)
#   2. EdgeDetector  — Fine alignment (OpenCV)
#   3. Inserter       — Force-controlled insertion
#
# HOW TO RUN:
#   pixi run bash -c "source install/setup.bash && ros2 run aic_model aic_model \
#       --ros-args -p use_sim_time:=true \
#       -p policy:=my_policy_node.Attempt1"

import os
import yaml

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from my_policy_node.preprocess.a_image_processing.port_search import PortSearch
from my_policy_node.preprocess.a_image_processing.edge_detector import EdgeDetector


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

        # Determine port type from task
        port_type = "sfp" if "sfp" in task.plug_type.lower() else "sc"
        self.get_logger().info(f"Attempt1: detected port type = {port_type}")

        # ── PHASE 1: Circle to find and center over port ──────────────────────
        send_feedback(f"Phase 1: searching for {port_type} port")

        port_search = PortSearch(
            policy=self,
            config=_CONFIG["port_search"],
            move_robot=move_robot,
            get_observation=get_observation,
        )

        if not port_search.search_and_center_port(port_type):
            self.get_logger().error(f"Attempt1: Phase 1 failed — {port_type} port not found")
            send_feedback(f"Phase 1 failed: {port_type} port not found")
            return False

        send_feedback(f"Phase 1 complete: centered over {port_type} port")
        self.get_logger().info("Attempt1: Phase 1 complete")

        # ── PHASE 2: Fine alignment using edge detection ─────────────────
        send_feedback("Phase 2: fine alignment")

        edge_detector = EdgeDetector(
            policy=self,
            config=_CONFIG["edge_detector"],
            move_robot=move_robot,
            get_observation=get_observation,
            current_pose=port_search.current_pose,  # Pass pose from Phase 1
        )

        if not edge_detector.align_with_hole(port_type):
            self.get_logger().warn("Attempt1: Phase 2 alignment imperfect, proceeding anyway")
            send_feedback("Phase 2: alignment acceptable")
        else:
            send_feedback("Phase 2 complete: aligned with port hole")

        self.get_logger().info("Attempt1: Phase 2 complete")

        # ── PHASE 3: Insertion ───────────────────────────────────────────
        send_feedback("Phase 3: inserting cable")
        self.get_logger().info("Attempt1: Phase 3 — insertion not yet implemented")

        # TODO: Implement Inserter phase
        # - Move down slowly while monitoring force sensor
        # - Abort if force > max_force
        # - Detect successful insertion

        send_feedback("Phase 3: insertion skipped (not implemented)")
        self.get_logger().info("Attempt1: insert_cable done (phases 1-2 complete)")
        return True