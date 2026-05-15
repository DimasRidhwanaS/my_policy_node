#!/usr/bin/env python3
#
# Attempt1.py
# Competition policy — full cable insertion pipeline.
#
# PIPELINE (per attempt):
#   1. Circle search until port detected        [_1_circle_move]
#   2. Move to port                             [_2_move_to_port]
#   3. Align port orientation                   [_3_2_align_port_edge]
#   4. Move to port again (post-alignment)      [_2_move_to_port]
#   5. Tilt arm perpendicular                   [_4_1_tilt_arm]
#   6. Descend to Z=_TARGET_Z_DESCENT           [_4_move_down]
#   7. Micro alignment (connector tip → port)   [_5_1_micro_alignment]
#   8. Wiggle insertion (10s timeout)           [_5_enter_port]
#
# RETRY:
#   - Port not detected in step 1 → restart from step 1
#   - Step 8 fails or times out after 10s → restart from step 1
#   - Any other step fails → restart from step 1
#   - Max _MAX_RETRIES attempts total
#
# HOW TO RUN (in separate terminals, sim already running):
#   Terminal 1:
#     cd /home/ridhwana/ws_aic/src/aic && pixi run bash -c \
#       "source install/setup.bash && ros2 run aic_model aic_model \
#        --ros-args -p use_sim_time:=true -p policy:=my_policy_node.Attempt1"
#   Terminal 2:
#     cd /home/ridhwana/ws_aic/src/aic && pixi run bash -c \
#       "source install/setup.bash && ros2 run aic_engine aic_engine \
#        --ros-args -p use_sim_time:=true \
#        -p config_file_path:=$(ros2 pkg prefix aic_engine)/share/aic_engine/config/sample_config.yaml"

import os
import sys
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

# Add preprocess/ to sys.path so standalone scripts are importable as packages
_PREPROCESS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess")
if _PREPROCESS_DIR not in sys.path:
    sys.path.insert(0, _PREPROCESS_DIR)

from a_image_processing._1_camera_stream import CameraStream
from b_move_robot._1_circle_move import CircleMover
from b_move_robot._2_move_to_port import MoveToPort
from b_move_robot._3_2_align_port_edge import AlignPortEdge
from b_move_robot._4_1_tilt_arm import TiltArm
from b_move_robot._4_move_down import MoveDown
from b_move_robot._5_1_micro_alignment import MicroAlignment
from b_move_robot._5_enter_port import EnterPort


# ── Tunable constants ─────────────────────────────────────────────────────────
_CIRCLE_CENTER_X   = -0.5
_CIRCLE_CENTER_Y   =  0.4
_CIRCLE_RADIUS     =  0.2    # m
_CIRCLE_HEIGHT     =  0.3    # m
_CIRCLE_STEP_DEG   =  30.0   # degrees per waypoint
_TARGET_Z_DESCENT  =  0.24   # m — MoveDown stops here
_TARGET_Z_INSERT   =  0.10   # m — EnterPort confirms insertion here (tune as needed)
_WIGGLE_TIMEOUT_S  =  10.0   # s — abort wiggle and restart if not done in this time
_MAX_RETRIES       =  3


class Attempt1(Policy):

    def __init__(self, parent_node):
        super().__init__(parent_node)

        # Start CameraStream headless in background threads.
        # This publishes /port_detection which all standalone scripts subscribe to.
        self._camera_node = CameraStream(port_type=None)
        self._camera_stop = threading.Event()
        self._camera_executor = SingleThreadedExecutor()
        self._camera_executor.add_node(self._camera_node)

        # Spin thread: dedicated executor so crashes don't propagate
        def _spin_camera():
            while not self._camera_stop.is_set():
                try:
                    self._camera_executor.spin_once(timeout_sec=0.1)
                except Exception as e:
                    self.get_logger().error(f"Camera spin error: {e}")

        self._spin_thread = threading.Thread(target=_spin_camera, daemon=True)
        self._spin_thread.start()

        # Detection loop: runs YOLO at ~30fps, publishes to /port_detection
        def _detect_loop():
            frame_count = 0
            last_log = time.time()
            while not self._camera_stop.is_set():
                try:
                    img = self._camera_node.get_active_image()
                    if img is not None:
                        self._camera_node.run_detection(img)
                        frame_count += 1
                        now = time.time()
                        if now - last_log >= 1.0:  # Log every second
                            self.get_logger().info(f"Camera: {self._camera_node.active}, frames: {frame_count}, target: {self._camera_node.target_class}")
                            last_log = now
                    else:
                        if frame_count == 0 and time.time() - last_log >= 2.0:
                            self.get_logger().warn("No camera images received - is /center_camera/image publishing?")
                            last_log = time.time()
                except Exception as e:
                    self.get_logger().error(f"Detection error: {e}")
                time.sleep(0.033)

        self._detect_thread = threading.Thread(target=_detect_loop, daemon=True)
        self._detect_thread.start()

        self.get_logger().info("Attempt1: camera stream started (headless)")
        self.get_logger().info("Attempt1: ready")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info("Attempt1: insert_cable called")
        self.get_logger().info(f"  plug: {task.plug_type} / {task.plug_name}")
        self.get_logger().info(f"  port: {task.port_type} / {task.port_name}")
        self.get_logger().info(f"  time_limit: {task.time_limit}s")

        port_type = "sfp" if "sfp" in task.plug_type.lower() else "sc"

        # Focus camera on this port type and reset target lock
        self._camera_node.port_type = port_type
        self._camera_node.target_class = f"{port_type}_port"
        self._camera_node.locked_target_pos = None
        self._camera_node.lock_lost_frames = 0
        # Use center camera for circle search and centering
        self._camera_node.active = "1_center"

        send_feedback(f"Starting pipeline: port={port_type}")

        for attempt in range(_MAX_RETRIES):
            self.get_logger().info(f"Attempt {attempt + 1}/{_MAX_RETRIES}")
            send_feedback(f"Attempt {attempt + 1}/{_MAX_RETRIES}")

            # ── Step 1: Circle search until port detected ──────────────────
            send_feedback("Step 1: circle search")
            circle_node = CircleMover()
            found = circle_node.trace_circle(
                center_x=_CIRCLE_CENTER_X,
                center_y=_CIRCLE_CENTER_Y,
                radius=_CIRCLE_RADIUS,
                height=_CIRCLE_HEIGHT,
                angle_step_deg=_CIRCLE_STEP_DEG,
                direction=-1,
                port_type=port_type,
            )
            circle_node.destroy_node()
            if not found:
                send_feedback("Step 1: port not detected — restarting")
                continue
            send_feedback("Step 1: port detected")

            # ── Step 2: Move to port ───────────────────────────────────────
            send_feedback("Step 2: moving to port")
            node = MoveToPort(port_type=port_type)
            ok = node.run()
            node.destroy_node()
            if not ok:
                send_feedback("Step 2: failed — restarting")
                continue
            send_feedback("Step 2: done")

            # ── Step 3: Align port orientation ────────────────────────────
            send_feedback("Step 3: aligning orientation")
            # Switch to right camera for alignment (sees port from angle)
            self._camera_node.active = "3_right"
            node = AlignPortEdge(port_type=port_type, camera="right", headless=True)
            ok = node.run()
            node.destroy_node()
            if not ok:
                send_feedback("Step 3: failed — restarting")
                continue
            send_feedback("Step 3: done")

            # ── Step 4: Move to port again (post-alignment re-center) ─────
            send_feedback("Step 4: re-centering after alignment")
            # Switch back to center camera for positioning
            self._camera_node.active = "1_center"
            node = MoveToPort(port_type=port_type, camera="center")
            ok = node.run()
            node.destroy_node()
            if not ok:
                send_feedback("Step 4: failed — restarting")
                continue
            send_feedback("Step 4: done")

            # ── Step 5: Tilt arm perpendicular ────────────────────────────
            send_feedback("Step 5: tilting arm")
            node = TiltArm()
            ok = node.run()
            node.destroy_node()
            if not ok:
                send_feedback("Step 5: failed — restarting")
                continue
            send_feedback("Step 5: done")

            # ── Step 6: Descend to approach Z ─────────────────────────────
            send_feedback(f"Step 6: descending to Z={_TARGET_Z_DESCENT}m")
            node = MoveDown(port_type=port_type, target_z=_TARGET_Z_DESCENT)
            ok = node.run()
            node.destroy_node()
            if not ok:
                send_feedback("Step 6: failed — restarting")
                continue
            send_feedback("Step 6: done")

            # ── Step 7: Micro alignment (connector tip → port tip) ────────
            send_feedback("Step 7: micro alignment")
            node = MicroAlignment(port_type=port_type, camera="center")
            ok = node.run()
            node.destroy_node()
            if not ok:
                send_feedback("Step 7: failed — restarting")
                continue
            send_feedback("Step 7: done")

            # ── Step 8: Wiggle insertion with 10s timeout ─────────────────
            send_feedback(f"Step 8: wiggle insertion (timeout={_WIGGLE_TIMEOUT_S}s)")
            enter_node = EnterPort(port_type=port_type, target_z=_TARGET_Z_INSERT)
            result_box = [False]

            def _run_enter(node=enter_node, box=result_box):
                box[0] = node.run()

            t = threading.Thread(target=_run_enter, daemon=True)
            t.start()
            t.join(timeout=_WIGGLE_TIMEOUT_S)

            if t.is_alive():
                send_feedback("Step 8: timed out — restarting")
                enter_node.destroy_node()
                continue

            ok = result_box[0]
            enter_node.destroy_node()

            if not ok:
                send_feedback("Step 8: wiggle failed — restarting")
                continue

            send_feedback("Pipeline complete — insertion confirmed!")
            self.get_logger().info("Attempt1: insert_cable complete")
            return True

        send_feedback("All attempts exhausted — failed")
        self.get_logger().warn("Attempt1: all attempts exhausted")
        return False
