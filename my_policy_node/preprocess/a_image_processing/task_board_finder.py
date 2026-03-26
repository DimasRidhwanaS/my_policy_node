#!/usr/bin/env python3
#
# task_board_finder.py
# Phase 1 — Find and center the robot over the taskboard.
#
# WHAT THIS DOES:
#   1. Move robot to search position (from config)
#   2. Detect taskboard using YOLO on center_camera
#   3. Center robot over taskboard (X,Y corrections)
#   4. Move down until taskboard is close enough (Z steps)
#
# RETURNS:
#   True  — robot is positioned close enough to taskboard
#   False — taskboard not found after max attempts
#
# USAGE:
#   finder = TaskBoardFinder(policy, config, move_robot, get_observation)
#   success = finder.search_and_center()

import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion
from ultralytics import YOLO


class TaskBoardFinder:
    def __init__(self, policy, config, move_robot, get_observation):
        """
        Args:
            policy:          Policy instance (for set_pose_target, logger, sleep_for)
            config:          dict from attempt1_config.yaml [task_board_finder] section
            move_robot:      MoveRobotCallback from insert_cable
            get_observation: GetObservationCallback from insert_cable
        """
        self.policy          = policy
        self.cfg             = config
        self.move_robot      = move_robot
        self.get_observation = get_observation
        self.logger          = policy.get_logger()
        self.bridge          = CvBridge()

        # Load YOLO model
        model_path = config.get("model_path")
        if not os.path.isabs(model_path):
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(package_dir, model_path)

        self.logger.info(f"TaskBoardFinder: loading YOLO from {model_path}")
        self.model      = YOLO(model_path)
        self.confidence = config.get("confidence_threshold", 0.5)
        self.class_name = "task_board"

        # Current robot pose tracking
        self.current_pose = Pose(
            position=Point(
                x=config["search_pose"]["x"],
                y=config["search_pose"]["y"],
                z=config["search_pose"]["z"],
            ),
            orientation=Quaternion(
                x=config["search_pose"]["qx"],
                y=config["search_pose"]["qy"],
                z=config["search_pose"]["qz"],
                w=config["search_pose"]["qw"],
            ),
        )

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def search_and_center(self) -> bool:
        """
        Full Phase 1 pipeline:
        1. Move to search position
        2. Detect taskboard
        3. Center over it (X,Y)
        4. Move down toward it (Z)

        Returns True if successfully positioned, False if taskboard not found.
        """
        self.logger.info("TaskBoardFinder: starting search_and_center")

        # Step 1: Move to search position
        self._move_to_search_position()

        # Step 2: Detect taskboard — scan if not visible
        if not self._find_taskboard_with_scan():
            self.logger.error("TaskBoardFinder: taskboard not found after scanning")
            return False

        # Step 3 + 4: Center and descend loop
        self.logger.info("TaskBoardFinder: starting center and descend loop")
        return self._center_and_descend()

    # ─────────────────────────────────────────
    # PRIVATE — MOVEMENT
    # ─────────────────────────────────────────

    def _move_to_search_position(self):
        """Move robot to initial search pose from config."""
        self.logger.info("TaskBoardFinder: moving to search position")
        self.policy.set_pose_target(
            move_robot=self.move_robot,
            pose=self.current_pose,
        )
        self.policy.sleep_for(3.0)
        self.logger.info("TaskBoardFinder: reached search position")

    def _move_by_xy(self, dx: float, dy: float):
        """Move robot by dx, dy. Z stays fixed."""
        self.current_pose.position.x += dx
        self.current_pose.position.y += dy
        self.policy.set_pose_target(
            move_robot=self.move_robot,
            pose=self.current_pose,
        )
        self.policy.sleep_for(1.5)

    def _move_down_by(self, dz: float):
        """Move robot down by dz. X,Y stay fixed."""
        self.current_pose.position.z -= dz  # z decreases = moving down
        self.policy.set_pose_target(
            move_robot=self.move_robot,
            pose=self.current_pose,
        )
        self.policy.sleep_for(1.5)

    # ─────────────────────────────────────────
    # PRIVATE — DETECTION
    # ─────────────────────────────────────────

    def _get_center_image(self):
        """Get latest center camera image as OpenCV BGR."""
        observation = self.get_observation()
        if observation is None:
            return None
        try:
            return self.bridge.imgmsg_to_cv2(
                observation.center_image, desired_encoding="bgr8"
            )
        except Exception as e:
            self.logger.error(f"TaskBoardFinder: image conversion failed: {e}")
            return None

    def _detect_taskboard(self, image):
        """
        Run YOLO on image, find taskboard bounding box.

        Returns:
            dict with keys: cx, cy, width, height
            None if not detected
        """
        results = self.model(image, conf=self.confidence, verbose=False)

        for box in results[0].boxes:
            class_id   = int(box.cls[0])
            class_name = self.model.names[class_id]

            if class_name == self.class_name:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                return {
                    "cx":     (x1 + x2) / 2,
                    "cy":     (y1 + y2) / 2,
                    "width":  x2 - x1,
                    "height": y2 - y1,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                }
        return None

    def _get_pixel_error(self, detection, image):
        """
        Calculate pixel error between taskboard center and image center.

        Returns:
            (error_x, error_y) in pixels
            positive error_x = taskboard is to the right of center
            positive error_y = taskboard is below center
        """
        img_h, img_w = image.shape[:2]
        img_cx = img_w / 2
        img_cy = img_h / 2
        error_x = detection["cx"] - img_cx
        error_y = detection["cy"] - img_cy
        return error_x, error_y

    # ─────────────────────────────────────────
    # PRIVATE — SEARCH + CENTER + DESCEND
    # ─────────────────────────────────────────

    def _find_taskboard_with_scan(self) -> bool:
        """
        Try to detect taskboard. If not visible, do a scan pattern.
        Returns True if taskboard found.
        """
        # First try at current position
        image = self._get_center_image()
        if image is not None and self._detect_taskboard(image) is not None:
            self.logger.info("TaskBoardFinder: taskboard detected immediately")
            return True

        # Scan pattern: left, right, forward, back
        step   = self.cfg.get("search_scan_step_m", 0.05)
        max_attempts = self.cfg.get("max_search_attempts", 8)

        scan_offsets = [
            ( step,  0),    # right
            (-step,  0),    # left
            ( 0,     step), # forward
            ( 0,    -step), # backward
            ( step,  step),
            (-step,  step),
            ( step, -step),
            (-step, -step),
        ]

        for i, (dx, dy) in enumerate(scan_offsets[:max_attempts]):
            self.logger.info(f"TaskBoardFinder: scan attempt {i+1} dx={dx} dy={dy}")
            self._move_by_xy(dx, dy)

            image = self._get_center_image()
            if image is None:
                continue

            detection = self._detect_taskboard(image)
            if detection is not None:
                self.logger.info(f"TaskBoardFinder: taskboard found on scan attempt {i+1}")
                return True

        return False

    def _center_and_descend(self) -> bool:
        """
        Loop:
        1. Detect taskboard
        2. Correct X,Y if not centered
        3. Move down one Z step
        4. Repeat until bbox wide enough
        """
        target_width  = self.cfg.get("target_bbox_width_px", 400)
        center_thresh = self.cfg.get("center_threshold_px", 100)
        xy_step       = self.cfg.get("xy_correction_step_m", 0.05)
        z_step        = self.cfg.get("z_step_m", 0.02)
        min_z         = self.cfg.get("min_z_m", 0.15)

        while True:
            # Get image
            image = self._get_center_image()
            if image is None:
                self.logger.warn("TaskBoardFinder: no image received")
                self.policy.sleep_for(0.5)
                continue

            # Detect taskboard
            detection = self._detect_taskboard(image)
            if detection is None:
                self.logger.warn("TaskBoardFinder: taskboard lost during descent")
                # try moving back up slightly
                self.current_pose.position.z += z_step
                self.policy.set_pose_target(
                    move_robot=self.move_robot,
                    pose=self.current_pose,
                )
                self.policy.sleep_for(1.5)
                continue

            bbox_width = detection["width"]
            self.logger.info(
                f"TaskBoardFinder: bbox_width={bbox_width:.0f}px "
                f"z={self.current_pose.position.z:.3f}m"
            )

            # Check if close enough
            if bbox_width >= target_width:
                self.logger.info(
                    f"TaskBoardFinder: close enough! "
                    f"bbox_width={bbox_width:.0f}px >= {target_width}px"
                )
                return True

            # Safety check
            if self.current_pose.position.z <= min_z:
                self.logger.error(
                    f"TaskBoardFinder: reached min height {min_z}m — stopping"
                )
                return False

            # Step 1: Correct X,Y centering
            error_x, error_y = self._get_pixel_error(detection, image)
            self.logger.info(
                f"TaskBoardFinder: pixel error x={error_x:.0f} y={error_y:.0f}"
            )

            if abs(error_x) > center_thresh or abs(error_y) > center_thresh:
                # Convert pixel error direction to robot movement direction
                # error_x > 0 = taskboard is right of center → move robot right
                # error_y > 0 = taskboard is below center → move robot forward
                dx = xy_step if error_x > 0 else -xy_step if error_x < -center_thresh else 0.0
                dy = xy_step if error_y > 0 else -xy_step if error_y < -center_thresh else 0.0

                if dx != 0.0 or dy != 0.0:
                    self.logger.info(f"TaskBoardFinder: correcting xy dx={dx} dy={dy}")
                    self._move_by_xy(dx, dy)
                    continue  # re-detect before moving down

            # Step 2: Move down
            self.logger.info(f"TaskBoardFinder: moving down {z_step}m")
            self._move_down_by(z_step)