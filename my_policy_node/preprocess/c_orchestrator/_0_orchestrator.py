#!/usr/bin/env python3
#
# _0_orchestrator.py
# Full insertion pipeline:
#   init → circle search → center port → align port → center again → tilt arm → move down
# Camera stream runs headless (detection + publish, no display window) in background threads.
#
# HOW TO RUN:
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/c_orchestrator/_0_orchestrator.py --port sfp
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/c_orchestrator/_0_orchestrator.py --port sfp --target-z 0.05
#

import os
import sys
import argparse
import threading
import time

# Allow imports from preprocess/ subdirectories as packages
_PREPROCESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PREPROCESS_DIR not in sys.path:
    sys.path.insert(0, _PREPROCESS_DIR)

import rclpy

from a_image_processing._1_camera_stream import CameraStream
from b_move_robot._0_move_to_init import MoveToInit
from b_move_robot._1_circle_move import CircleMover
from b_move_robot._2_move_to_port import MoveToPort
from b_move_robot._3_2_align_port_edge import AlignPortEdge
from b_move_robot._4_1_tilt_arm import TiltArm
from b_move_robot._4_move_down import MoveDown
from b_move_robot._5_1_micro_alignment import MicroAlignment


def start_camera_stream(port_type: str):
    """
    Start CameraStream node in background threads (headless — no OpenCV window).
    Returns (node, stop_event). Call stop_event.set() to stop the detection loop.
    """
    node = CameraStream(port_type=port_type)

    # ROS spin thread — processes image subscriptions
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    stop_event = threading.Event()

    # Detection loop thread — runs YOLO and publishes /port_detection at ~30fps
    def _detection_loop():
        while not stop_event.is_set():
            image = node.get_active_image()
            if image is not None:
                node.run_detection(image)
            time.sleep(0.033)

    detect_thread = threading.Thread(target=_detection_loop, daemon=True)
    detect_thread.start()

    return node, stop_event


def step1_move_to_init() -> bool:
    """Move robot to initial/spawn position."""
    print("\n=== STEP 1: Move to init position ===")
    node = MoveToInit()
    deadline = time.time() + 10.0
    while node.current_joints is None and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.current_joints is None:
        node.get_logger().error("No joint state received — aborting")
        node.destroy_node()
        return False
    node.move_to_init()
    node.destroy_node()
    print("✓ Step 1 complete — at init position")
    return True


def step2_circle_search(port_type: str) -> bool:
    """Circle the robot to search for port. Returns True if port detected."""
    print(f"\n=== STEP 2: Circle search for {port_type} port ===")
    node = CircleMover()
    found = node.trace_circle(
        center_x=-0.5,
        center_y=0.4,
        radius=0.2,
        height=0.3,
        angle_step_deg=30.0,
        direction=-1,
        port_type=port_type,
    )
    node.destroy_node()
    if found:
        print("✓ Step 2 complete — port detected")
    else:
        print("✗ Step 2: port not detected after full circle")
    return found


def step3_center_port(port_type: str) -> bool:
    """Center port in camera frame using visual servoing."""
    print(f"\n=== STEP 3: Center {port_type} port in camera ===")
    node = MoveToPort(port_type=port_type)
    result = node.run()
    node.destroy_node()
    if result:
        print("✓ Step 3 complete — port centered")
    else:
        print("✗ Step 3: centering failed")
    return result


def step4_align_port(port_type: str, camera: str = "right") -> bool:
    """Align port orientation using YOLO polygon (rotate wrist_3)."""
    print(f"\n=== STEP 4: Align {port_type} port orientation ===")
    print(f"  Using camera: {camera}")
    node = AlignPortEdge(port_type=port_type, camera=camera)
    result = node.run()
    node.destroy_node()
    if result:
        print("✓ Step 4 complete — port aligned")
    else:
        print("✗ Step 4: alignment failed")
    return result


def step5_center_port_again(port_type: str) -> bool:
    """Re-center port after alignment (wrist rotation may have shifted view)."""
    print(f"\n=== STEP 5: Re-center {port_type} port in camera ===")
    node = MoveToPort(port_type=port_type)
    result = node.run()
    node.destroy_node()
    if result:
        print("✓ Step 5 complete — port re-centered")
    else:
        print("✗ Step 5: re-centering failed")
    return result


def step6_tilt_arm() -> bool:
    """Tilt arm so connector is perpendicular to ground (straight down)."""
    print("\n=== STEP 6: Tilt arm perpendicular to ground ===")
    node = TiltArm()
    result = node.run()
    node.destroy_node()
    if result:
        print("✓ Step 6 complete — arm tilted")
    else:
        print("✗ Step 6: tilt failed")
    return result


def step7_move_down(port_type: str, target_z: float) -> bool:
    """Move down with continuous XY centering until target Z or force contact."""
    print(f"\n=== STEP 7: Move down to Z={target_z}m ===")
    node = MoveDown(port_type=port_type, target_z=target_z)
    result = node.run()
    node.destroy_node()
    if result:
        print("✓ Step 7 complete — target reached")
    else:
        print("✗ Step 7: move down failed or interrupted")
    return result


def step8_micro_alignment(port_type: str, camera: str = "center") -> bool:
    """Micro alignment - position port so connector tip is at camera center."""
    print(f"\n=== STEP 8: Micro alignment for {port_type} ===")
    print(f"  Using camera: {camera}")
    node = MicroAlignment(port_type=port_type, camera=camera)
    result = node.run()
    node.destroy_node()
    if result:
        print("✓ Step 8 complete — micro aligned")
    else:
        print("✗ Step 8: micro alignment failed")
    return result


def run_pipeline(port_type: str, target_z: float, align_camera: str = "right", micro_camera: str = "center"):
    """
    Run full cable insertion pipeline.

    Pipeline:
        1. Move to init position
        2. Circle search for port
        3. Center port in camera
        4. Align port orientation
        5. Re-center port (after wrist rotation)
        6. Tilt arm perpendicular
        7. Move down to target Z
        8. Micro alignment (connector tip positioning)
    """
    rclpy.init()

    camera_node = None
    camera_stop = None

    try:
        # Start camera stream in background (headless)
        print("\n=== Starting camera stream (headless) ===")
        camera_node, camera_stop = start_camera_stream(port_type)
        print(f"✓ Camera stream running for {port_type} port")

        # Give the camera stream a moment to subscribe and start receiving images
        time.sleep(2.0)

        if not step1_move_to_init():
            print("Pipeline aborted at step 1.")
            return

        if not step2_circle_search(port_type):
            print("Port not found. Pipeline aborted.")
            return

        if not step3_center_port(port_type):
            print("Port centering failed. Pipeline aborted.")
            return

        if not step4_align_port(port_type, camera=align_camera):
            print("Port alignment failed. Pipeline aborted.")
            return

        if not step5_center_port_again(port_type):
            print("Port re-centering failed. Pipeline aborted.")
            return

        if not step6_tilt_arm():
            print("Arm tilt failed. Pipeline aborted.")
            return

        if not step7_move_down(port_type, target_z):
            print("Move down failed or interrupted. Pipeline aborted.")
            return

        if not step8_micro_alignment(port_type, camera=micro_camera):
            print("Micro alignment failed. Pipeline aborted.")
            return

        print("\n" + "="*50)
        print("=== PIPELINE COMPLETE ===")
        print("="*50)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        if camera_stop:
            camera_stop.set()
        if camera_node:
            camera_node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Run full cable insertion pipeline")
    parser.add_argument(
        "--port", choices=["sfp", "sc"], default="sfp",
        help="Port type to insert (sfp or sc)"
    )
    parser.add_argument(
        "--target-z", type=float, default=0.23,
        help="Target Z position in meters for descent (default: 0.23)"
    )
    parser.add_argument(
        "--align-camera", choices=["center", "left", "right"], default="right",
        help="Camera to use for alignment (default: right)"
    )
    parser.add_argument(
        "--micro-camera", choices=["center", "left", "right"], default="center",
        help="Camera to use for micro alignment (default: center)"
    )
    args = parser.parse_args()

    print(f"Starting pipeline: port={args.port}, target_z={args.target_z}m")
    print(f"  align_camera={args.align_camera}, micro_camera={args.micro_camera}")
    run_pipeline(args.port, args.target_z, args.align_camera, args.micro_camera)


if __name__ == "__main__":
    main()