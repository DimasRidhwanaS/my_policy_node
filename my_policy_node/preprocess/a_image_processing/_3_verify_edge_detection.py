#!/usr/bin/env python3
#
# _3_verify_edge_detection.py
# Simple approach: Use YOLO polygon directly - find top-left and top-right corners
# Single window with step-by-step visualization (press 1-4 to switch steps)
#
# Run:
#   Terminal 1: camera stream
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sfp --camera right
#
#   Terminal 2: verify edge detection
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_3_verify_edge_detection.py --port sfp --camera right
#

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import json
import threading
import argparse

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


CAMERA_TOPICS = {
    "center": "/center_camera/image",
    "left": "/left_camera/image",
    "right": "/right_camera/image",
}

# Camera angle offsets (degrees) - relative to center camera
# These compensate for camera mounting angle
CAMERA_ANGLE_OFFSETS = {
    "center": 0.0,
    "left": -30.0,   # Left camera sees port from left side
    "right": 30.0,   # Right camera sees port from right side
}


def put_text_with_background(img, text, pos, font, scale, color, thickness=1, bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for readability."""
    x, y = pos
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


class VerifyEdgeDetectionV2(Node):
    """Simple approach: Use YOLO polygon vertices directly for angle."""

    def __init__(self, port_type: str = "sfp", camera: str = "right"):
        super().__init__("verify_edge_detection_v2")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.camera = camera.lower()

        if self.camera not in CAMERA_TOPICS:
            raise ValueError(f"Invalid camera: {self.camera}")

        self.camera_topic = CAMERA_TOPICS[self.camera]

        self.bridge = CvBridge()
        self.latest_detection = None
        self.detection_lock = threading.Lock()
        self.latest_image = None
        self.image_lock = threading.Lock()

        # Current step to display (0 = final result, 1-4 = intermediate steps)
        self.current_step = 0

        self.create_subscription(String, "/port_detection", self.detection_callback, 10)
        self.create_subscription(Image, self.camera_topic, self.image_callback, 10)

        self.get_logger().info(f"V2: YOLO Polygon Corners - Simple & Stable")
        self.get_logger().info(f"  Camera: {self.camera} ({self.camera_topic})")
        self.get_logger().info(f"  Port: {self.target_class}")
        self.get_logger().info("  Press 0-4 to switch views")

    def detection_callback(self, msg):
        try:
            detection = json.loads(msg.data)
            with self.detection_lock:
                self.latest_detection = detection
        except Exception as e:
            self.get_logger().error(f"Parse error: {e}")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.latest_image = image
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")

    def process_image(self, image, detection):
        """Process image using YOLO polygon - find 4 rectangle corners."""
        if detection is None or not detection.get("detected", False):
            return None, {}

        # Get polygon from YOLO detection
        polygon = detection.get("polygon", None)
        cx = int(detection.get("cx", 0))
        cy = int(detection.get("cy", 0))

        if polygon is None or len(polygon) < 3:
            return None, {}

        polygon_np = np.array(polygon, dtype=np.int32)

        # Store all steps
        steps = {}

        # STEP 1: Show polygon on image
        steps[1] = image.copy()
        cv2.polylines(steps[1], [polygon_np], True, (0, 255, 0), 2)
        cv2.circle(steps[1], (cx, cy), 5, (0, 0, 255), -1)
        put_text_with_background(steps[1], "STEP 1: YOLO Polygon", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        put_text_with_background(steps[1], f"Vertices: {len(polygon)}", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # STEP 2: Find 4 corners of the rectangle
        # Use approxPolyDP to simplify polygon to quadrilateral
        epsilon = 0.02 * cv2.arcLength(polygon_np, True)
        approx = cv2.approxPolyDP(polygon_np, epsilon, True)

        # If approx has more than 4 points, use convex hull and try again
        if len(approx) > 4:
            hull = cv2.convexHull(polygon_np)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

        # If still more than 4, take the 4 extreme points
        if len(approx) > 4:
            # Find 4 corners using bounding box approach
            rect = cv2.minAreaRect(polygon_np)
            box = cv2.boxPoints(rect)
            approx = box.astype(np.int32)

        if len(approx) < 4:
            return None, {}

        # Reshape to (N, 2) format
        corners = approx.reshape(-1, 2)

        # Order corners: top-left, top-right, bottom-right, bottom-left
        # Sort by Y (top first)
        sorted_by_y = sorted(corners, key=lambda p: p[1])
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]

        # Sort top two by X (left first)
        top_two_sorted = sorted(top_two, key=lambda p: p[0])
        top_left = tuple(top_two_sorted[0])
        top_right = tuple(top_two_sorted[1])

        # Sort bottom two by X (left first)
        bottom_two_sorted = sorted(bottom_two, key=lambda p: p[0])
        bottom_left = tuple(bottom_two_sorted[0])
        bottom_right = tuple(bottom_two_sorted[1])

        ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])

        steps[2] = image.copy()
        cv2.polylines(steps[2], [polygon_np], True, (0, 255, 0), 1)
        # Draw all approx points
        for pt in corners:
            cv2.circle(steps[2], tuple(pt), 5, (0, 255, 255), -1)
        # Draw ordered corners with labels
        cv2.circle(steps[2], top_left, 10, (255, 0, 0), -1)
        cv2.circle(steps[2], top_right, 10, (0, 0, 255), -1)
        cv2.circle(steps[2], bottom_left, 10, (255, 165, 0), -1)
        cv2.circle(steps[2], bottom_right, 10, (0, 255, 255), -1)
        cv2.putText(steps[2], "TL", (top_left[0] - 20, top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(steps[2], "TR", (top_right[0] + 5, top_right[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(steps[2], "BL", (bottom_left[0] - 20, bottom_left[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        cv2.putText(steps[2], "BR", (bottom_right[0] + 5, bottom_right[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        put_text_with_background(steps[2], "STEP 2: Find 4 Corners", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        put_text_with_background(steps[2], f"Blue=TL, Red=TR, Orange=BL, Cyan=BR", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # STEP 3: Find longest edge and calculate angle
        # Calculate all 4 edge lengths
        edges = [
            ("top", top_left, top_right),
            ("right", top_right, bottom_right),
            ("bottom", bottom_right, bottom_left),
            ("left", bottom_left, top_left),
        ]

        # Find longest edge
        longest_edge = max(edges, key=lambda e: np.sqrt((e[2][0] - e[1][0])**2 + (e[2][1] - e[1][1])**2))
        edge_name, p1, p2 = longest_edge[0], longest_edge[1], longest_edge[2]
        edge_len = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # Calculate angle from longest edge
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        raw_angle_deg = np.degrees(np.arctan2(dy, dx))

        # Apply camera angle offset to get true angle relative to robot
        camera_offset = CAMERA_ANGLE_OFFSETS.get(self.camera, 0.0)
        angle_deg = raw_angle_deg - camera_offset

        steps[3] = image.copy()
        cv2.polylines(steps[3], [polygon_np], True, (0, 255, 0), 1)
        cv2.polylines(steps[3], [ordered_corners], True, (0, 255, 0), 2)
        # Draw all edges
        cv2.line(steps[3], top_left, top_right, (255, 0, 0), 2)      # Blue - top
        cv2.line(steps[3], top_right, bottom_right, (0, 255, 0), 2)  # Green - right
        cv2.line(steps[3], bottom_right, bottom_left, (0, 0, 255), 2) # Red - bottom
        cv2.line(steps[3], bottom_left, top_left, (255, 255, 0), 2)   # Cyan - left
        # Highlight longest edge (yellow)
        cv2.line(steps[3], p1, p2, (0, 255, 255), 4)
        cv2.circle(steps[3], p1, 10, (0, 255, 255), -1)
        cv2.circle(steps[3], p2, 10, (0, 255, 255), -1)
        put_text_with_background(steps[3], "STEP 3: Longest Edge", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        put_text_with_background(steps[3], f"Edge: {edge_name} ({edge_len:.0f}px) | Angle: {angle_deg:.1f} deg", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # STEP 4: Final result with all info
        debug = image.copy()
        cv2.polylines(debug, [polygon_np], True, (0, 255, 0), 1)
        cv2.polylines(debug, [ordered_corners], True, (0, 255, 0), 2)
        cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(debug, p1, p2, (0, 255, 255), 3)
        cv2.circle(debug, p1, 10, (0, 255, 255), -1)
        cv2.circle(debug, p2, 10, (0, 255, 255), -1)

        # Status
        status = "ALIGNED" if abs(angle_deg) < 2 else f"ROTATE {angle_deg:+.1f} deg"
        color = (0, 255, 0) if abs(angle_deg) < 2 else (0, 165, 255)
        put_text_with_background(debug, f"Angle: {angle_deg:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        put_text_with_background(debug, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        put_text_with_background(debug, f"Edge: {edge_name} ({edge_len:.0f}px)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        put_text_with_background(debug, f"Polygon: {len(polygon)} pts -> 4 corners", (10, debug.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        steps[4] = debug
        steps[0] = debug

        return angle_deg, steps

    def run(self):
        """Display loop."""
        self.get_logger().info("Waiting for detection and image...")

        window_name = f"Polygon Corners - {self.camera.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 750)

        print("\n" + "="*60)
        print("POLYGON CORNERS - Rectangle Detection from YOLO Polygon")
        print("="*60)
        print(f"Camera: {self.camera.upper()}")
        print("")
        print("KEYBOARD CONTROLS:")
        print("  0 = Final result (angle overlay)")
        print("  1 = Step 1: YOLO Polygon")
        print("  2 = Step 2: Find 4 Corners (rectangle)")
        print("  3 = Step 3: Draw Line & Angle")
        print("  4 = Step 4: Final Result")
        print("  q = Quit")
        print("")
        print("Method:")
        print("  1. Get YOLO polygon (many vertices)")
        print("  2. Use approxPolyDP to find 4 corners (rectangle)")
        print("  3. Order corners: TL, TR, BR, BL")
        print("  4. Use top-left to top-right for angle")
        print("  -> Works for tilted ports!")
        print("="*60 + "\n")

        current_step = 0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            with self.detection_lock:
                det = self.latest_detection
            with self.image_lock:
                img = self.latest_image

            if det is None or img is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                put_text_with_background(blank, "Waiting for camera...",
                           (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                put_text_with_background(blank, f"Current view: STEP {current_step}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(window_name, blank)
            elif not det.get("detected", False):
                blank = img.copy()
                put_text_with_background(blank, "No port detected",
                           (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                put_text_with_background(blank, f"Current view: STEP {current_step}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(window_name, blank)
            elif det.get("port_type", "") != self.target_class:
                blank = img.copy()
                put_text_with_background(blank, f"Wrong port: {det.get('port_type')}",
                           (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                put_text_with_background(blank, f"Current view: STEP {current_step}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(window_name, blank)
            else:
                angle, steps = self.process_image(img, det)

                if current_step in steps:
                    display = steps[current_step].copy()
                    cv2.imshow(window_name, display)
                else:
                    # No result, show error
                    blank = img.copy()
                    put_text_with_background(blank, "Processing failed (no polygon?)", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(window_name, blank)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key >= ord('0') and key <= ord('4'):
                current_step = int(chr(key))
                print(f"\nSwitched to STEP {current_step}")

        print("\n")
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Verify polygon corners approach")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp")
    parser.add_argument("--camera", choices=["center", "left", "right"], default="right")
    args = parser.parse_args()

    rclpy.init()
    node = VerifyEdgeDetectionV2(
        port_type=args.port,
        camera=args.camera
    )

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()