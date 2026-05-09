#!/usr/bin/env python3
#
# _1_camera_stream.py
# Live camera feed with YOLO detections overlaid.
# Publishes detection results to /port_detection topic.
#
# USAGE:
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sc
#   cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sfp
#
# OUTPUT TOPICS:
#   /port_detection (std_msgs/String) - JSON with detection results
#
# CONTROLS:
#   q → quit
#   1 → center camera
#   2 → left camera
#   3 → right camera

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import threading
import os
import json
import argparse

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")
CONFIDENCE = 0.5

CAMERA_TOPICS = {
    "1_center": "/center_camera/image",
    "2_left":   "/left_camera/image",
    "3_right":  "/right_camera/image",
}

# Color per class (BGR)
CLASS_COLORS = {
    "task_board":    (0,   255,  0),    # green
    "sfp_port":       (255, 0,    0),    # blue
    "sfp_port_hole":  (255, 128,  0),   # cyan
    "sc_port":        (0,   0, 255),     # red
    "sc_port_hole":   (0,   255, 255),  # yellow
}

# Port classes we care about
PORT_CLASSES = {"sfp_port", "sc_port"}


class CameraStream(Node):

    def __init__(self, port_type: str = None):
        super().__init__("camera_stream")
        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH)
        self.images = {name: None for name in CAMERA_TOPICS}
        self.active = "1_center"
        self.lock = threading.Lock()

        # Port type filter (e.g., "sfp" or "sc")
        self.port_type = port_type
        self.target_class = f"{port_type}_port" if port_type else None

        # Publisher for detection results
        self.detection_pub = self.create_publisher(String, "/port_detection", 10)

        # Subscribe to all cameras
        for name, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image, topic,
                lambda msg, n=name: self._image_callback(msg, n),
                10
            )

        if self.port_type:
            self.get_logger().info(f"CameraStream: filtering for {self.target_class}")
        self.get_logger().info(f"CameraStream: ready — model loaded from {MODEL_PATH}")
        self.get_logger().info(f"  Model classes: {self.model.names}")
        self.get_logger().info("  Press 1/2/3 to switch camera, q to quit")
        self.get_logger().info("  Publishing detection to: /port_detection")

    def _image_callback(self, msg, camera_name):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.images[camera_name] = image
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")

    def get_active_image(self):
        with self.lock:
            return self.images.get(self.active, None)

    def run_detection(self, image):
        """Run YOLO, draw boxes, and publish detection results."""
        results = self.model(image, conf=CONFIDENCE, verbose=False)
        annotated = image.copy()

        img_h, img_w = image.shape[:2]
        cx, cy = img_w // 2, img_h // 2

        # Draw center crosshair (more visible)
        cv2.line(annotated, (cx-30, cy), (cx+30, cy), (0, 255, 255), 2)  # yellow horizontal
        cv2.line(annotated, (cx, cy-30), (cx, cy+30), (0, 255, 255), 2)  # yellow vertical
        cv2.circle(annotated, (cx, cy), 15, (0, 255, 255), 2)  # yellow circle
        cv2.putText(annotated, "CENTER", (cx+20, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Find best port detection
        best_port = None
        best_conf = 0.0

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2
            box_w = x2 - x1
            box_h = y2 - y1

            # Get color for this class
            color = CLASS_COLORS.get(class_name, (200, 200, 200))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw center dot
            cv2.circle(annotated, (box_cx, box_cy), 4, color, -1)

            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw pixel error from image center
            error_x = box_cx - cx
            error_y = box_cy - cy
            error_text = f"err x:{error_x:+d} y:{error_y:+d}"
            cv2.putText(annotated, error_text, (x1, y2+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw bbox size
            size_text = f"w:{box_w}px h:{box_h}px"
            cv2.putText(annotated, size_text, (x1, y2+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Track best port detection
            # If filtering by port type, only consider matching class
            if self.target_class:
                if class_name == self.target_class and confidence > best_conf:
                    best_port = {
                        "port_type": class_name,
                        "cx": float(box_cx),
                        "cy": float(box_cy),
                        "confidence": confidence,
                        "img_width": img_w,
                        "img_height": img_h,
                        "detected": True,
                    }
                    best_conf = confidence
            else:
                # No filter - track any port class
                if class_name in PORT_CLASSES and confidence > best_conf:
                    best_port = {
                        "port_type": class_name,
                        "cx": float(box_cx),
                        "cy": float(box_cy),
                        "confidence": confidence,
                        "img_width": img_w,
                        "img_height": img_h,
                        "detected": True,
                    }
                    best_conf = confidence

        # Draw arrow from center to target port AND X/Y component arrows
        if best_port is not None:
            port_cx = int(best_port["cx"])
            port_cy = int(best_port["cy"])

            # Calculate arrow direction
            error_x = port_cx - cx
            error_y = port_cy - cy
            distance = np.sqrt(error_x**2 + error_y**2)

            # Draw main arrow from center to port (green - combined direction)
            arrow_length = min(distance * 0.5, 100)
            if distance > 0:
                arrow_end_x = int(cx + (error_x / distance) * arrow_length)
                arrow_end_y = int(cy + (error_y / distance) * arrow_length)
            else:
                arrow_end_x = cx
                arrow_end_y = cy
            cv2.arrowedLine(annotated, (cx, cy), (arrow_end_x, arrow_end_y),
                           (0, 255, 0), 3, tipLength=0.3)

            # Draw X component arrow (RED - horizontal error)
            # Shows how much to move left/right
            x_scale = 0.3  # Scale factor for visualization
            x_arrow_end = int(cx + error_x * x_scale)
            if abs(error_x) > 10:  # Only draw if significant
                cv2.arrowedLine(annotated, (cx, cy), (x_arrow_end, cy),
                               (0, 0, 255), 2, tipLength=0.2)  # Red for X
                x_label = f"X: {error_x:+d}px"
                cv2.putText(annotated, x_label, (x_arrow_end + 5, cy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Draw Y component arrow (BLUE - vertical error)
            # Shows how much to move up/down
            y_scale = 0.3
            y_arrow_end = int(cy + error_y * y_scale)
            if abs(error_y) > 10:  # Only draw if significant
                cv2.arrowedLine(annotated, (cx, cy), (cx, y_arrow_end),
                               (255, 0, 0), 2, tipLength=0.2)  # Blue for Y
                y_label = f"Y: {error_y:+d}px"
                cv2.putText(annotated, y_label, (cx + 5, y_arrow_end - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Draw direction text
            direction = ""
            if abs(error_x) > 20 or abs(error_y) > 20:
                if abs(error_x) > abs(error_y):
                    direction = "LEFT" if error_x < 0 else "RIGHT"
                else:
                    direction = "UP" if error_y < 0 else "DOWN"
            else:
                direction = "CENTERED"

            cv2.putText(annotated, f"→ {direction}", (cx + 50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw legend
            cv2.putText(annotated, "Green: Combined | Red: X (L/R) | Blue: Y (U/D)",
                       (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Publish detection result
        if best_port is not None:
            msg = String()
            msg.data = json.dumps(best_port)
            self.detection_pub.publish(msg)
        else:
            # No port detected
            no_detection = {
                "port_type": None,
                "cx": 0.0,
                "cy": 0.0,
                "confidence": 0.0,
                "img_width": img_w,
                "img_height": img_h,
                "detected": False,
            }
            msg = String()
            msg.data = json.dumps(no_detection)
            self.detection_pub.publish(msg)

        # Draw camera name and active indicator
        filter_text = f" | Filter: {self.target_class}" if self.target_class else ""
        cv2.putText(annotated, f"Camera: {self.active}{filter_text} | 1/2/3=switch | q=quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Draw detection count
        n = len(results[0].boxes)
        cv2.putText(annotated, f"Detections: {n}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Draw port detection status
        if best_port:
            cv2.putText(annotated, f"Target: {best_port['port_type']} ({best_port['confidence']:.2f})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            if self.target_class:
                cv2.putText(annotated, f"No {self.target_class} detected",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(annotated, "No port detected",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return annotated


def main():
    parser = argparse.ArgumentParser(description="Camera stream with YOLO detection and port filtering")
    parser.add_argument("--port", choices=["sfp", "sc"], default=None, help="Port type to filter (sfp or sc). If not specified, shows all ports.")
    args = parser.parse_args()

    rclpy.init()
    node = CameraStream(port_type=args.port)

    # Spin ROS in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    if args.port:
        print(f"Filtering for {args.port.upper()} port...")
    print("Waiting for camera images...")
    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Stream", 800, 600)

    while rclpy.ok():
        image = node.get_active_image()

        if image is None:
            # Show waiting screen
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera...",
                        (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Camera Stream", blank)
        else:
            annotated = node.run_detection(image)
            cv2.imshow("Camera Stream", annotated)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            node.active = "1_center"
            print("Switched to center camera")
        elif key == ord('2'):
            node.active = "2_left"
            print("Switched to left camera")
        elif key == ord('3'):
            node.active = "3_right"
            print("Switched to right camera")

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()