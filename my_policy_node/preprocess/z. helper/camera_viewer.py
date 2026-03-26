#!/usr/bin/env python3
#
# live_detection.py
# Shows live camera feed with YOLO detections overlaid.
# Run this while simulation is running to verify your model works.
#
# USAGE:
#   # Make sure simulation is running first, then:
#   source ~/yolo_env/bin/activate
#   pixi run python3 live_detection.py
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
from cv_bridge import CvBridge
from ultralytics import YOLO
import threading
import os

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
MODEL_PATH = "/home/ridhwana/ws_aic/src/aic/my_policy_node/my_policy_node/preprocess/a_image_processing/yolov8n.pt"
CONFIDENCE = 0.5

CAMERA_TOPICS = {
    "1_center": "/center_camera/image",
    "2_left":   "/left_camera/image",
    "3_right":  "/right_camera/image",
}

# Color per class (BGR)
CLASS_COLORS = {
    "task_board": (0,   255,  0),    # green
    "sfp_port":   (255, 0,    0),    # blue
    "sc_port":    (0,   0,  255),    # red
}


class LiveDetection(Node):

    def __init__(self):
        super().__init__("live_detection")
        self.bridge  = CvBridge()
        self.model   = YOLO(MODEL_PATH)
        self.images  = {name: None for name in CAMERA_TOPICS}
        self.active  = "1_center"
        self.lock    = threading.Lock()

        # Subscribe to all cameras
        for name, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image, topic,
                lambda msg, n=name: self._image_callback(msg, n),
                10
            )

        self.get_logger().info("LiveDetection: ready — press 1/2/3 to switch camera, q to quit")

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
        """Run YOLO and draw boxes on image."""
        results = self.model(image, conf=CONFIDENCE, verbose=False)
        annotated = image.copy()

        img_h, img_w = image.shape[:2]
        cx, cy = img_w // 2, img_h // 2

        # Draw crosshair at image center
        cv2.line(annotated, (cx-20, cy), (cx+20, cy), (255,255,255), 1)
        cv2.line(annotated, (cx, cy-20), (cx, cy+20), (255,255,255), 1)

        for box in results[0].boxes:
            class_id   = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2
            box_w  = x2 - x1
            box_h  = y2 - y1

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

        # Draw camera name and active indicator
        cv2.putText(annotated, f"Camera: {self.active} | press 1/2/3 to switch | q=quit",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Draw detection count
        n = len(results[0].boxes)
        cv2.putText(annotated, f"Detections: {n}",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return annotated


def main():
    rclpy.init()
    node = LiveDetection()

    # Spin ROS in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("Waiting for camera images...")
    cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Detection", 800, 600)

    while rclpy.ok():
        image = node.get_active_image()

        if image is None:
            # Show waiting screen
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera...",
                       (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Live Detection", blank)
        else:
            annotated = node.run_detection(image)
            cv2.imshow("Live Detection", annotated)

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