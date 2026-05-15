#!/usr/bin/env python3
#
# _5_1_verify_micro_alignment.py
# Visualize micro alignment: show connector position and port tip.
# Use this to verify the offset parameters before running alignment.
#
# Run:
#   Terminal 1: camera stream
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sfp --camera center
#
#   Terminal 2: verify micro alignment
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_5_1_verify_micro_alignment.py --port sfp --camera center
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

# Connector position ratio in camera image (adjust for your setup)
# Connector is BELOW center (larger y = lower in image)
# 1/6 from middle to bottom: 0.5 + 0.5/6 ≈ 0.58
CONNECTOR_POSITION = {
    "center": {
        "x_ratio": 0.5,    # Horizontally centered
        "y_ratio": 0.605,   # Below center (adjust for your setup)
    },
    "left": {
        "x_ratio": 0.5,
        "y_ratio": 0.58,
    },
    "right": {
        "x_ratio": 0.5,
        "y_ratio": 0.58,
    },
}

# Port tip offset from port center (as ratio of bbox dimensions)
PORT_TIP_OFFSET = {
    "sfp": {
        "x_ratio": 0.0,
        "y_ratio": -0.2,   # Adjust for actual geometry
    },
    "sc": {
        "x_ratio": 0.0,
        "y_ratio": 0.0,
    },
}


def put_text_with_background(img, text, pos, font, scale, color, thickness=1, bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for readability."""
    x, y = pos
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


class VerifyMicroAlignment(Node):
    """Visualize micro alignment targets."""

    def __init__(self, port_type: str = "sfp", camera: str = "center"):
        super().__init__("verify_micro_alignment")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.camera = camera.lower()

        if self.camera not in CAMERA_TOPICS:
            raise ValueError(f"Invalid camera: {self.camera}")

        self.camera_topic = CAMERA_TOPICS[self.camera]

        # Get positions
        self.connector_x_ratio = CONNECTOR_POSITION[self.camera]["x_ratio"]
        self.connector_y_ratio = CONNECTOR_POSITION[self.camera]["y_ratio"]
        self.tip_x_ratio = PORT_TIP_OFFSET[self.port_type]["x_ratio"]
        self.tip_y_ratio = PORT_TIP_OFFSET[self.port_type]["y_ratio"]

        self.bridge = CvBridge()
        self.latest_detection = None
        self.detection_lock = threading.Lock()
        self.latest_image = None
        self.image_lock = threading.Lock()

        self.create_subscription(String, "/port_detection", self.detection_callback, 10)
        self.create_subscription(Image, self.camera_topic, self.image_callback, 10)

        self.get_logger().info(f"Verify Micro Alignment - {self.camera.upper()} camera")
        self.get_logger().info(f"  Port: {self.target_class}")
        self.get_logger().info(f"  Connector position: ({self.connector_x_ratio:.2f} × W, {self.connector_y_ratio:.2f} × H)")
        self.get_logger().info(f"  Port tip offset: ({self.tip_x_ratio:.2f} × bbox_w, {self.tip_y_ratio:.2f} × bbox_h)")
        self.get_logger().info("  Press 'q' to quit")

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
        """Draw visualization."""
        debug = image.copy()
        img_h, img_w = debug.shape[:2]

        # Calculate connector position (fixed target)
        connector_x = int(img_w * self.connector_x_ratio)
        connector_y = int(img_h * self.connector_y_ratio)

        # Draw connector position (GREEN - the target)
        cv2.circle(debug, (connector_x, connector_y), 15, (0, 255, 0), 2)
        cv2.circle(debug, (connector_x, connector_y), 5, (0, 255, 0), -1)
        cv2.putText(debug, "CONNECTOR", (connector_x + 20, connector_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw image center (BLUE - reference)
        img_cx = img_w // 2
        img_cy = img_h // 2
        cv2.circle(debug, (img_cx, img_cy), 10, (255, 0, 0), 1)
        cv2.putText(debug, "CENTER", (img_cx + 15, img_cy - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if detection is None or not detection.get("detected", False):
            put_text_with_background(debug, "No detection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return debug

        port_type = detection.get("port_type", "")
        if port_type != self.target_class:
            put_text_with_background(debug, f"Wrong port: {port_type}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return debug

        # Get port center and bbox
        cx = int(detection.get("cx", 0))
        cy = int(detection.get("cy", 0))
        bbox = detection.get("bbox", None)
        polygon = detection.get("polygon", None)

        if bbox is None:
            put_text_with_background(debug, "No bbox", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return debug

        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Calculate port tip position
        port_tip_x = cx + self.tip_x_ratio * bbox_w
        port_tip_y = cy + self.tip_y_ratio * bbox_h

        # Draw port center (YELLOW)
        cv2.circle(debug, (cx, cy), 8, (0, 255, 255), -1)
        cv2.putText(debug, "PORT CENTER", (cx + 10, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw port tip (RED)
        cv2.circle(debug, (int(port_tip_x), int(port_tip_y)), 10, (0, 0, 255), -1)
        cv2.putText(debug, "PORT TIP", (int(port_tip_x) + 15, int(port_tip_y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw line from port tip to connector
        cv2.line(debug, (int(port_tip_x), int(port_tip_y)), (connector_x, connector_y), (255, 255, 0), 2)

        # Draw bbox
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Draw polygon if available
        if polygon is not None and len(polygon) >= 3:
            polygon_np = np.array(polygon, dtype=np.int32)
            cv2.polylines(debug, [polygon_np], True, (0, 255, 0), 1)

        # Calculate error
        error_x = port_tip_x - connector_x
        error_y = port_tip_y - connector_y

        # Status
        status = "ALIGNED" if (abs(error_x) < 10 and abs(error_y) < 10) else "MOVE"
        color = (0, 255, 0) if status == "ALIGNED" else (0, 165, 255)

        put_text_with_background(debug, f"Connector: ({connector_x}, {connector_y})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        put_text_with_background(debug, f"Port tip: ({port_tip_x:.0f}, {port_tip_y:.0f})", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        put_text_with_background(debug, f"Error: ({error_x:+.0f}px, {error_y:+.0f}px)", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        put_text_with_background(debug, f"Status: {status}", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Config info
        put_text_with_background(debug, f"Connector pos: ({self.connector_x_ratio:.2f}×W, {self.connector_y_ratio:.2f}×H)", (10, img_h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        put_text_with_background(debug, f"Tip offset: ({self.tip_x_ratio:.2f}×bbox_w, {self.tip_y_ratio:.2f}×bbox_h)", (10, img_h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return debug

    def run(self):
        """Display loop."""
        self.get_logger().info("Waiting for detection and image...")

        window_name = f"Micro Alignment - {self.camera.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 750)

        print("\n" + "="*60)
        print("MICRO ALIGNMENT VERIFICATION")
        print("="*60)
        print(f"Camera: {self.camera.upper()}")
        print(f"Port: {self.port_type.upper()}")
        print("")
        print("LEGEND:")
        print("  GREEN circle = Connector position (TARGET)")
        print("  BLUE circle  = Image center (reference)")
        print("  YELLOW dot   = Port center (from YOLO)")
        print("  RED dot      = Port tip (center + offset)")
        print("  YELLOW line  = Error (port tip → connector)")
        print("")
        print("Press 'q' to quit")
        print("="*60 + "\n")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            with self.detection_lock:
                det = self.latest_detection
            with self.image_lock:
                img = self.latest_image

            if img is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                put_text_with_background(blank, "Waiting for camera...",
                           (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, blank)
            else:
                display = self.process_image(img, det)
                cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Verify micro alignment visualization")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp")
    parser.add_argument("--camera", choices=["center", "left", "right"], default="center")
    args = parser.parse_args()

    rclpy.init()
    node = VerifyMicroAlignment(port_type=args.port, camera=args.camera)

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()