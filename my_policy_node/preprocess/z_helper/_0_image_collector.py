#!/usr/bin/env python3
#
# capture_cameras.py
# Captures images from left, center, and right cameras every second.
# Usage: pixi run python3 capture_cameras.py
# Images saved to: ./captured_images/TIMESTAMP_CAMERA.png

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from datetime import datetime


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_images")
CAPTURE_INTERVAL_SEC = 1.0  # capture every N seconds

CAMERA_TOPICS = {
    "left":   "/left_camera/image",
    "center": "/center_camera/image",
    "right":  "/right_camera/image",
}


class CameraCapture(Node):
    def __init__(self):
        super().__init__("camera_capture")
        self.bridge = CvBridge()
        self.latest_images = {name: None for name in CAMERA_TOPICS}
        self.count = 0

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.get_logger().info(f"Saving images to: {OUTPUT_DIR}")

        # Subscribe to all three cameras
        for name, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image,
                topic,
                lambda msg, n=name: self._image_callback(msg, n),
                10,
            )

        # Timer to capture every second
        self.create_timer(CAPTURE_INTERVAL_SEC, self._capture_callback)

    def _image_callback(self, msg: Image, camera_name: str):
        """Store the latest image for each camera."""
        try:
            self.latest_images[camera_name] = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert {camera_name} image: {e}")

    def _capture_callback(self):
        """Save the latest image from each camera."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
        saved = []
        missing = []

        for name, image in self.latest_images.items():
            if image is not None:
                filename = os.path.join(OUTPUT_DIR, f"{timestamp}_{name}.png")
                cv2.imwrite(filename, image)
                saved.append(name)
            else:
                missing.append(name)

        if saved:
            self.count += 1
            self.get_logger().info(
                f"[{self.count}] Saved: {', '.join(saved)}"
                + (f" | Missing: {', '.join(missing)}" if missing else "")
            )
        else:
            self.get_logger().warn("No images received yet from any camera.")


def main():
    rclpy.init()
    node = CameraCapture()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f"Stopped. Total captures: {node.count}")
        node.get_logger().info(f"Images saved to: {OUTPUT_DIR}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()