#!/usr/bin/env python3
#
# _2_verify_edge_detection.py
# Verify edge detection works before running alignment.
# Shows the detected edges and angle - NO ROBOT MOVEMENT.
#
# HOW TO RUN:
#   Terminal 1: camera stream (use --camera to select which camera publishes detection)
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_1_camera_stream.py --port sfp --camera right
#
#   Terminal 2: verify edge detection (use --camera to match)
#     cd /home/ridhwana/ws_aic/src/aic && pixi run python3 my_policy_node/my_policy_node/preprocess/a_image_processing/_2_verify_edge_detection.py --port sfp --camera right
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


# Camera topic mapping
CAMERA_TOPICS = {
    "center": "/center_camera/image",
    "left": "/left_camera/image",
    "right": "/right_camera/image",
}


class VerifyEdgeDetection(Node):
    """Show edge detection results without moving robot."""

    def __init__(self, port_type: str = "sfp", camera: str = "center", padding: int = 10, use_mask: bool = True):
        super().__init__("verify_edge_detection")

        self.port_type = port_type.lower()
        self.target_class = f"{self.port_type}_port"
        self.camera = camera.lower()

        if self.camera not in CAMERA_TOPICS:
            self.get_logger().error(f"Invalid camera: {self.camera}. Must be one of: {list(CAMERA_TOPICS.keys())}")
            raise ValueError(f"Invalid camera: {self.camera}")

        self.camera_topic = CAMERA_TOPICS[self.camera]

        self.bridge = CvBridge()
        self.latest_detection = None
        self.detection_lock = threading.Lock()
        self.latest_image = None
        self.image_lock = threading.Lock()

        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.blur_kernel = (5, 5)
        self.padding = padding  # Pixels to add around YOLO bbox
        self.use_mask = use_mask  # Use YOLO polygon to mask out background

        # Subscribe to detection (published by _1_camera_stream.py)
        self.create_subscription(String, "/port_detection", self.detection_callback, 10)

        # Subscribe to selected camera
        self.create_subscription(Image, self.camera_topic, self.image_callback, 10)

        self.get_logger().info(f"Verifying edge detection for {self.target_class}")
        self.get_logger().info(f"  Camera: {self.camera} ({self.camera_topic})")
        self.get_logger().info(f"  Padding: {self.padding}px around YOLO bbox")
        self.get_logger().info(f"  Use polygon mask: {self.use_mask}")
        self.get_logger().info("Watch the window - NO robot movement will occur")

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

    def detect_edges(self, image, detection):
        """Edge detection using YOLO bbox with padding."""
        if detection is None or not detection.get("detected", False):
            debug = image.copy()
            cv2.putText(debug, "NO DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None, None, None, None, debug

        # Get bounding box from YOLO detection
        bbox = detection.get("bbox", None)
        if bbox is None:
            # Fallback: estimate from center if no bbox
            cx = int(detection.get("cx", 0))
            cy = int(detection.get("cy", 0))
            port_type = detection.get("port_type", "sfp_port")
            half_w = 40 if "sfp" in port_type.lower() else 50
            half_h = 25 if "sfp" in port_type.lower() else 30
            x1, y1 = cx - half_w, cy - half_h
            x2, y2 = cx + half_w, cy + half_h
        else:
            x1, y1, x2, y2 = bbox

        # Add padding around YOLO bbox
        x1_m = max(0, x1 - self.padding)
        y1_m = max(0, y1 - self.padding)
        x2_m = min(image.shape[1], x2 + self.padding)
        y2_m = min(image.shape[0], y2 + self.padding)

        crop = image[y1_m:y2_m, x1_m:x2_m].copy()

        if crop.size == 0:
            debug = image.copy()
            cv2.putText(debug, "EMPTY CROP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None, None, None, None, debug

        # Optional: Use YOLO polygon to mask out background
        polygon = detection.get("polygon", None)
        if self.use_mask and polygon is not None:
            # Create mask from polygon
            pts = np.array(polygon, dtype=np.int32)

            # Shift polygon coordinates to crop coordinates
            pts[:, 0] -= x1_m  # shift x
            pts[:, 1] -= y1_m  # shift y

            # Create mask
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

            # Apply mask to crop
            crop_masked = cv2.bitwise_and(crop, crop, mask=mask)

            # Convert to grayscale
            gray = cv2.cvtColor(crop_masked, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # DEBUG: Show crop, edges, and contours in a separate window
        debug_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
        if len(debug_crop.shape) == 2:
            debug_crop = cv2.cvtColor(debug_crop, cv2.COLOR_GRAY2BGR)

        # Draw all contours on crop
        cv2.drawContours(debug_crop, contours, -1, (0, 255, 0), 1)

        # Add info text
        cv2.putText(debug_crop, f"Crop: {crop.shape[1]}x{crop.shape[0]}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(debug_crop, f"Contours: {len(contours)}", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(debug_crop, f"Mask: {self.use_mask}", (5, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Show debug windows
        cv2.imshow("Crop (Grayscale)", gray)
        cv2.imshow("Canny Edges", edges)
        cv2.imshow("Contours on Crop", debug_crop)

        if len(contours) < 2:
            debug = image.copy()
            cv2.rectangle(debug, (x1_m, y1_m), (x2_m, y2_m), (0, 255, 255), 2)
            cv2.putText(debug, "Not enough contours", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return None, None, None, None, debug

        center_x = crop.shape[1] // 2
        left_contours = [c for c in contours if cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2 < center_x]
        right_contours = [c for c in contours if cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2 >= center_x]

        if not left_contours or not right_contours:
            debug = image.copy()
            cv2.rectangle(debug, (x1_m, y1_m), (x2_m, y2_m), (0, 255, 255), 2)
            cv2.putText(debug, f"L:{len(left_contours)} R:{len(right_contours)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return None, None, None, None, debug

        # Filter by minimum area to avoid noise
        min_area = 20
        left_contours = [c for c in left_contours if cv2.contourArea(c) > min_area]
        right_contours = [c for c in right_contours if cv2.contourArea(c) > min_area]

        if not left_contours or not right_contours:
            debug = image.copy()
            cv2.rectangle(debug, (x1_m, y1_m), (x2_m, y2_m), (0, 255, 255), 2)
            cv2.putText(debug, f"Too small (filter: area>{min_area})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return None, None, None, None, debug

        left_largest = max(left_contours, key=cv2.contourArea)
        right_largest = max(right_contours, key=cv2.contourArea)

        M_left = cv2.moments(left_largest)
        M_right = cv2.moments(right_largest)

        if M_left["m00"] == 0 or M_right["m00"] == 0:
            debug = image.copy()
            cv2.putText(debug, "Zero moment", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return None, None, None, None, debug

        left_cx = int(M_left["m10"] / M_left["m00"])
        left_cy = int(M_left["m01"] / M_left["m00"])
        right_cx = int(M_right["m10"] / M_right["m00"])
        right_cy = int(M_right["m01"] / M_right["m00"])

        left_midpoint = (left_cx + x1_m, left_cy + y1_m)
        right_midpoint = (right_cx + x1_m, right_cy + y1_m)

        dx = right_midpoint[0] - left_midpoint[0]
        dy = right_midpoint[1] - left_midpoint[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Build debug image
        debug = image.copy()

        # Show crop region (tight)
        cv2.rectangle(debug, (x1_m, y1_m), (x2_m, y2_m), (0, 255, 255), 2)

        # Show center line (split left/right)
        center_line_x = x1_m + (x2_m - x1_m) // 2
        cv2.line(debug, (center_line_x, y1_m), (center_line_x, y2_m), (255, 255, 255), 1)

        # Draw all contours (gray)
        cv2.drawContours(debug[y1_m:y2_m, x1_m:x2_m], contours, -1, (100, 100, 100), 1)

        # Highlight left contours (light blue)
        cv2.drawContours(debug[y1_m:y2_m, x1_m:x2_m], left_contours, -1, (255, 200, 100), 1)

        # Highlight right contours (light red)
        cv2.drawContours(debug[y1_m:y2_m, x1_m:x2_m], right_contours, -1, (100, 200, 255), 1)

        # Draw selected edges (thick)
        cv2.drawContours(debug[y1_m:y2_m, x1_m:x2_m], [left_largest], -1, (255, 0, 0), 3)
        cv2.drawContours(debug[y1_m:y2_m, x1_m:x2_m], [right_largest], -1, (0, 0, 255), 3)

        # Draw centroids
        cv2.circle(debug, left_midpoint, 8, (255, 0, 0), -1)
        cv2.circle(debug, right_midpoint, 8, (0, 0, 255), -1)

        # Draw line connecting midpoints
        cv2.line(debug, left_midpoint, right_midpoint, (0, 255, 0), 3)

        # Labels
        cv2.putText(debug, "L", (left_midpoint[0] - 20, left_midpoint[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(debug, "R", (right_midpoint[0] + 10, right_midpoint[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Angle display
        cv2.putText(debug, f"Edge Angle: {angle_deg:.1f}°",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Status
        status = "ALIGNED" if abs(angle_deg) < 2 else f"ROTATE {angle_deg:+.1f} deg"
        color = (0, 255, 0) if abs(angle_deg) < 2 else (0, 165, 255)
        cv2.putText(debug, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Info
        cv2.putText(debug, f"Padding: {self.padding}px | Mask: {self.use_mask}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug, f"Contours: {len(contours)} | Left: {len(left_contours)} | Right: {len(right_contours)}",
                   (10, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug, f"Left area: {cv2.contourArea(left_largest):.0f} | Right area: {cv2.contourArea(right_largest):.0f}",
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return angle_deg, left_midpoint, right_midpoint, (x1_m, y1_m, x2_m, y2_m), debug

    def run(self):
        """Display loop."""
        self.get_logger().info("Waiting for detection and image...")

        window_name = f"Edge Detection - {self.camera.upper()} Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 750)

        print("\n" + "="*60)
        print("EDGE DETECTION VERIFICATION")
        print("="*60)
        print(f"Camera: {self.camera.upper()}")
        print("Settings:")
        print(f"  - Padding: {self.padding}px around YOLO bbox")
        print(f"  - Polygon mask: {'ON' if self.use_mask else 'OFF'}")
        print("")
        print("Watch for:")
        print("  - YELLOW box: YOLO bbox + padding")
        print("  - BLUE contour: LEFT edge (selected)")
        print("  - RED contour: RIGHT edge (selected)")
        print("  - GREEN line: connecting centroids (orientation)")
        print("  - Angle: should match visual tilt of port")
        print("")
        print("Side camera notes:")
        print("  - LEFT camera: port edges appear more vertical")
        print("  - RIGHT camera: opposite side view")
        print("  - Detected angle may need sign correction for wrist rotation")
        print("")
        print("Adjust padding:")
        print("  - --padding 5  = tighter (less margin)")
        print("  - --padding 20 = looser (more margin)")
        print("  - --no-mask    = disable polygon masking")
        print("="*60 + "\n")

        print("Press 'q' to quit\n")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            with self.detection_lock:
                det = self.latest_detection
            with self.image_lock:
                img = self.latest_image

            if det is None or img is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera...",
                           (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Edge Detection Verify", blank)
            elif not det.get("detected", False):
                blank = img.copy()
                cv2.putText(blank, "No port detected",
                           (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Edge Detection Verify", blank)
            elif det.get("port_type", "") != self.target_class:
                blank = img.copy()
                cv2.putText(blank, f"Wrong port: {det.get('port_type')}",
                           (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.imshow("Edge Detection Verify", blank)
            else:
                angle, left_pt, right_pt, crop_box, debug = self.detect_edges(img, det)
                cv2.imshow("Edge Detection Verify", debug)

                if angle is not None:
                    print(f"\rEdge angle: {angle:+6.1f}°  |  L: {left_pt}  R: {right_pt}    ", end="", flush=True)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break

        print("\n")
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Verify edge detection")
    parser.add_argument("--port", choices=["sfp", "sc"], default="sfp")
    parser.add_argument("--camera", choices=["center", "left", "right"], default="center",
                       help="Camera to use for edge detection")
    parser.add_argument("--padding", type=int, default=10,
                       help="Pixels to add around YOLO bbox. Default 10")
    parser.add_argument("--no-mask", action="store_true",
                       help="Disable polygon masking (use raw crop)")
    args = parser.parse_args()

    rclpy.init()
    node = VerifyEdgeDetection(
        port_type=args.port,
        camera=args.camera,
        padding=args.padding,
        use_mask=not args.no_mask
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