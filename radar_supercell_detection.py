#!/usr/bin/env python3

import os
import time
import datetime
import logging
import requests
import cv2
import numpy as np
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def download_bom_radar_image(save_path: str, radar_url: str, timeout: int = 10) -> None:
    """
    Downloads a single radar image from BOM's web server.
    :param save_path: Where to save the downloaded image.
    :param radar_url: URL to the BOM radar image.
    :param timeout: Request timeout in seconds.
    """
    logging.info(f"Downloading radar image from {radar_url}...")
    response = requests.get(radar_url, stream=True, timeout=timeout)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Radar image saved to {save_path}.")
    else:
        logging.error(f"Failed to download. Status code: {response.status_code}")


def load_image_as_cv2(image_path: str) -> np.ndarray:
    """
    Loads an image from disk and converts it to an OpenCV-compatible numpy array (BGR).
    :param image_path: Path to the image on disk.
    :return: OpenCV-compatible image (numpy array in BGR format).
    """
    with Image.open(image_path) as pil_image:
        # Convert PIL image to a numpy array (RGB)
        image_np = np.array(pil_image)
    # Convert RGB to BGR
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


def detect_supercell(image_bgr: np.ndarray, model, conf_threshold: float = 0.25):
    """
    Runs the YOLO object detection model on a radar image to locate supercells.
    :param image_bgr: The image in BGR format (OpenCV).
    :param model: A YOLO model (e.g., YOLOv5 or YOLOv8).
    :param conf_threshold: Minimum confidence threshold to keep detections.
    :return: detection results (bounding boxes, confidences, class IDs, etc.)
    """
    # Convert BGR to RGB since YOLO models typically expect RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Run inference using the YOLO model
    results = model(image_rgb, size=640)

    # Filter results by confidence threshold
    filtered = []
    for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
        if conf >= conf_threshold:
            filtered.append((box, conf, cls_id))

    return filtered


def draw_detections(image_bgr: np.ndarray, detections, class_names: dict) -> np.ndarray:
    """
    Draw bounding boxes for each detected supercell (or other classes) on the image.
    :param image_bgr: The image (BGR).
    :param detections: List of detections (box, confidence, class).
    :param class_names: Dictionary mapping class IDs to class labels.
    :return: image with drawn boxes.
    """
    for (box, conf, cls_id) in detections:
        x1, y1, x2, y2 = map(int, box)
        label = class_names.get(int(cls_id), "Unknown")
        # Color-code bounding boxes based on label
        if label == "supercell":
            color = (0, 0, 255)  # Red
        elif label == "storm":
            color = (0, 255, 255)  # Yellow
        else:
            color = (255, 0, 0)    # Blue
        
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_bgr,
            f"{label} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    return image_bgr


def main_loop():
    """
    Continuously downloads the latest BOM radar image, runs supercell detection,
    and saves annotated images at a fixed time interval.
    """

    # ---------------------------
    # 1. SETUP - Modify as needed
    # ---------------------------
    radar_image_url = "https://example.com/path/to/bom_radar_image.png"
    output_dir = "radar_images"
    os.makedirs(output_dir, exist_ok=True)

    model_path = "yolo/best.pt"
    class_names = {
        0: "supercell",
        1: "storm",
        2: "rainband",
    }

    conf_threshold = 0.30
    scan_interval_seconds = 300  # e.g., 5 minutes

    # ------------------------------------
    # 2. LOAD MODEL (only once!)
    # ------------------------------------
    logging.info("Loading YOLO model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    logging.info("YOLO model loaded.")

    # ------------------------------------
    # 3. CONTINUOUS LOOP
    # ------------------------------------
    while True:
        try:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            downloaded_image_path = os.path.join(output_dir, f"radar_{timestamp_str}.png")
            annotated_image_path = os.path.join(output_dir, f"annotated_radar_{timestamp_str}.png")

            # -----------------------
            # Download the radar image
            # -----------------------
            download_bom_radar_image(downloaded_image_path, radar_image_url)

            # -----------------------
            # Load and run detection
            # -----------------------
            image_bgr = load_image_as_cv2(downloaded_image_path)
            detections = detect_supercell(image_bgr, model, conf_threshold)

            # -----------------------
            # Draw detections
            # -----------------------
            image_with_boxes = draw_detections(image_bgr, detections, class_names)

            # -----------------------
            # Save the annotated image
            # -----------------------
            cv2.imwrite(annotated_image_path, image_with_boxes)
            logging.info(f"Annotated image saved to {annotated_image_path}")

        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

        logging.info(f"Waiting {scan_interval_seconds} seconds until next scan...\n")
        time.sleep(scan_interval_seconds)


if __name__ == "__main__":
    main_loop()
