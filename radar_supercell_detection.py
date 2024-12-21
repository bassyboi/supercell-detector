#!/usr/bin/env python3

import os
import time
import requests
import cv2
import numpy as np
from PIL import Image
import torch
import datetime

def download_bom_radar_image(save_path: str, radar_url: str):
    """
    Downloads a single radar image from BOM's web server.
    :param save_path: Where to save the downloaded image.
    :param radar_url: URL to the BOM radar image.
    """
    print(f"Downloading radar image from {radar_url}...")
    response = requests.get(radar_url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Radar image saved to {save_path}.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def load_image_as_cv2(image_path: str) -> np.ndarray:
    """
    Loads an image from disk and converts it to an OpenCV-compatible numpy array (BGR).
    :param image_path: Path to the image on disk.
    :return: OpenCV-compatible image (numpy array in BGR format).
    """
    pil_image = Image.open(image_path)
    # Convert PIL image to a numpy array (RGB)
    image_np = np.array(pil_image)
    # Convert RGB to BGR
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr

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

def draw_detections(image_bgr: np.ndarray, detections, class_names: dict):
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
        color = (0, 0, 255) if label == "supercell" else (255, 0, 0)
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
    # Replace with an actual BOM radar image link, ensuring you have correct permissions.
    radar_image_url = "https://example.com/path/to/bom_radar_image.png"
    
    # This directory will store the raw downloaded images and annotated images
    output_dir = "radar_images"
    os.makedirs(output_dir, exist_ok=True)

    # YOLO model path (must be a model trained/fine-tuned for your radar data).
    # This example uses a YOLOv5 PyTorch model from Ultralytics.
    model_path = "yolo/best.pt"

    # Class names for your model (assuming 0 is "supercell", 1 is "storm", etc.)
    class_names = {
        0: "supercell",
        1: "storm",
        2: "rainband",
        # ...
    }

    # Confidence threshold
    conf_threshold = 0.30

    # ------------------------------------
    # 2. LOAD MODEL (only once!)
    # ------------------------------------
    print("Loading YOLO model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    print("YOLO model loaded.")

    # ------------------------------------
    # 3. CONTINUOUS LOOP
    # ------------------------------------
    scan_interval_seconds = 300  # e.g., 5 minutes

    while True:
        try:
            # Timestamp to differentiate saved images
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Raw radar image path
            downloaded_image_path = os.path.join(output_dir, f"radar_{timestamp_str}.png")
            
            # Annotated image path
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
            print(f"Annotated image saved to {annotated_image_path}")

            # If desired, you can show the result in a window
            # (Comment out if running on a headless server without a GUI)
            # cv2.imshow("Supercell Detection", image_with_boxes)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
        
        except Exception as e:
            print(f"An error occurred: {e}")

        # -----------------------
        # Wait for the next scan
        # -----------------------
        print(f"Waiting {scan_interval_seconds} seconds until next scan...\n")
        time.sleep(scan_interval_seconds)

if __name__ == "__main__":
    main_loop()
