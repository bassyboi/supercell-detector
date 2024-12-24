#!/usr/bin/env python3

import os
import re
import time
import requests
import cv2
import numpy as np
from PIL import Image
import torch
import datetime
from bs4 import BeautifulSoup

def scrape_bom_radar_image(url: str) -> str:
    """
    Scrapes the BOM radar loop page for the most recent radar image URL.
    Returns the absolute URL to the image, or None if not found.
    """
    print(f"Scraping BOM radar loop page: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to access page. Status code: {response.status_code}")
        return None

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Example: search for lines like /radar/IDR193.T.202308100927.png
    # or script blocks containing a var imageList = [...]
    # We'll do a simple approach: find all <script> blocks with "IDR193"
    script_tags = soup.find_all("script")

    # This regex aims to find something like /radar/IDR193.T.202308100927.png
    pattern = re.compile(r'(/radar/IDR193[^"]+\.png)')

    found_images = []
    for tag in script_tags:
        if tag.string and "IDR193" in tag.string:
            matches = pattern.findall(tag.string)
            if matches:
                found_images.extend(matches)

    if not found_images:
        print("No IDR193 image paths found in the page scripts.")
        return None
    
    # The frames are often in chronological order. 
    # The last entry is typically the most recent.
    latest_path = found_images[-1]

    # The BOM site usually uses relative paths like /radar/IDR193.T.yyyymmddHHMM.png
    # Prepend domain if not already present
    if latest_path.startswith("/"):
        latest_url = f"http://www.bom.gov.au{latest_path}"
    else:
        # If it's absolute or something else
        latest_url = latest_path
    
    print(f"Found latest radar image: {latest_url}")
    return latest_url

def download_image(save_path: str, image_url: str):
    """
    Downloads an image from the provided URL to save_path.
    """
    print(f"Downloading radar image from {image_url}...")
    response = requests.get(image_url, stream=True)
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
    """
    pil_image = Image.open(image_path)
    image_np = np.array(pil_image)  # RGB
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr

def detect_supercell(image_bgr: np.ndarray, model, conf_threshold: float = 0.25):
    """
    Runs the YOLO model on a radar image to locate supercells (or any target classes).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # YOLO inference
    results = model(image_rgb, size=640)

    # Filter by confidence
    filtered = []
    for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
        if conf >= conf_threshold:
            filtered.append((box, conf, cls_id))
    return filtered

def draw_detections(image_bgr: np.ndarray, detections, class_names: dict):
    """
    Draws bounding boxes for each detection on the radar image.
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
    Continuously fetches the latest BOM radar image from IDR193.loop.shtml, 
    runs supercell detection, and saves annotated images.
    """
    # URL for the IDR193 radar loop page
    bom_loop_url = "http://www.bom.gov.au/products/IDR193.loop.shtml"
    
    # Output directory
    output_dir = "radar_images"
    os.makedirs(output_dir, exist_ok=True)

    # YOLO model info
    model_path = "yolo/best.pt"  # replace with your model path
    class_names = {
        0: "supercell",
        1: "storm",
        2: "rainband",
    }
    conf_threshold = 0.30

    # Load YOLO model once
    print("Loading YOLO model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    print("YOLO model loaded.")

    # Scan interval (e.g., 5 minutes)
    scan_interval_seconds = 300

    while True:
        try:
            # 1. Scrape the BOM loop page for the latest image link
            latest_image_url = scrape_bom_radar_image(bom_loop_url)
            if not latest_image_url:
                print("No valid image URL found; skipping this cycle.")
            else:
                # 2. Download it
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                downloaded_image_path = os.path.join(output_dir, f"radar_{timestamp_str}.png")
                annotated_image_path = os.path.join(output_dir, f"annotated_radar_{timestamp_str}.png")
                
                download_image(downloaded_image_path, latest_image_url)

                # 3. Detect supercells
                image_bgr = load_image_as_cv2(downloaded_image_path)
                detections = detect_supercell(image_bgr, model, conf_threshold)

                # 4. Draw bounding boxes
                image_with_boxes = draw_detections(image_bgr, detections, class_names)
                cv2.imwrite(annotated_image_path, image_with_boxes)
                print(f"Annotated image saved to {annotated_image_path}")
        
        except Exception as e:
            print(f"Error during cycle: {e}")

        print(f"Waiting {scan_interval_seconds} seconds until next scan...\n")
        time.sleep(scan_interval_seconds)

if __name__ == "__main__":
    main_loop()
