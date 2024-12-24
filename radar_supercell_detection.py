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

# --- NEW IMPORTS FOR TKINTER FILE DIALOG ---
import tkinter as tk
from tkinter import filedialog

def scrape_bom_radar_image(url: str) -> str:
    """
    Scrapes the BOM radar loop page for the most recent radar image URL.
    Returns the absolute URL to the image, or None if not found.
    Includes basic error handling.
    """
    print(f"Scraping BOM radar loop page: {url}")
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Request error while accessing {url}: {e}")
        return None
    
    if response.status_code != 200:
        print(f"Failed to access page. Status code: {response.status_code}")
        return None

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Regex aims to find something like /radar/IDR193.T.202308100927.png
    pattern = re.compile(r'(/radar/IDR193[^"]+\.png)')
    
    # We'll look for script tags containing references to "IDR193"
    script_tags = soup.find_all("script")
    found_images = []
    for tag in script_tags:
        if tag.string and "IDR193" in tag.string:
            matches = pattern.findall(tag.string)
            if matches:
                found_images.extend(matches)

    if not found_images:
        print("No IDR193 image paths found in the page scripts.")
        return None
    
    # The frames are often in chronological order; the last entry is usually the most recent
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


def download_image(save_path: str, image_url: str) -> bool:
    """
    Downloads an image from the provided URL to save_path.
    Returns True if successful, False otherwise.
    Includes basic error handling.
    """
    if not image_url:
        print("No image URL provided to download.")
        return False

    print(f"Downloading radar image from {image_url}...")
    try:
        response = requests.get(image_url, stream=True, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Request error while downloading image: {e}")
        return False

    if response.status_code == 200:
        try:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Radar image saved to {save_path}.")
            return True
        except IOError as e:
            print(f"Error writing file {save_path}: {e}")
            return False
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        return False


def load_image_as_cv2(image_path: str) -> np.ndarray:
    """
    Loads an image from disk and converts it to an OpenCV-compatible numpy array (BGR).
    If the image cannot be loaded, returns None.
    """
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return None

    try:
        pil_image = Image.open(image_path)
        image_np = np.array(pil_image)  # RGB
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def detect_supercell(image_bgr: np.ndarray, model, conf_threshold: float = 0.25):
    """
    Runs the YOLO model on a radar image to locate targets (echo hook, shower, storm cell, supercell).
    Returns a list of detections or an empty list if none are found.
    """
    if image_bgr is None:
        print("Empty image data; skipping detection.")
        return []

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        # YOLO inference
        results = model(image_rgb, size=640)
        detections = []
        for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
            if conf >= conf_threshold:
                detections.append((box, conf, cls_id))
        return detections
    except Exception as e:
        print(f"Error during detection: {e}")
        return []


def draw_detections(image_bgr: np.ndarray, detections, class_names: dict):
    """
    Draws bounding boxes for each detection on the radar image.
    Returns the annotated image. If an error occurs, returns original image_bgr.
    """
    if image_bgr is None or not detections:
        return image_bgr

    try:
        for (box, conf, cls_id) in detections:
            x1, y1, x2, y2 = map(int, box)
            label = class_names.get(int(cls_id), "Unknown")
            
            # Assign a color for each label (you can customize as needed)
            if label == "echo hook":
                color = (255, 255, 0)   # cyan
            elif label == "shower":
                color = (0, 255, 255)  # yellow
            elif label == "storm cell":
                color = (255, 0, 0)    # blue
            elif label == "supercell":
                color = (0, 0, 255)    # red
            else:
                color = (255, 255, 255)  # white (unknown)

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
    except Exception as e:
        print(f"Error drawing detections: {e}")
        return image_bgr


def get_model_path_via_tkinter() -> str:
    """
    Opens a Tkinter file dialog to let the user select a .pt model file.
    Returns the file path or an empty string if none selected.
    """
    print("Opening file dialog to select the PyTorch model (.pt file)...")
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    model_path = filedialog.askopenfilename(
        title='Select PyTorch Model',
        filetypes=[('PyTorch Model', '*.pt'), ('All Files', '*.*')]
    )
    root.destroy()

    if model_path:
        print(f"Selected model path: {model_path}")
    else:
        print("No model file was selected.")

    return model_path


def main_loop():
    """
    Continuously fetches the latest BOM radar image from IDR193.loop.shtml, 
    runs detection on the classes:
        - echo hook
        - shower
        - storm cell
        - supercell
    and saves annotated images.
    Incorporates error handling and uses tkinter to select a .pt model.
    """
    # URL for the IDR193 radar loop page
    bom_loop_url = "http://www.bom.gov.au/products/IDR193.loop.shtml"
    
    # Output directory
    output_dir = "radar_images"
    os.makedirs(output_dir, exist_ok=True)

    # Ask the user to select a YOLO PyTorch model file
    model_path = get_model_path_via_tkinter()
    if not model_path:
        print("No model path provided. Exiting.")
        return
    
    # Define your class names for the 4 classes
    class_names = {
        0: "echo hook",
        1: "shower",
        2: "storm cell",
        3: "supercell",
    }
    conf_threshold = 0.30

    # Load YOLO model once
    print("Loading YOLO model...")
    try:
        # Using YOLOv5 from ultralytics
        model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path=model_path, 
            force_reload=True
        )
        print("YOLO model loaded.")
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return

    # Scan interval (e.g., 5 minutes)
    scan_interval_seconds = 300

    while True:
        try:
            # 1. Scrape the BOM loop page for the latest image link
            latest_image_url = scrape_bom_radar_image(bom_loop_url)
            if not latest_image_url:
                print("No valid image URL found; skipping this cycle.")
            else:
                # 2. Download the image
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                downloaded_image_path = os.path.join(output_dir, f"radar_{timestamp_str}.png")
                annotated_image_path = os.path.join(output_dir, f"annotated_radar_{timestamp_str}.png")
                
                success = download_image(downloaded_image_path, latest_image_url)
                if success:
                    # 3. Detect classes in the image
                    image_bgr = load_image_as_cv2(downloaded_image_path)
                    detections = detect_supercell(image_bgr, model, conf_threshold)

                    # 4. Draw bounding boxes
                    image_with_boxes = draw_detections(image_bgr, detections, class_names)
                    if image_with_boxes is not None:
                        try:
                            cv2.imwrite(annotated_image_path, image_with_boxes)
                            print(f"Annotated image saved to {annotated_image_path}")
                        except Exception as e:
                            print(f"Error saving annotated image: {e}")
        
        except Exception as e:
            print(f"Unexpected error during cycle: {e}")

        print(f"Waiting {scan_interval_seconds} seconds until next scan...\n")
        time.sleep(scan_interval_seconds)


if __name__ == "__main__":
    main_loop()
