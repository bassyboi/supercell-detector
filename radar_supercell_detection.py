import os
import cv2
import numpy as np
from PIL import Image
import torch
import datetime

# --- Imports for tkinter ---
import tkinter as tk
from tkinter import filedialog

def select_model_path() -> str:
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

def select_local_image() -> str:
    """
    Opens a Tkinter file dialog to let the user select an image file.
    Returns the file path or an empty string if none selected.
    """
    print("Opening file dialog to select an image...")
    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title='Select Local Image',
        filetypes=[
            ('Image files', '*.jpg *.jpeg *.png *.bmp'),
            ('All Files', '*.*')
        ]
    )
    root.destroy()

    if image_path:
        print(f"Selected image path: {image_path}")
    else:
        print("No image was selected.")
    return image_path

def load_image_as_cv2(image_path: str) -> np.ndarray:
    """
    Loads an image from disk and converts it to an OpenCV-compatible numpy array (BGR).
    Returns None if load fails.
    """
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return None

    try:
        pil_image = Image.open(image_path).convert('RGB')
        image_np = np.array(pil_image)  # RGB
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def detect_objects(image_bgr: np.ndarray, model, conf_threshold: float = 0.25):
    """
    Runs the YOLO model on an image to locate targets.
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
                detections.append((box, conf, int(cls_id)))
        return detections
    except Exception as e:
        print(f"Error during detection: {e}")
        return []

def draw_detections(image_bgr: np.ndarray, detections, class_names: dict):
    """
    Draw bounding boxes for each detection on the image.
    Returns the annotated image. If an error occurs, returns the original image.
    """
    if image_bgr is None or not detections:
        return image_bgr

    try:
        for (box, conf, cls_id) in detections:
            x1, y1, x2, y2 = map(int, box)
            label = class_names.get(cls_id, "Unknown")

            # Color-coding example
            # Adjust as desired for your 4 classes:
            #   0: "echo hook"
            #   1: "shower"
            #   2: "storm cell"
            #   3: "supercell"
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

def run_local_detection():
    """
    1. Prompts the user to pick a YOLO model (.pt file)
    2. Prompts the user to pick a local image
    3. Loads and runs detection
    4. Draws bounding boxes
    5. Saves the annotated image
    """
    # Ask user to select a YOLO .pt model
    model_path = select_model_path()
    if not model_path:
        print("No model selected. Exiting.")
        return
    
    # Example class names for your 4 classes
    class_names = {
        0: "echo hook",
        1: "shower",
        2: "storm cell",
        3: "supercell"
    }

    conf_threshold = 0.30

    # Load YOLO model
    print("Loading YOLO model...")
    try:
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=model_path,
            force_reload=True  # Force refresh in case of caching issues
        )
        print("YOLO model loaded.")
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return

    # Ask user to select a local image
    image_path = select_local_image()
    if not image_path:
        print("No image selected. Exiting.")
        return
    
    # Prepare output directory
    output_dir = "local_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image as OpenCV
    image_bgr = load_image_as_cv2(image_path)
    if image_bgr is None:
        print("Could not load image data. Exiting.")
        return

    # Run detection
    detections = detect_objects(image_bgr, model, conf_threshold)
    
    # Draw bounding boxes
    annotated_image = draw_detections(image_bgr, detections, class_names)
    
    # Save result
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"annotated_{timestamp_str}.png"
    output_path = os.path.join(output_dir, output_filename)
    try:
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")
    except Exception as e:
        print(f"Error saving annotated image: {e}")


if __name__ == "__main__":
    run_local_detection()
