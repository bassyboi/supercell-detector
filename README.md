
# BOM Radar Supercell Detection

**A Python-based tool for detecting supercells using radar images from the Australian Bureau of Meteorology (BOM).**

---

## **Overview**

- Continuously downloads radar images from BOM.
- Detects supercells using a YOLO-based model.
- Annotates images and saves them for analysis.

---

## **Features**

1. Automated radar image downloads.
2. YOLO-based supercell detection.
3. Easy-to-review annotated image pipeline.

---

## **Prerequisites**

- **Python 3.7+** (ideally 3.8 or 3.9)
- **Git** for cloning the repository.
- **pip** for dependency management.

---

## **Installation Steps**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<yourusername>/bom-supercell-detection.git
   cd bom-supercell-detection
   ```

2. **Create Necessary Files:**
   - `install.sh`
   - `radar_supercell_detection.py`

3. **Run `install.sh`:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

4. **Activate Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

---

## **Usage**

1. **Run the Detection Script:**
   ```bash
   python radar_supercell_detection.py
   ```

2. **Deactivate Environment (Optional):**
   ```bash
   deactivate
   ```

---

## **Tips & Customizations**

- **Run in Background:** Use `nohup` or `cron` (Linux/Mac) or Task Scheduler (Windows).
- **Model Updates:** Replace YOLO weights for better accuracy.

---

## **Contributing**

1. Fork the repository.
2. Create a new branch.
3. Commit changes and open a pull request.

---

## **License**

This project is licensed under the **MIT License**.

**Happy Detecting!**
"""

