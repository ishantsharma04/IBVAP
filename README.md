# IBVAP — Intelligent Border Video Analytics Platform

SIH prototype for AI-based border surveillance using existing CCTV infrastructure.

## Included modules
- YOLO object detection
- ByteTrack object tracking
- Person / vehicle / animal detection
- Virtual-fence intrusion detection
- ANPR using Tesseract OCR
- Local entity registry for demo watchlist/authorized records
- Separate Add Person and Add Vehicle pages
- Event logging
- Offline-mode simulation
- CCTV webcam or uploaded video source

## Run on Windows
1. Open PowerShell in this folder.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Make sure Tesseract OCR is installed at:
   `C:\Program Files\Tesseract-OCR\tesseract.exe`
4. Start:
   `python app.py`
5. Open:
   `http://127.0.0.1:5000`

Use fictional/demo records for the SIH prototype. A registry match is a potential match for human verification; it should not by itself establish identity or criminal status.
