"""
PROJECT: Object Detection with Voice Feedback using YOLOv8n

DESCRIPTION:
This project performs real-time object detection using a webcam and 
announces detected objects using offline text-to-speech (pyttsx3). 
It uses the ultralytics YOLOv8n model for fast inference.

FEATURES:
- Real-time webcam frame capture
- YOLOv8n object detection
- Bounding box visualization
- Offline voice output for newly detected objects
- Cooldown system to avoid repeating detection audio
- Adjustable confidence and device configuration

FILES:
- main.py  → Main program
- README.md → Documentation
- requirements.txt → Python dependencies

USAGE:
1. Install dependencies:
   pip install -r requirements.txt

2. Run the program:
   python main.py

AUTHOR:
(Write your name)
"""
