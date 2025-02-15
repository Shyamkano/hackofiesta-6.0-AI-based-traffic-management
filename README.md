# hackofiesta-6.0-AI-based-traffic-management
# SOS & Surveillance Hackathon Prototype

## Overview

This project is a multi-modal safety and surveillance solution developed during the hackathon. Our prototype integrates four key components:
  
- **Flutter SOS App:**  
  A mobile application that sends emergency SOS alerts (with geolocation) to pre-designated contacts.

- **Vehicle Detection Model:**  
  A computer vision module (using YOLO) that monitors live video feeds for vehicle detection.

- **Human Behavior Detection Model:**  
  A deep learning module (using MediaPipe and PyTorch) that analyzes human poses to detect abnormal behaviors (e.g., running or fighting).

- **Voice Distress Detection:**  
  A voice distress module that listens via a microphone for distress signals—particularly aimed at enhancing women's safety. When specific distress keywords or unusual audio patterns are detected, an alert is triggered.

*Note:* Although we planned a real-time dashboard for analytics, that component was not completed during the hackathon.

## Key Features

- **Emergency SOS Alerts:**  
  The Flutter SOS app quickly sends alerts with location data during emergencies.

- **Real-Time Vehicle Monitoring:**  
  The vehicle detection module processes video feeds to identify vehicles using a YOLO-based object detection algorithm.

- **Abnormal Human Behavior Detection:**  
  The human behavior detection module extracts pose landmarks using MediaPipe and classifies behaviors using a PyTorch model.

- **Voice Distress Detection:**  
  Continuously monitors ambient audio via a microphone for distress signals or keywords (e.g., “help”, “save me”) to provide an additional safety layer for women.

- **Automated Analytics & Alerting:**  
  The system logs detected incidents and sends email alerts when thresholds are met. It also generates basic analytics reports.

## Technology Stack

- **Mobile Application:** Flutter, Dart  
- **Backend & Models:**  
  Python, OpenCV, MediaPipe, PyTorch, YOLO (Ultralytics)  
- **Voice Distress Module:**  
  Python SpeechRecognition (or similar library), Google Speech API (or offline recognition libraries)  
- **Data Handling & Analytics:** Pandas, NumPy, scikit-learn, Matplotlib, Seaborn  
- **Communication:** SMTP for email alerts

## System Architecture

1. **Flutter SOS App:**  
   - Developed for Android and iOS.
   - Captures and transmits geolocation data along with SOS signals.

2. **Vehicle Detection Module:**  
   - Uses YOLO to detect vehicles in real-time video feeds.

3. **Human Behavior Detection Module:**  
   - Uses MediaPipe to extract pose landmarks from video frames.
   - Processes these landmarks with a deep learning classifier to determine if behavior is normal or abnormal.

4. **Voice Distress Module:**  
   - Continuously monitors audio through a connected microphone.
   - Uses speech recognition techniques to detect distress signals and triggers an alert when keywords or unusual patterns are found.

5. **Analytics & Alerts:**  
   - Logs incidents with timestamps and locations.
   - Sends email alerts when detection confidence exceeds set thresholds.
   - Generates analytics reports (e.g., hourly incident charts, heatmaps).

