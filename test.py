import cv2
import time
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import pickle
import json
from pathlib import Path
import logging
from collections import deque
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from threading import Thread
import queue
import os

# -------------------------
# Advanced Behavior Classifier
# (This definition must match the one used during training.)
# -------------------------
class AdvancedBehaviorClassifier(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_classes=2):
        """
        input_size: 33 landmarks x 4 = 132 features.
        num_classes: number of behavior classes (dynamically set based on training).
        """
        super(AdvancedBehaviorClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# -------------------------
# Alert System
# -------------------------
class AlertSystem:
    def __init__(self, config):
        self.config = config
        self.alert_queue = queue.Queue()
        self.last_alert_time = {}  # For cooldown tracking
        self.alert_thread = Thread(target=self._process_alerts, daemon=True)
        self.alert_thread.start()
        
    def _process_alerts(self):
        while True:
            alert = self.alert_queue.get()
            self._send_email_alert(alert)
            self._log_alert(alert)
            self.alert_queue.task_done()
    
    def _send_email_alert(self, alert_data):
        try:
            msg = MIMEMultipart()
            email_conf = self.config["email"]
            msg['From'] = email_conf['sender']
            msg['To'] = email_conf['recipient']
            msg['Subject'] = f"Security Alert: {alert_data['type']}"
            body = f"""
Security Alert Details:
Type: {alert_data['type']}
Location: {alert_data['location']}
Time: {alert_data['timestamp']}
Confidence: {alert_data['confidence']}%
Description: {alert_data['description']}
            """
            msg.attach(MIMEText(body, 'plain'))
            with smtplib.SMTP(email_conf['smtp_server'], 587) as server:
                server.starttls()
                server.login(email_conf['username'], email_conf['password'])
                server.send_message(msg)
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
    
    def _log_alert(self, alert_data):
        try:
            with open('security_alerts.log', 'a') as f:
                json.dump(alert_data, f)
                f.write('\n')
        except Exception as e:
            logging.error(f"Failed to log alert: {e}")
    
    def send_alert(self, alert_type, location, confidence, description):
        cooldown = self.config.get("alert_settings", {}).get("alert_cooldown", 60)
        now = time.time()
        if alert_type in self.last_alert_time and (now - self.last_alert_time[alert_type]) < cooldown:
            return
        self.last_alert_time[alert_type] = now
        alert_data = {
            'type': alert_type,
            'location': location,
            'timestamp': datetime.datetime.now().isoformat(),
            'confidence': confidence,
            'description': description
        }
        self.alert_queue.put(alert_data)

# -------------------------
# Analytics
# -------------------------
class Analytics:
    def __init__(self, save_dir='analytics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.incidents_df = pd.DataFrame(columns=['timestamp', 'incident_type', 'location', 'confidence'])
        self.behavior_stats = {}
        
    def update_stats(self, incident_type, location, confidence):
        new_row = {
            'timestamp': datetime.datetime.now(),
            'incident_type': incident_type,
            'location': location,
            'confidence': confidence
        }
        self.incidents_df = pd.concat([self.incidents_df, pd.DataFrame([new_row])], ignore_index=True)
        self.behavior_stats[incident_type] = self.behavior_stats.get(incident_type, 0) + 1
        
    def generate_reports(self):
        if self.incidents_df.empty:
            logging.info("No incidents to report.")
            return "No incidents recorded."
        self.incidents_df['timestamp'] = pd.to_datetime(self.incidents_df['timestamp'])
        hourly_incidents = self.incidents_df.set_index('timestamp').resample('h').size()
        plt.figure(figsize=(12, 6))
        hourly_incidents.plot(kind='bar')
        plt.title("Incidents by Hour")
        plt.savefig(self.save_dir / "hourly_incidents.png")
        plt.close()
        
        plt.figure(figsize=(10, 8))
        incident_locations = self.incidents_df['location'].value_counts()
        sns.heatmap(incident_locations.values.reshape(-1, 1), yticklabels=incident_locations.index, annot=True, cmap="YlOrRd")
        plt.title("Incident Location Heatmap")
        plt.savefig(self.save_dir / "location_heatmap.png")
        plt.close()
        
        plt.figure(figsize=(10, 10))
        plt.pie(list(self.behavior_stats.values()), labels=list(self.behavior_stats.keys()), autopct='%1.1f%%')
        plt.title("Behavior Distribution")
        plt.savefig(self.save_dir / "behavior_distribution.png")
        plt.close()
        
        try:
            most_common = self.incidents_df['incident_type'].mode()[0]
        except Exception:
            most_common = "N/A"
        report = f"""
Security Analysis Report
Generated: {datetime.datetime.now()}

Total Incidents: {len(self.incidents_df)}
Most Common Incident: {most_common}
High-Risk Locations: {', '.join(self.incidents_df['location'].unique())}

Behavior Statistics:
{pd.Series(self.behavior_stats).to_string()}
"""
        with open(self.save_dir / "analysis_report.txt", "w") as f:
            f.write(report)
        return report

# -------------------------
# Temporal Aggregator (for smoothing predictions)
# -------------------------
class TemporalAggregator:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.last_logged = None

    def add_prediction(self, pred):
        self.predictions.append(pred)
        if len(self.predictions) == self.window_size:
            agg_pred = np.bincount(list(self.predictions)).argmax()
            if self.last_logged is None or agg_pred != self.last_logged:
                self.last_logged = agg_pred
                return agg_pred
        return None

# -------------------------
# Enhanced Behavior Detection System
# -------------------------
class EnhancedBehaviorDetectionSystem:
    def __init__(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Load object detection settings.
        obj_conf = self.config.get("object_detection", {})
        yolo_model_path = obj_conf.get("yolo_model_path", "yolov8n-pose.pt")
        self.conf_thresh = obj_conf.get("confidence_threshold", 0.25)
        self.input_resolution = tuple(obj_conf.get("input_resolution", [640, 384]))
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize MediaPipe Pose.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, 
                                      min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        # Load behavior classifier settings.
        behav_conf = self.config.get("behavior_detection", {})
        classifier_path = behav_conf.get("classifier_model_path", "")
        scaler_path = behav_conf.get("scaler_path", "")
        label_encoder_path = behav_conf.get("label_encoder_path", "")
        
        # Load label encoder and determine the number of classes dynamically.
        self.label_encoder = None
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            num_classes = len(self.label_encoder.classes_)
            self.behavior_labels = { idx: label for idx, label in enumerate(self.label_encoder.classes_) }
            logging.info(f"Loaded label encoder with classes: {self.label_encoder.classes_}")
        else:
            logging.warning("Label encoder not found. Defaulting to 2 classes: Normal and Running.")
            num_classes = 2
            self.behavior_labels = {0: "Normal", 1: "Running"}
        
        # Initialize classifier with the correct number of classes.
        self.behavior_classifier = AdvancedBehaviorClassifier(input_size=132, hidden_size=128, num_classes=num_classes)
        self._load_or_initialize_classifier(classifier_path, scaler_path)
        
        self.alert_system = AlertSystem(self.config.get("email", {}))
        analytics_dir = self.config.get("analytics", {}).get("save_directory", "analytics")
        self.analytics = Analytics(save_dir=analytics_dir)
        self.aggregator = TemporalAggregator(window_size=self.config.get("temporal_aggregator", {}).get("window_size", 10))
        
        self.prev_frame_time = 0

    def _load_or_initialize_classifier(self, classifier_path, scaler_path):
        if os.path.exists(classifier_path):
            try:
                state_dict = torch.load(classifier_path, map_location=torch.device('cpu'))
                self.behavior_classifier.load_state_dict(state_dict)
                self.behavior_classifier.eval()
                logging.info("Loaded behavior classifier model.")
            except Exception as e:
                logging.error(f"Error loading classifier: {e}")
        else:
            logging.warning("Classifier model not found. Using untrained model.")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logging.info("Loaded feature scaler.")
            except Exception as e:
                logging.error(f"Error loading scaler: {e}")
        else:
            logging.warning("Feature scaler not found.")
            self.scaler = None

    def _detect_basic_behavior(self, pose_result):
        if not pose_result or not pose_result.pose_landmarks:
            return 0  # Default to "Normal"
        landmarks = pose_result.pose_landmarks.landmark
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
        features = np.array(features)
        if self.scaler:
            features = self.scaler.transform([features])
        else:
            features = features.reshape(1, -1)
        tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            output = self.behavior_classifier(tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]
            predicted = np.argmax(probs)
            # Only accept the prediction if its probability is above the threshold
            threshold = self.config.get("behavior_detection", {}).get("confidence_threshold", 0.6)
            if probs[predicted] < threshold:
                predicted = 0  # Default to "Normal" if not confident
        return predicted

    def _draw_detection(self, frame, box, behavior):
        disp_conf = self.config.get("display", {})
        thickness = disp_conf.get("box_thickness", 2)
        # Use green for Normal; red for any suspicious behavior.
        label = self.behavior_labels.get(behavior, "Unknown")
        color = (0, 255, 0) if label.lower() == "normal" else (0, 0, 255)
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        cv2.putText(frame, label, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, disp_conf.get("font_scale", 0.5), color, 2)

    def process_frame(self, frame):
        try:
            orig_h, orig_w = frame.shape[:2]
            resized_frame = cv2.resize(frame, self.input_resolution)
            results = self.yolo_model(resized_frame, conf=self.conf_thresh)
            person_boxes = []
            pose_results = []
            for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
                # Process only if detected object is a person (class 0)
                if int(results[0].boxes.cls[i]) == 0:
                    scale_x = orig_w / self.input_resolution[0]
                    scale_y = orig_h / self.input_resolution[1]
                    box = [box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y]
                    person_boxes.append(box)
                    x1, y1, x2, y2 = map(int, box)
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size == 0:
                        pose_results.append(None)
                        continue
                    pose_result = self.pose.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                    # (Optional) Draw pose landmarks on the person_img for debugging:
                    # if pose_result.pose_landmarks:
                    #     mp.solutions.drawing_utils.draw_landmarks(person_img, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    pose_results.append(pose_result)
            for i, box in enumerate(person_boxes):
                pred = self._detect_basic_behavior(pose_results[i])
                agg_pred = self.aggregator.add_prediction(pred)
                behavior_index = agg_pred if agg_pred is not None else pred
                behavior_label = self.behavior_labels.get(behavior_index, "Unknown")
                # Trigger an alert if behavior is not "Normal" (and if the aggregator has enough samples)
                if behavior_label.lower() != "normal" and agg_pred is not None:
                    alert_msg = f"{behavior_label} detected in Zone_{i}"
                    self.alert_system.send_alert(behavior_label, f"Zone_{i}", 100.0, alert_msg)
                self.analytics.update_stats(behavior_label, f"Zone_{i}", 100.0)
                self._draw_detection(frame, box, behavior_index)
            if self.config.get("display", {}).get("show_fps", False):
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - self.prev_frame_time + 1e-6)
                self.prev_frame_time = new_frame_time
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return frame
        except Exception as e:
            self.logger.error(f"Error in frame processing: {e}")
            return frame

    def run(self, video_source, frame_skip=1):
        cap = cv2.VideoCapture(video_source)
        # Get the resize factor from the config. If not provided, default is 1.0 (no resizing)
        resize_factor = self.config.get("display", {}).get("resize_factor", 1.0)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.behavior_classifier.to(device)
        
        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every nth frame to reduce computation
                if frame_id % frame_skip == 0:
                    processed_frame = self.process_frame(frame)
                else:
                    processed_frame = frame  # show unprocessed frame
                frame_id += 1

                # Resize the frame for display using the configured resize_factor.
                disp_frame = cv2.resize(processed_frame, None, fx=resize_factor, fy=resize_factor)
                cv2.imshow("Enhanced Security System", disp_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            self.logger.error(f"Error during video capture: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            report = self.analytics.generate_reports()
            logging.info("Analytics Report Generated:")
            logging.info(report)


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    config_path = "security_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    print("Loaded config keys:", config.keys())
    detection_system = EnhancedBehaviorDetectionSystem(config)
    detection_system.run("videos/walking1.mp4")  # Replace with your video file path
