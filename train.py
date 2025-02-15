import cv2
import mediapipe as mp
import csv
import os

# List your videos here
video_paths = ["videos/fighting.mp4"]

# Use a constant CSV filename to enable appending data
output_csv = "pose_training_data.csv"

# Check if the CSV file already exists
file_exists = os.path.isfile(output_csv)

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Open file in append mode
with open(output_csv, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header only if file doesn't exist yet
    if not file_exists:
        header = []
        for i in range(33):
            header += [f"x{i}", f"y{i}", f"z{i}", f"vis{i}"]
        header.append("label")
        writer.writerow(header)
    
    # Process each video
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for MediaPipe
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                features = []
                for lm in result.pose_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
                # Label based on video file name (adjust as needed)
                if "fighting" in video_path:
                    label = "fighting"
                # elif "walking1" in video_path:
                #     label = "walking"
                else:
                    label = "Running"
                writer.writerow(features + [label])
        cap.release()

print("Data extraction complete! Data appended to", output_csv)
