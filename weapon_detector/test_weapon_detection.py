import cv2
import time
import pygame
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO  # Make sure ultralytics YOLOv8 is installed

# Load YOLOv8 model
model = YOLO(r'Z:\Security_Surveillance_System\weapon_detector\best.pt')  # Replace with your actual model path

# Alarm setup
pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(r"Z:\Security_Surveillance_System\weapon_detector\src\alarm.mp3")

# Create snapshots directory if it doesn't exist
Path("snapshots").mkdir(exist_ok=True)

# Define threat categories
HIGH_THREATS = ['Gun', 'Rifle', 'Grenade', 'Pistol', 'Handgun']
LOW_THREATS = ['Knife']

# Initialize camera
cap = cv2.VideoCapture(0)

# Snapshot timing
last_snapshot_time = 0
snapshot_interval = 5  # seconds

# Alarm control
last_alarm_time = 0
alarm_cooldown = 10  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]  # Get results from YOLOv8

    high_threat_detected = False
    low_threat_detected = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in HIGH_THREATS:
            high_threat_detected = True
            color = (0, 0, 255)  # Red for high threat
        elif label in LOW_THREATS:
            low_threat_detected = True
            color = (0, 165, 255)  # Orange for low threat
        else:
            continue

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display threat message and handle alarm/snapshots
    current_time = time.time()

    if high_threat_detected:
        cv2.putText(frame, "HIGH THREAT DETECTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Alarm only if cooldown passed
        if current_time - last_alarm_time > alarm_cooldown:
            alarm_sound.play()
            last_alarm_time = current_time

        # Take snapshot if 5 sec passed
        if current_time - last_snapshot_time > snapshot_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"snapshots/high_{timestamp}.jpg"
            cv2.imwrite(path, frame)
            print(f"[HIGH THREAT] Snapshot saved: {path}")
            last_snapshot_time = current_time

    elif low_threat_detected:
        cv2.putText(frame, "LOW THREAT (Knife) DETECTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

        # Only take snapshot, no alarm
        if current_time - last_snapshot_time > snapshot_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"snapshots/low_{timestamp}.jpg"
            cv2.imwrite(path, frame)
            print(f"[LOW THREAT] Snapshot saved: {path}")
            last_snapshot_time = current_time

    # Display output
    cv2.imshow("Threat Detection - YOLOv8", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
