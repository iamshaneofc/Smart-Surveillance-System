import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the trained model
model = load_model(r'Z:\security_camera\Violence_system\modelnew.h5')
labels = ["NonViolence", "Violence"]

# Path to video
video_path = r'Z:\security_camera\Violence_system\test_video.mp4'

if not os.path.exists(video_path):
    print(f"[❌] Video file not found at: {video_path}")
    exit()

# Load the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[❌] Failed to open video: {video_path}")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video path
output_path = r'Z:\security_camera\Violence_system\output_prediction.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    print(f"[❌] VideoWriter failed to open for: {output_path}")
    cap.release()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized = cv2.resize(frame, (128, 128))
    image = img_to_array(resized) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image, verbose=0)
    label_index = int(prediction[0] > 0.5)
    label = labels[label_index]

    # Color based on prediction
    color = (0, 0, 255) if label == "Violence" else (0, 255, 0)

    # Draw rectangle and label
    text = f"{label}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 20), color, -1)
    cv2.putText(frame, text, (15, 10 + text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write frame
    out.write(frame)

# Clean up
cap.release()
out.release()
print(f"✅ Output saved at: {output_path}")
