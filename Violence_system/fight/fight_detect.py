import cv2
from ultralytics import YOLO

# Load your custom trained model
model = YOLO(r'Z:\Security_Surveillance_System\Violence_system\fight\Yolo_nano_weights.pt')  # Replace with your trained model path

# Start webcam (0 = default webcam, change if using external cam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 inference on the current frame
    results = model(frame)

    # Loop through detections
    for detection in results[0].boxes:
        class_id = int(detection.cls)
        confidence = float(detection.conf)
        x1, y1, x2, y2 = map(int, detection.xyxy[0])

        # Only process class 1 (Violence)
        if class_id == 1:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'Violence {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show the result
    cv2.imshow("Violence Detection", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()