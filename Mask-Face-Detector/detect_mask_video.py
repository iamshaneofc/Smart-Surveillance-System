import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load face detector model
face_net = cv2.dnn.readNetFromCaffe(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)

# Load mask detection model
mask_model = load_model('mask_detector.h5')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Ensure bounding box stays within the frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_array = img_to_array(face_resized)
            face_array = np.expand_dims(face_array, axis=0) / 255.0

            # Predict mask vs no mask
            (mask, no_mask) = mask_model.predict(face_array, verbose=0)[0]

            label = "Mask" if mask > no_mask else "Face"
            confidence_percent = max(mask, no_mask) * 100
            label_text = f"{label}: {confidence_percent:.2f}%"

            color = (0, 0, 255) if label == "Mask" else (0, 255, 0)

            # Draw rectangle & label
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
