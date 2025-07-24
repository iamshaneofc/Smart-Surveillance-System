import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model(r'Z:\security_camera\Violence_system\modelnew.h5')
labels = [" Normal ", "Suspicious: Violence"]
# Open the webcam (0 = default)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows fix

if not cap.isOpened():
    print("[❌] Failed to access the camera.")
    exit()

print("[✅] Camera started. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[❌] Failed to grab frame.")
            break

        resized = cv2.resize(frame, (128, 128))
        image = img_to_array(resized) / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image, verbose=0)
        label_index = int(prediction[0] > 0.5)
        label = labels[label_index]

        color = (0, 0, 255) if label == "Violence" else (0, 255, 0)

        text = f"{label}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 20), color, -1)
        cv2.putText(frame, text, (15, 10 + text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the result window
        cv2.imshow("Fight Detection - Live Feed", frame)

        # Break loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[🛑] Interrupted by user.")

finally:
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        print("[⚠️] Could not destroy OpenCV windows (GUI issue).")

print("[✅] Camera stopped.")
