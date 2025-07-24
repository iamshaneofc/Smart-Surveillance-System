import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==== 1. Config ====
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['climbing', 'crawling', 'falling', 'sitting', 'standing', 'walking']

suspicious_actions = {'climbing', 'crawling', 'falling'}
normal_actions = {'sitting', 'standing', 'walking'}

# ==== 2. Load Model ====
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("human_action_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ==== 3. Transform ====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ==== 4. Start Camera ====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        max_prob, predicted = torch.max(probs, 1)
        confidence = max_prob.item()
        label = class_names[predicted.item()]
        print(f"Predicted: {label} ({confidence:.2f})")

    # ==== Show label only if confident enough ====
    if confidence >= 0.6:
        if label in suspicious_actions:
            color = (0, 0, 255)  # Red
            text = f"Suspicious: {label}"
        elif label in normal_actions:
            color = (0, 255, 0)  # Green
            text = f"Action: {label}"
        else:
            color = None
            text = ""

        if color:
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # else: do not display anything

    cv2.imshow("Action Detection", frame)

    if label in suspicious_actions and confidence >= 0.6:
        print("⚠️ Suspicious activity detected:", label)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
