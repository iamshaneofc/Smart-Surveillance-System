# 🛡️ AI Surveillance System

The **AI Surveillance System** is a cutting-edge computer vision-based security solution designed to enhance monitoring in real-time environments such as schools, offices, malls, public spaces, and more. This project combines multiple deep learning models to detect suspicious activity, violent behavior, weapon presence, and whether individuals are wearing face masks—all through live video feeds. It brings together AI and security to create a powerful, automated, and intelligent monitoring tool.

This system aims to provide **real-time alerts** for potential threats and policy violations, helping prevent incidents before they escalate. Whether it's identifying a weapon in someone’s hand or detecting an ongoing fight, this AI system acts as a digital watchdog, making human surveillance smarter, faster, and more reliable.

---

## ⚙️ Tools & Technologies Used

- **🧠 Deep Learning (Keras, TensorFlow):** For training and deploying the models.
- **🧾 OpenCV:** For real-time image processing and video stream handling.
- **🔍 YOLO (You Only Look Once):** Used for object detection, particularly weapons.
- **🔌 Circuit Setup (optional):** Can be integrated with external alarm circuits or alert systems for real-time notifications.
- **🧰 Python:** The main programming language for building and orchestrating the components.

---

## 📚 Datasets Used

| Component            | Dataset Source | Description                                |
|---------------------|----------------|--------------------------------------------|
| **Face Mask Detection** | Kaggle         | Image dataset containing masked & unmasked faces |
| **Suspicious Activity** | Roboflow       | Custom-built image dataset for crawling, running, falling, climbing, etc. |
| **Violence Detection** | Kaggle         | Video dataset of real-life violent events (1000+ samples) |
| **Weapon Detection**   | Roboflow       | Gun and knife detection dataset for object recognition |

---

## 🔎 Project Modules

### 😷 1. Face Mask Detection
This module identifies whether a person is wearing a mask or not. It is especially useful in scenarios like post-pandemic public health enforcement, hospitals, or controlled entry zones where masks are mandatory.

- Helps enforce mask policies in real-time.
- Alerts security staff or blocks entry if a mask is not detected.

---

### 🥊 2. Violence Detection
Detects violent movements or physical fights using deep learning models trained on real-world violent video data. Ideal for schools, prisons, or public events where violence needs to be immediately spotted.

- Flags fights or aggressive behavior.
- Helps intervene faster in potentially dangerous scenarios.

---

### 🔫 3. Weapon Detection (Gun/Knife)
This module detects dangerous weapons like guns or knives using YOLO-based object detection models trained on Roboflow datasets.

- Crucial for detecting threats early.
- Helps security teams prepare or respond faster.

---
<h3>🚧 Suspicious Activity Detection (Under Construction)</h3>

<p>
  <strong>⚠️ Status:</strong> <em>This model is currently under development.</em><br>
  🧪 Accuracy is currently suboptimal and does not meet deployment standards.<br>
  🔄 Improvements and further training are in progress to enhance performance and reliability.
</p>


### 🏃 4. Suspicious Activity Detection
Detects activities like crawling, running, falling, climbing—any behavior that deviates from the normal. Trained on Roboflow dataset with labeled action frames.

- Can detect intrusions or escape attempts.
- Useful in border security, banks, or prisons.

---

## 🧠 Combined Power of AI for Smart Surveillance

Together, these modules form a unified surveillance system that watches over an environment with **24/7 intelligence**. Whether it’s detecting a person sneaking in, a fight breaking out, a weapon being drawn, or a health guideline being ignored—**this AI-powered security system provides real-time awareness, alerting, and logging for efficient and proactive safety management**.

---

## 📌 Future Improvements

- Add face recognition to identify specific individuals.
- Integrate with cloud for remote alerting.
- Add voice alerts or messaging systems.
- Extend detection to include fire, smoke, or medical emergencies.

---

## 👨‍💻 Author

**Snehanshu(Shane)** — [@iamshaneofc](https://github.com/iamshaneofc)  
"Making the world a safer place, one model at a time."

---

> ⭐ Star this repository if you found it helpful.  
> 🛠️ Pull requests and suggestions are welcome!

