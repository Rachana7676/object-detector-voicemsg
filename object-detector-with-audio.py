import cv2
import pyttsx3
from transformers import pipeline
from PIL import Image

# 1️⃣ Load object detection model
detector = pipeline("object-detection", model="hustvl/yolos-tiny")

# 2️⃣ Initialize webcam
cap = cv2.VideoCapture(0)

# 3️⃣ Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

spoken_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🔥 FIX: Convert NumPy → PIL image
    pil_image = Image.fromarray(rgb_frame)

    # 4️⃣ Perform object detection
    results = detector(pil_image)

    current_objects = set()

    for obj in results:
        label = obj["label"]
        score = obj["score"]
        box = obj["box"]

        if score < 0.5:
            continue

        x1, y1 = int(box["xmin"]), int(box["ymin"])
        x2, y2 = int(box["xmax"]), int(box["ymax"])

        # Draw bounding boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        current_objects.add(label)

    # Speak new objects
    new_objects = current_objects - spoken_objects
    if new_objects:
        text = "I see " + ", ".join(new_objects)
        print(text)
        engine.say(text)
        engine.runAndWait()
        spoken_objects.update(new_objects)

    # Show display
    cv2.imshow("Object Detection with Voice", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
