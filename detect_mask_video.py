import cv2
import numpy as np
import time
import os
import winsound
import csv
from datetime import datetime

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------- LOAD MODELS --------------------
faceNet = cv2.dnn.readNet(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)

maskNet = load_model("mask_detector.keras")

# -------------------- SETUP --------------------
os.makedirs("violations", exist_ok=True)

mask_count = 0
no_mask_count = 0
violations = 0

last_alert_time = 0
COOLDOWN = 3

prev_time = 0

# Tracking
next_object_id = 0
objects = {}

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('violation.avi', fourcc, 20.0, (640, 480))

# -------------------- FUNCTIONS --------------------
def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def detect(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs = [], []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face = frame[y1:y2, x1:x2]
            if face.shape[0] == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_array = img_to_array(face_resized)
            face_array = preprocess_input(face_array)

            faces.append(face_array)
            locs.append((x1, y1, x2, y2, face))

    preds = []
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces)

    return locs, preds


# -------------------- START CAMERA --------------------
cap = cv2.VideoCapture(0)
print("[INFO] FINAL SYSTEM RUNNING...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    locs, preds = detect(frame)
    current_time = time.time()

    face_count = len(locs)
    current_objects = {}

    for (box, pred) in zip(locs, preds):
        (x1, y1, x2, y2, face_img) = box
        (mask, noMask) = pred

        label = "Mask" if mask > noMask else "No Mask"
        confidence = max(mask, noMask)

        centroid = get_centroid((x1, y1, x2, y2))

        matched = False
        for obj_id, prev_centroid in objects.items():
            distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))

            if distance < 50:
                current_objects[obj_id] = centroid
                matched = True
                break

        if not matched:
            current_objects[next_object_id] = centroid
            next_object_id += 1

        objects = current_objects

        # -------------------- ALERT + SAVE --------------------
        if label == "No Mask" and (current_time - last_alert_time > COOLDOWN):
            violations += 1
            no_mask_count += 1

            winsound.Beep(1200, 400)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violations/violation_{timestamp}.jpg"
            cv2.imwrite(filename, face_img)

            out.write(frame)

            last_alert_time = current_time

        if label == "Mask":
            mask_count += 1

        # -------------------- DRAW --------------------
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {confidence*100:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------- FPS --------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # -------------------- DASHBOARD --------------------
    cv2.rectangle(frame, (0, 0), (300, 170), (50, 50, 50), -1)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Faces: {face_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Mask: {mask_count}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"No Mask: {no_mask_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"Violations: {violations}", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if face_count > 3:
        cv2.putText(frame, "Crowd Detected!", (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("🔥 FINAL MASK SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------- SAVE REPORT --------------------
with open("report.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Mask", "No Mask", "Violations"])
    writer.writerow([mask_count, no_mask_count, violations])

print("\n===== SESSION SUMMARY =====")
print(f"Mask: {mask_count}")
print(f"No Mask: {no_mask_count}")
print(f"Violations: {violations}")

cap.release()
out.release()
cv2.destroyAllWindows()