import face_recognition
import cv2
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# --- ⚙️ Main Configurations ⚙️ ---
# สร้าง Path แบบสมบูรณ์เพื่อป้องกันข้อผิดพลาด
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# 1. Paths for Recognition
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pkl")

# 2. Paths for Face Detection (OpenCV's DNN Model)
PROTOTXT_PATH = os.path.join(BASE_DIR, "liveness_model", "deploy.prototxt")
WEIGHTS_PATH = os.path.join(BASE_DIR, "liveness_model", "res10_300x300_ssd_iter_140000.caffemodel")

# 3. Path for Liveness Detection Model
LIVENESS_MODEL_PATH = os.path.join(BASE_DIR, "liveness_model", "liveness.model")

# --- Model Parameters ---
RECOGNITION_TOLERANCE = 0.5  # เกณฑ์การยอมรับใบหน้า (ค่ายิ่งน้อย ยิ่งเข้มงวด)
LIVENESS_CONFIDENCE_THRESHOLD = 0.9 # เกณฑ์ความมั่นใจว่าเป็นคนจริง (แนะนำให้สูงๆ)
FACE_DETECTION_CONFIDENCE = 0.6 # เกณฑ์ความมั่นใจในการตรวจจับใบหน้า

# ==================== 🚀 1. โหลดโมเดลทั้งหมด 🚀 ====================

# --- โหลดโมเดล Face Recognition ---
print("[INFO] กำลังโหลดฐานข้อมูลใบหน้า (encodings.pkl)...")
if not os.path.exists(ENCODINGS_FILE):
    print(f"❌ ERROR: ไม่พบไฟล์ '{ENCODINGS_FILE}' กรุณารัน 'create_encodings.py' ก่อน")
    exit()
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

# --- โหลดโมเดล OpenCV Face Detector ---
print("[INFO] กำลังโหลดโมเดลตรวจจับใบหน้า (OpenCV DNN)...")
if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(WEIGHTS_PATH):
    print("❌ ERROR: ไม่พบไฟล์โมเดลของ OpenCV Face Detector")
    exit()
face_net = cv2.dnn.readNet(PROTOTXT_PATH, WEIGHTS_PATH)

# --- โหลดโมเดล Liveness Detection ---
print("[INFO] กำลังโหลดโมเดลตรวจสอบใบหน้าจริง (Liveness)...")
if not os.path.exists(LIVENESS_MODEL_PATH):
    print(f"❌ ERROR: ไม่พบไฟล์ '{LIVENESS_MODEL_PATH}'")
    exit()
liveness_net = load_model(LIVENESS_MODEL_PATH)


# ==================== 🎥 2. เริ่มต้นการทำงาน 🎥 ====================
print("\n✅ โมเดลทั้งหมดพร้อมใช้งาน! เริ่มเปิดกล้อง...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: ไม่สามารถเปิดกล้องได้")
    exit()

print("✅ เริ่มการตรวจสอบใบหน้าแบบ Real-time... (กด 'q' เพื่อออก)")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    (h, w) = frame.shape[:2]

    # ⭐️ ใช้ OpenCV DNN เพื่อตรวจจับตำแหน่งใบหน้าทั้งหมดในเฟรม
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # วนลูปใบหน้าที่ตรวจจับได้ทั้งหมด
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > FACE_DETECTION_CONFIDENCE:
            # คำนวณตำแหน่งของกรอบรอบใบหน้า
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            
            # ================= STAGE 1: LIVENESS CHECK =================
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size == 0: continue

            # Preprocess the face for the liveness model
            face_for_liveness = cv2.resize(face_roi, (32, 32))
            face_for_liveness = face_for_liveness.astype("float") / 255.0
            face_for_liveness = np.expand_dims(face_for_liveness, axis=0)
            
            # Predict liveness
            preds = liveness_net.predict(face_for_liveness, verbose=0)[0]
            j = np.argmax(preds)
            liveness_label = 'spoof' if j == 1 else 'real'
            liveness_confidence = preds[j]

            # ตั้งค่าเริ่มต้นสำหรับแสดงผล
            display_name = ""
            box_color = (0, 255, 255) # สีเหลือง (Checking...)
            
            # ================= STAGE 2: FACE RECOGNITION (IF REAL) =================
            if liveness_label == 'real' and liveness_confidence > LIVENESS_CONFIDENCE_THRESHOLD:
                # ถ้าเป็นคนจริง ให้เริ่มระบุตัวตน
                
                # Convert frame for face_recognition library
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert OpenCV box to face_recognition box format (top, right, bottom, left)
                face_location_dlib = [(startY, endX, endY, startX)]
                
                current_face_encoding = face_recognition.face_encodings(rgb_frame, face_location_dlib)

                name = "Unknown"
                if len(current_face_encoding) > 0:
                    matches = face_recognition.compare_faces(known_encodings, current_face_encoding[0], tolerance=RECOGNITION_TOLERANCE)
                    face_distances = face_recognition.face_distance(known_encodings, current_face_encoding[0])
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index] and face_distances[best_match_index] < RECOGNITION_TOLERANCE:
                            name = known_names[best_match_index]
                
                # กำหนดสีและข้อความตามผลลัพธ์
                if name != "Unknown":
                    display_name = f"{name}"
                    box_color = (0, 255, 0) # สีเขียว (Known Person)
                else:
                    display_name = "Unknown (Real)"
                    box_color = (0, 165, 255) # สีส้ม (Unknown Person)
            
            else: # ถ้าเป็นของปลอม (SPOOF)
                display_name = f"SPOOF ({liveness_confidence:.2f})"
                box_color = (0, 0, 255) # สีแดง (SPOOF)
            
            # --- วาดผลลัพธ์ลงบนเฟรม ---
            cv2.putText(frame, display_name, (startX, startY - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, box_color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)

    cv2.imshow("Face Recognition with Liveness Detection - By Tawan", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program terminated.")