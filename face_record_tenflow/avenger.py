import face_recognition
import cv2
import os
import numpy as np

# --- ⚙️ การตั้งค่าหลัก ⚙️ ---
# 🚩 แก้ไข: ใช้ list [] เพื่อเก็บ path ของทุกคน
KNOWN_FACES_DIRS = [
    r"E:\Project By Tawan\dectect face\train\Tawan",
    # r"E:\Project By Tawan\dectect face\train\Dong",
    r"E:\Project By Tawan\dectect face\train\Poy",
    # r"E:\Project By Tawan\dectect face\train\Ho",
]

# Tolerance for face matching. Lower value means a stricter match.
TOLERANCE = 0.5

# Resize factor for real-time processing.
RESIZE_FACTOR = 4

# --- Functions ---
def load_known_faces():
    """
    Loads images from a list of specified directories and encodes them.
    The person's name is taken from the last part of the directory path.
    """
    print(f"✅ กำลังโหลดใบหน้าจาก {len(KNOWN_FACES_DIRS)} โฟลเดอร์...")
    known_face_encodings = []
    known_face_names = []

    # 🚩 แก้ไข: วนลูปอ่านทุก path ที่อยู่ใน KNOWN_FACES_DIRS
    for dir_path in KNOWN_FACES_DIRS:
        # Get the person's name from the directory name (e.g., 'Tawan')
        person_name = os.path.basename(dir_path)

        if not os.path.exists(dir_path):
            print(f"❌ Error: ไม่พบโฟลเดอร์ '{dir_path}'. ข้ามไป...")
            continue
        
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(dir_path, filename)
                try:
                    image = face_recognition.load_image_file(file_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                        print(f"   -> Loaded face of {person_name} from {filename}")
                    else:
                        print(f"   -> ⚠️ Warning: ไม่พบใบหน้าในรูปภาพ: {filename}")
                except Exception as e:
                    print(f"   -> ❌ Error processing {filename}: {e}")

    if not known_face_encodings:
        print("❌ ERROR: No valid faces found in any of the specified folders.")
        exit()
        
    print(f"✅ โหลดใบหน้าสำเร็จ: {len(known_face_encodings)} ใบหน้า")
    return known_face_encodings, known_face_names

# --- Main Logic ---
known_encodings, known_names = load_known_faces()

print("กำลังเปิดกล้อง...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: ไม่สามารถเปิดกล้องได้. โปรดตรวจสอบการเชื่อมต่อหรือสิทธิ์การเข้าถึง.")
    exit()

print("✅ เริ่มการตรวจจับแบบ Real-time... (กด 'q' เพื่อออกจากโปรแกรม)")
face_locations, face_names = [], []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Failed to read frame from webcam.")
        break
    
    frame = cv2.flip(frame, 1)

    if frame_count % 5 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=1/RESIZE_FACTOR, fy=1/RESIZE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"
            
            if True in matches:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            
            face_names.append(name)
    
    frame_count += 1

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= RESIZE_FACTOR
        right *= RESIZE_FACTOR
        bottom *= RESIZE_FACTOR
        left *= RESIZE_FACTOR

        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Face Recognition - By Tawan', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nUser quit the program.")
        break

cap.release()
cv2.destroyAllWindows()
print("Program terminated successfully.")
