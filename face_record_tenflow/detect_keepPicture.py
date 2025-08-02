import face_recognition
import cv2
import os
import numpy as np

# --- ⚙️ Main Configurations ⚙️ ---
# Path to the folder containing known faces.
KNOWN_FACES_DIR = r"E:\Project By Tawan\dectect face\train\Tawan"

# Tolerance for face matching. Lower value means a stricter match.
# 🚩 แก้ไข: ลดค่า TOLERANCE ลงเพื่อให้การจดจำเข้มงวดขึ้น
TOLERANCE = 0.5 

# Resize factor for real-time processing. Higher value means faster but less accurate.
RESIZE_FACTOR = 4

# --- Functions ---
def load_known_faces():
    """
    Loads all images from the KNOWN_FACES_DIR and encodes them.
    The person's name is taken from the directory name.
    """
    print(f"✅ กำลังโหลดใบหน้าจากโฟลเดอร์: {KNOWN_FACES_DIR}")
    known_face_encodings = []
    known_face_names = []

    person_name = os.path.basename(KNOWN_FACES_DIR)

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(file_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)
                    print(f"   -> Loaded face of {person_name} from {filename}")
                else:
                    print(f"   -> ⚠️ Warning: No face found in image: {filename}")
            except Exception as e:
                print(f"   -> ❌ Error processing {filename}: {e}")

    if not known_face_encodings:
        print("❌ ERROR: No valid faces found in the specified folder.")
        print("Please check that the images contain a clear face.")
        exit()
        
    print(f"✅ Successfully loaded {len(known_face_encodings)} faces for '{person_name}'.")
    return known_face_encodings, known_face_names

# --- Main Logic ---
known_encodings, known_names = load_known_faces()

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not open webcam. Please check your connection or permissions.")
    exit()

print("✅ Webcam opened successfully. Starting real-time face recognition. (Press 'q' to quit)")
face_locations, face_names = [], []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Failed to read frame from webcam.")
        break
    
    frame = cv2.flip(frame, 1)

    # Process every N frames for better performance
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

    # Draw results on the frame
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

