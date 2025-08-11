import cv2
import os
import numpy as np
import pickle
import shutil
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
# <<< ใหม่: นำเข้า MediaPipe >>>
import mediapipe as mp
from scipy.spatial import distance as dist

# --- Global Settings ---
dataset_path = 'dataset'
camera_index = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_size = (200, 200)

# <<< ใหม่: การตั้งค่า MediaPipe Face Mesh >>>
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# <<< ใหม่: Indices ของ Landmark สำหรับดวงตาใน MediaPipe (มีทั้งหมด 478 จุด) >>>
# จุดสำคัญรอบดวงตาซ้ายและขวาสำหรับคำนวณ EAR
# [ขอบตาแนวนอนซ้าย, ขอบตาแนวตั้งบน, ขอบตาแนวตั้งล่าง, ขอบตาแนวนอนขวา]
EYE_INDICES_LEFT = [362, 385, 387, 263, 373, 380]
EYE_INDICES_RIGHT = [33, 160, 158, 133, 153, 144]

# ฟังก์ชันนี้ยังใช้ได้เหมือนเดิม ไม่ต้องแก้
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- 1. Capture Faces (ไม่มีการเปลี่ยนแปลง) ---
def capture_faces():
    person_name = input("Enter person's name (English, no spaces): ").strip()
    if not person_name:
        print("❌ Invalid name.")
        return
    save_path = os.path.join(dataset_path, person_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam.")
        return
    print("\nLook at the camera. Press 'q' to quit.")
    count = 0
    samples_to_take = 150
    while count < samples_to_take:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = cv2.resize(gray[y:y+h, x:x+w], img_size)
            cv2.imwrite(os.path.join(save_path, f"{count+1}.jpg"), face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            text = f"Saved: {count}/{samples_to_take}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished capturing {count} images for '{person_name}'.")

# --- 2. Train Model (ไม่มีการเปลี่ยนแปลง) ---
# --- 2. Train Model (แก้ไขให้มีการ resize รูปภาพ) ---

# --- 2. Train Model (เวอร์ชันอัปเกรดครั้งใหญ่) ---
def train_model():
    print("\n🔄 Loading images and training CNN model...")
    X, y = [], []

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ Dataset folder is empty. Please capture faces first.")
        return

    for folder_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(person_path): continue

        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: continue
            
            image = cv2.resize(image, img_size) # Ensure all images are the same size
            X.append(image)
            y.append(folder_name)

    if len(set(y)) < 2:
        print("❌ Training requires at least 2 different people. Please add more data.")
        return

    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1) / 255.0

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    num_classes = len(le.classes_)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)), MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'), MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'), MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_categorical, epochs=20, batch_size=16, validation_split=0.2)

    model.save('cnn_model.h5')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("✅ Training complete! Model saved as 'cnn_model.h5'.")

# --- 3. Recognize Faces (อัปเกรดเป็น MediaPipe) ---
def recognize_faces():
    try:
        model = load_model('cnn_model.h5')
        with open('label_encoder.pkl', 'rb') as f: le = pickle.load(f)
    except Exception as e:
        print(f"❌ Model/encoder not found ({e}). Please train first.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("\nStarting secure recognition (using MediaPipe). Please blink to verify.")
    EAR_THRESHOLD, EAR_CONSEC_FRAMES = 0.2, 3
    CONFIDENCE_THRESHOLD = 85.0
    blink_counter = 0
    verified = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        # MediaPipe ต้องการภาพสี BGR
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        display_text = "Blink to Verify"
        display_color = (0, 165, 255) # สีส้ม

        if results.multi_face_landmarks:
            # ใช้ใบหน้าแรกที่เจอ
            face_landmarks = results.multi_face_landmarks[0]
            
            # --- คำนวณ EAR จาก MediaPipe Landmarks ---
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=int)
            
            left_eye = np.array([landmarks[i] for i in EYE_INDICES_LEFT])
            right_eye = np.array([landmarks[i] for i in EYE_INDICES_RIGHT])
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EAR_CONSEC_FRAMES:
                    verified = True
                blink_counter = 0
            
            # --- วาดกรอบรอบใบหน้า (ต้องคำนวณเอง) ---
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            if verified:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_roi = gray[y_min:y_max, x_min:x_max]

                if face_roi.size == 0: continue # ข้ามถ้าใบหน้าเล็กไป
                
                face_roi_resized = cv2.resize(face_roi, img_size)
                face_roi_normalized = face_roi_resized.reshape(1, img_size[0], img_size[1], 1) / 255.0
                
                predictions = model.predict(face_roi_normalized, verbose=0)
                confidence = np.max(predictions) * 100
                
                if confidence > CONFIDENCE_THRESHOLD:
                    label_idx = np.argmax(predictions)
                    name = le.inverse_transform([label_idx])[0]
                    display_text = f"{name} ({confidence:.1f}%)"
                    display_color = (0, 255, 0) # สีเขียว
                else:
                    display_text = "Unknown"
                    display_color = (0, 0, 255) # สีแดง

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), display_color, 2)
            cv2.putText(frame, display_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        else:
            verified = False # รีเซ็ตถ้าไม่เจอหน้า

        cv2.imshow('Secure Face Recognition (MediaPipe)', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break
            
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

# --- 4. Delete Person Data (ไม่มีการเปลี่ยนแปลง) ---
def delete_person_data():
    person_name = input("Enter name of person's data to delete: ").strip()
    person_path = os.path.join(dataset_path, person_name)
    if os.path.exists(person_path):
        try:
            shutil.rmtree(person_path)
            print(f"✅ Successfully deleted '{person_name}'. Please retrain the model.")
        except OSError as e:
            print(f"❌ Error deleting folder: {e.strerror}")
    else:
        print(f"❌ Dataset for '{person_name}' not found.")

# --- Main Menu (ไม่มีการเปลี่ยนแปลง) ---
def main():
    if not os.path.exists(dataset_path): os.makedirs(dataset_path)
    while True:
        print("\n" + "="*30 + "\n   Face Recognition System\n" + "="*30)
        print("1. Capture new faces")
        print("2. Train robust model")
        print("3. Start secure recognition")
        print("4. Delete a person's data")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ").strip()
        if choice == '1': capture_faces()
        elif choice == '2': train_model()
        elif choice == '3': recognize_faces()
        elif choice == '4': delete_person_data()
        elif choice == '5': print("Exiting..."); break
        else: print("❌ Invalid choice.")
            
if __name__ == "__main__":
    main()
