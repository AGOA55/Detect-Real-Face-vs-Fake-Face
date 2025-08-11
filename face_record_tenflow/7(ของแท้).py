import cv2
import os
import numpy as np
import pickle
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial import distance as dist
import mediapipe as mp
from mtcnn import MTCNN
from keras_facenet import FaceNet

# --- Global Settings ---
dataset_path = 'dataset'
embeddings_path = 'face_embeddings.pkl'
camera_index = 0
img_size = (160, 160)  # FaceNet requires 160x160 images

# --- Initializing Models ---
detector = MTCNN()  # Using MTCNN for face detection
facenet_model = FaceNet()
print("✅ FaceNet Model Loaded.")

# <<< ใหม่: การตั้งค่า MediaPipe Face Mesh >>>
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# <<< ใหม่: Indices ของ Landmark สำหรับดวงตาใน MediaPipe >>>
# [ขอบตาแนวนอนซ้าย, ขอบตาแนวตั้งบน, ขอบตาแนวตั้งล่าง, ขอบตาแนวนอนขวา]
EYE_INDICES_LEFT = [362, 385, 387, 263, 373, 380]
EYE_INDICES_RIGHT = [33, 160, 158, 133, 153, 144]

# --- Helper Function ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- 1. Capture Faces (ปรับปรุงเล็กน้อย) ---
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
    samples_to_take = 50  # Capture fewer images since FaceNet needs fewer samples

    while count < samples_to_take:
        ret, frame = cap.read()
        if not ret: break

        # Use MTCNN for robust face detection during capture
        faces = detector.detect_faces(frame)

        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            x1, y1 = abs(x), abs(y)
            x2, y2 = x1 + w, y1 + h
            face_img = frame[y1:y2, x1:x2]

            if face_img.size > 0:
                face_img_resized = cv2.resize(face_img, img_size)
                # FaceNet works best with BGR images, no need for grayscale
                cv2.imwrite(os.path.join(save_path, f"{count+1}.jpg"), face_img_resized)
                count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"Saved: {count}/{samples_to_take}"
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished capturing {count} images for '{person_name}'.")

# --- 2. Train Model (เปลี่ยนเป็น Generate Embeddings) ---
def generate_embeddings():
    print("\n🔄 Generating Face Embeddings...")
    embeddings = {}

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ Dataset folder is empty. Please capture faces first.")
        return

    for folder_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(person_path): continue

        person_embeddings = []
        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            image = cv2.imread(img_path)
            if image is None: continue

            # Use FaceNet to get embedding
            face_embedding = facenet_model.embeddings([image])[0]
            person_embeddings.append(face_embedding)

        if person_embeddings:
            # Take the average embedding for a single, robust representation
            embeddings[folder_name] = np.mean(person_embeddings, axis=0)

    if not embeddings:
        print("❌ No valid images found to generate embeddings.")
        return

    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print("✅ Embeddings generated and saved to 'face_embeddings.pkl'.")

# --- 3. Recognize Faces (ปรับปรุงใหม่ด้วย FaceNet) ---
def apply_clahe_color(img):
    # ปรับแสงภาพด้วย CLAHE บนช่อง L ใน LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final

def recognize_faces():
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
    except FileNotFoundError:
        print("❌ Embeddings file not found. Please generate embeddings first.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("\nStarting secure recognition (using FaceNet). Please blink to verify.")
    EAR_THRESHOLD, EAR_CONSEC_FRAMES = 0.2, 3
    SIMILARITY_THRESHOLD = 0.65  # Cosine Similarity Threshold
    blink_counter = 0
    verified = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        display_text = "Blink to Verify"
        display_color = (0, 165, 255)  # Orange

        faces = detector.detect_faces(frame)

        if results.multi_face_landmarks and faces:
            # --- EAR Verification ---
            face_landmarks = results.multi_face_landmarks[0]
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

            # --- Face Recognition (after verification) ---
            x_min, y_min, w_mtcnn, h_mtcnn = faces[0]['box']
            x_max, y_max = x_min + w_mtcnn, y_min + h_mtcnn

            if verified:
                face_roi = frame[y_min:y_max, x_min:x_max]
                if face_roi.size == 0:
                    continue

                # ปรับแสงด้วย CLAHE
                face_roi_clahe = apply_clahe_color(face_roi)

                # Resize และแปลงสีให้เหมาะสมกับ FaceNet
                face_roi_resized = cv2.resize(face_roi_clahe, img_size)
                face_roi_rgb = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2RGB)

                live_embedding = facenet_model.embeddings([face_roi_rgb])[0]

                best_match_name = "Unknown"
                best_similarity = 0

                for name, stored_embedding in embeddings.items():
                    similarity = 1 - dist.cosine(live_embedding, stored_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_name = name

                if best_similarity > SIMILARITY_THRESHOLD:
                    display_text = f"{best_match_name} ({best_similarity:.2f})"
                    display_color = (0, 255, 0)  # Green
                else:
                    display_text = "Unknown"
                    display_color = (0, 0, 255)  # Red

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), display_color, 2)
            cv2.putText(frame, display_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        else:
            verified = False

        cv2.imshow('Secure Face Recognition (FaceNet)', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

# --- ฟังก์ชันแสดงรายชื่อคนในฐานข้อมูล ---
def list_persons():
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder does not exist.")
        return []
    persons = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    if not persons:
        print("❌ No persons found in the dataset.")
    else:
        print("\n📋 Persons in database:")
        for idx, name in enumerate(persons, 1):
            print(f"{idx}. {name}")
    return persons

# --- ปรับปรุงฟังก์ชันลบข้อมูลให้เลือกจากรายชื่อ ---
def delete_person_data():
    persons = list_persons()
    if not persons:
        return
    try:
        choice = input("Enter the number of the person to delete (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            print("Canceled deletion.")
            return
        choice_num = int(choice)
        if 1 <= choice_num <= len(persons):
            person_name = persons[choice_num - 1]
            person_path = os.path.join(dataset_path, person_name)
            confirm = input(f"Are you sure you want to delete all data for '{person_name}'? (y/n): ").strip().lower()
            if confirm == 'y':
                shutil.rmtree(person_path)
                print(f"✅ Successfully deleted '{person_name}'. Please generate embeddings again.")
            else:
                print("Deletion canceled.")
        else:
            print("❌ Invalid number.")
    except ValueError:
        print("❌ Invalid input.")

# --- Main Menu (ปรับปรุงคำสั่ง) ---
def main():
    if not os.path.exists(dataset_path): os.makedirs(dataset_path)
    while True:
        print("\n" + "="*30 + "\n   Secure Face Recognition System\n" + "="*30)
        print("1. Capture new faces")
        print("2. Generate Face Embeddings")
        print("3. Start secure recognition")
        print("4. Show persons in database")
        print("5. Delete a person's data")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ").strip()
        if choice == '1':
            capture_faces()
        elif choice == '2':
            generate_embeddings()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            list_persons()
        elif choice == '5':
            delete_person_data()
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
