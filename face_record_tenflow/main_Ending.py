# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import pickle
import shutil
import time
import subprocess
import sys
import ctypes
from scipy.spatial import distance as dist
from mtcnn import MTCNN
from keras_facenet import FaceNet
import serial
import mediapipe as mp
import serial.tools.list_ports
import psutil

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# --- Global Settings ---
dataset_path = 'dataset'
embeddings_path = 'face_embeddings.pkl'
liveness_dataset_path = 'liveness_dataset'
liveness_model_path = 'liveness_model.h5'
camera_index = 0
img_size = (160, 160)
liveness_img_size = (64, 64)

# --- Initializing Models ---
print("🔄 Initializing models, please wait...")
detector = MTCNN()
facenet_model = FaceNet()
try:
    liveness_model = load_model(liveness_model_path)
    print("✅ Liveness Model Loaded.")
except Exception:
    liveness_model = None
    print("⚠️ Liveness Model not found. Please train it using option 7.")

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
# *** MODIFIED: Set max_num_faces to 1 for efficiency ***
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, # 只偵測一張人臉
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

print("✅ All Models (MTCNN, FaceNet, MediaPipe) Loaded.")

# --- Arduino Connection Functions ---
def is_admin():
    """ตรวจสอบสิทธิ์ Administrator"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def kill_arduino_processes():
    """ฆ่า process ที่อาจใช้พอร์ต COM"""
    processes_to_kill = ['arduino', 'arduinoide', 'python', 'putty', 'teraterm', 'serial']
    killed_count = 0

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            proc_name = proc.info['name'].lower()
            for target in processes_to_kill:
                if target in proc_name and proc.info['pid'] != os.getpid():
                    try:
                        proc.terminate()
                        killed_count += 1
                        print(f"🔪 Terminated: {proc.info['name']} (PID: {proc.info['pid']})")
                    except:
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if killed_count > 0:
        time.sleep(2)
        print(f"✅ Terminated {killed_count} processes")
    else:
        print("ℹ️ No conflicting processes found")

    return killed_count > 0

def reset_com_port_registry(port_name):
    """รีเซ็ต registry สำหรับ COM port"""
    try:
        if not is_admin():
            print("⚠️ Need administrator privileges for registry operations")
            return False

        reg_paths = [
            f'HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\COM Name Arbiter\\Devices\\{port_name}',
            f'HKEY_LOCAL_MACHINE\\HARDWARE\\DEVICEMAP\\SERIALCOMM',
        ]

        for reg_path in reg_paths:
            try:
                subprocess.run(['reg', 'delete', reg_path, '/f'],
                             capture_output=True, stderr=subprocess.DEVNULL)
            except:
                pass

        print(f"🗂️ Registry cleaned for {port_name}")
        return True
    except Exception as e:
        print(f"⚠️ Registry reset failed: {e}")
        return False

def disable_enable_device(port_name):
    """ปิด-เปิด USB device ผ่าน PowerShell"""
    try:
        ps_script = f'''
        $devices = Get-WmiObject -Class Win32_PnPEntity | Where-Object {{$_.Name -like "*CH340*" -or $_.Name -like "*{port_name}*"}}
        foreach ($device in $devices) {{
            $deviceID = $device.DeviceID
            Write-Output "Disabling device: $deviceID"
            (Get-WmiObject -Class Win32_PnPEntity | Where-Object {{$_.DeviceID -eq $deviceID}}).Disable()
            Start-Sleep -Seconds 3
            Write-Output "Enabling device: $deviceID"
            (Get-WmiObject -Class Win32_PnPEntity | Where-Object {{$_.DeviceID -eq $deviceID}}).Enable()
            Start-Sleep -Seconds 5
        }}
        '''

        result = subprocess.run(['powershell', '-Command', ps_script],
                              capture_output=True, text=True, timeout=20)

        if result.returncode == 0:
            print(f"🔄 Device reset successful for {port_name}")
            return True
        else:
            print(f"⚠️ Device reset failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Device reset error: {e}")
        return False

def advanced_port_recovery(port_name):
    """การกู้คืนพอร์ตแบบขั้นสูง"""
    print(f"\n🛠️ Advanced recovery for {port_name}...")
    steps_successful = 0

    print("1️⃣ Terminating conflicting processes...")
    if kill_arduino_processes():
        steps_successful += 1
        time.sleep(2)

    print("2️⃣ Cleaning registry entries...")
    if reset_com_port_registry(port_name):
        steps_successful += 1

    print("3️⃣ Resetting USB device...")
    if disable_enable_device(port_name):
        steps_successful += 1

    print("4️⃣ Waiting for device to be ready...")
    time.sleep(5)
    steps_successful += 1

    print(f"✅ Recovery completed: {steps_successful}/4 steps successful")
    return steps_successful >= 2

def test_arduino_connection_multi_port():
    """ลองเชื่อมต่อหลายพอร์ตและหลายความเร็ว"""
    ports_to_try = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8']
    baud_rates = [9600, 115200, 57600]

    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    print(f"\n🔍 Available ports: {available_ports}")

    for port in ports_to_try:
        if port not in available_ports:
            continue

        for baud_rate in baud_rates:
            try:
                print(f"🔄 Trying {port} at {baud_rate} baud...")
                ser = serial.Serial(port, baud_rate, timeout=1)
                time.sleep(2)
                ser.write(b'R')
                time.sleep(0.5)
                ser.write(b'O')
                print(f"✅ Success! Connected to {port} at {baud_rate} baud")
                return ser
            except serial.SerialException as e:
                if "PermissionError" not in str(e):
                    print(f"❌ {port}@{baud_rate}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ {port}@{baud_rate}: {e}")
                continue

    return None

def find_processes_using_port(port_name):
    """หาโปรเซสที่ใช้พอร์ตอยู่"""
    using_processes = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if hasattr(conn, 'laddr') and conn.laddr:
                        if port_name.upper() in str(conn).upper():
                            using_processes.append(f"PID {proc.info['pid']}: {proc.info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:
        pass
    return using_processes

def list_available_ports():
    """แสดงพอร์ตที่มีอยู่ทั้งหมด"""
    ports = serial.tools.list_ports.comports()
    print("\n📋 Available COM Ports:")
    for i, port in enumerate(ports):
        print(f"{i+1}. {port.device} - {port.description}")
    return [port.device for port in ports]

def test_arduino_connection(force_reset=False):
    """ทดสอบการเชื่อมต่อ Arduino พร้อมแก้ไขปัญหา Permission"""
    available_ports = list_available_ports()

    if not available_ports:
        print("❌ No COM ports found!")
        return None

    for port in available_ports:
        try:
            print(f"\n🔄 Trying to connect to {port}...")
            ser = serial.Serial(port, 9600, timeout=2)
            time.sleep(3)
            print("📤 Sending test commands...")
            ser.write(b'R')
            time.sleep(1)
            ser.write(b'G')
            time.sleep(1)
            ser.write(b'O')
            print(f"✅ Successfully connected to Arduino on {port}")
            return ser
        except serial.SerialException as e:
            if "PermissionError" in str(e) or "Access is denied" in str(e):
                print(f"🔒 Permission denied on {port}")
                print("🔍 Checking processes using this port...")
                processes = find_processes_using_port(port)
                if processes:
                    print("📋 Processes using this port:")
                    for proc in processes:
                        print(f"   - {proc}")
                if force_reset or input(f"🔄 Try advanced recovery for {port}? (y/n): ").lower() == 'y':
                    if advanced_port_recovery(port):
                        try:
                            ser = serial.Serial(port, 9600, timeout=2)
                            time.sleep(3)
                            ser.write(b'R')
                            print(f"✅ Successfully connected after recovery!")
                            return ser
                        except Exception as e2:
                            print(f"❌ Still failed after recovery: {e2}")
                    else:
                        print(f"❌ Recovery failed for {port}")
                else:
                    print(f"⏭️ Skipping {port}")
            else:
                print(f"❌ Failed to connect to {port}: {e}")
            continue

    print("\n🔄 Trying multi-port scan...")
    return test_arduino_connection_multi_port()

# --- Helper Functions ---
def eye_aspect_ratio(all_face_landmarks, img_w, img_h, is_right):
    if is_right:
        indices = RIGHT_EYE_INDICES
    else:
        indices = LEFT_EYE_INDICES

    eye_landmarks = [all_face_landmarks[i] for i in indices]
    coords = np.array([(int(lm.x * img_w), int(lm.y * img_h)) for lm in eye_landmarks])

    v1 = dist.euclidean(coords[1], coords[15])
    v2 = dist.euclidean(coords[2], coords[14])
    v3 = dist.euclidean(coords[3], coords[13])
    v4 = dist.euclidean(coords[4], coords[12])
    v5 = dist.euclidean(coords[5], coords[11])
    v6 = dist.euclidean(coords[6], coords[10])

    h = dist.euclidean(coords[0], coords[8])

    ear = (v1 + v2 + v3 + v4 + v5 + v6) / (6.0 * h) if h != 0 else 0
    return ear

def apply_clahe_color(img):
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    except cv2.error:
        return img

# --- Main Functions ---
def capture_faces():
    person_name = input("Enter person's name (English, no spaces): ").strip()
    if not person_name:
        print("❌ Invalid name.")
        return
    save_path = os.path.join(dataset_path, person_name)
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam.")
        return
    print("\nLook at the camera. Press 's' to start capturing, 'q' to quit.")
    count, samples_to_take, capturing = 0, 50, False
    while True:
        ret, frame = cap.read()
        if not ret: break
        display_frame = frame.copy()
        faces = detector.detect_faces(frame)
        largest_face, max_area = None, 0
        if faces:
            for face in faces:
                x, y, w, h = face['box']
                if w * h > max_area:
                    max_area, largest_face = w * h, face
        if largest_face:
            x, y, w, h = largest_face['box']
            x1, y1, x2, y2 = abs(x), abs(y), abs(x) + w, abs(y) + h
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if capturing and count < samples_to_take:
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    cv2.imwrite(os.path.join(save_path, f"{count+1}.jpg"), cv2.resize(face_img, img_size))
                    count += 1
                    cv2.putText(display_frame, f"Saved: {count}/{samples_to_take}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif count >= samples_to_take:
                 cv2.putText(display_frame, "Capture Complete!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if not capturing:
             cv2.putText(display_frame, "Press 's' to start", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        cv2.imshow('Capturing Faces', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            capturing = True
            print("Starting capture...")
        if key == ord('q') or count >= samples_to_take: break
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished capturing {count} images for '{person_name}'.")

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
            try:
                image = cv2.imread(img_path)
                if image is None: continue
                face_embedding = facenet_model.embeddings([image])[0]
                person_embeddings.append(face_embedding)
            except Exception as e:
                print(f"⚠️ Warning: Could not process {img_path}. Error: {e}")
        if person_embeddings:
            embeddings[folder_name] = np.mean(person_embeddings, axis=0)
    if not embeddings:
        print("❌ No valid images found to generate embeddings.")
        return
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print("✅ Embeddings generated and saved to 'face_embeddings.pkl'.")

def capture_liveness_data(data_type):
    if data_type not in ['real', 'spoof']:
        print("❌ Invalid type. Please enter 'real' or 'spoof'.")
        return

    save_path = os.path.join(liveness_dataset_path, data_type)
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return
    instruction = "Show your REAL face to the camera." if data_type == 'real' else "Show a SPOOF face (phone/tablet/photo) to the camera."
    print(f"\n{instruction}\nCapturing will start in 3 seconds. Press 'q' to stop.")
    time.sleep(3)
    count = len(os.listdir(save_path))
    capture_limit = 300
    while count < capture_limit:
        ret, frame = cap.read()
        if not ret: break
        display_frame = frame.copy()
        faces = detector.detect_faces(frame)
        if faces:
            # Find the largest face
            largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
            x, y, w, h = largest_face['box']
            x1, y1, x2, y2 = abs(x), abs(y), x + w, y + h
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                img_path = os.path.join(save_path, f'{data_type}_{time.time()}.jpg')
                cv2.imwrite(img_path, face_roi)
                count += 1
            color = (0, 255, 0) if data_type == 'real' else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            progress_text = f"Saved: {count}/{capture_limit}"
            cv2.putText(display_frame, progress_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow(f'Capturing Liveness Data ({data_type.upper()})', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Finished. Captured {count} images for '{data_type}'.")

def train_liveness_model():
    if not os.path.exists(liveness_dataset_path):
        print("❌ Liveness dataset folder not found. Please capture data first.")
        return
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                 rotation_range=20, width_shift_range=0.2,
                                 height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True,
                                 fill_mode='nearest')
    train_generator = datagen.flow_from_directory(
        liveness_dataset_path,
        target_size=liveness_img_size,
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        liveness_dataset_path,
        target_size=liveness_img_size,
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    if train_generator.samples == 0:
        print("❌ No training data found. Please capture data first.")
        return
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(liveness_img_size[0], liveness_img_size[1], 3)),
        BatchNormalization(), MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding="same", activation='relu'),
        BatchNormalization(), MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), padding="same", activation='relu'),
        BatchNormalization(), MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    print("\n🧠 Starting Liveness Model Training (Upgraded)...")
    model.fit(train_generator, epochs=30, validation_data=validation_generator)
    model.save(liveness_model_path)
    print(f"✅ Liveness model trained and saved to {liveness_model_path}")
    global liveness_model
    liveness_model = load_model(liveness_model_path)


# === MODIFIED MAIN FUNCTION: recognize_faces for SINGLE person ===
def recognize_faces():
    if liveness_model is None:
        print("\n❌ Liveness Model required. Please run option 7 to train it.")
        return
    try:
        with open(embeddings_path, 'rb') as f:
            known_embeddings = pickle.load(f)
    except FileNotFoundError:
        print("\n❌ Embeddings file not found. Please generate embeddings first.")
        return

    # --- Arduino Connection ---
    ser = None
    try:
        print("\n🔄 Setting up Arduino connection...")
        ser = test_arduino_connection(force_reset=False)
        if ser:
            print("✅ Arduino connected! Testing lights...")
            ser.write(b'R'); time.sleep(1)
            ser.write(b'G'); time.sleep(1)
            ser.write(b'O'); time.sleep(1)
            ser.write(b'R') # Initial state
        else:
            print("⚠️ Running in visual-only mode.")
    except Exception as e:
        print(f"❌ Arduino setup error: {e}")
        ser = None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        if ser: ser.close()
        return

    print("\nStarting SINGLE-FACE Secure Recognition...")
    print("Look at the camera and BLINK to be verified.")

    # --- Parameters ---
    SIMILARITY_THRESHOLD = 0.70
    EAR_THRESHOLD = 0.20
    BLINK_CONSEC_FRAMES = 2
    UNLOCK_DURATION = 5.0

    # --- State Variables for a single person ---
    blink_counter = 0
    status_text = "❓ Unknown"
    status_color = (0, 0, 255) # Red for unknown
    system_status = "LOCKED"
    unlock_start_time = 0
    last_arduino_command = ""
    authorized_person_in_frame = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()
        img_h, img_w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(frame_rgb)
        
        face_detected = False
        authorized_person_in_frame = False
        
        # --- System auto-lock logic ---
        if system_status == "UNLOCKED" and current_time - unlock_start_time > UNLOCK_DURATION:
            system_status = "LOCKED"
            if ser and last_arduino_command != "R":
                ser.write(b'R')
                last_arduino_command = "R"
                print("🔴 Auto-lock: Timeout. Switching to RED light.")
        
        if mesh_results.multi_face_landmarks:
            face_detected = True
            # Since max_num_faces=1, we only get the most prominent face
            face_landmarks = mesh_results.multi_face_landmarks[0]

            # Get bounding box from landmarks
            lms_x = [lm.x * img_w for lm in face_landmarks.landmark]
            lms_y = [lm.y * img_h for lm in face_landmarks.landmark]
            x1, x2 = int(min(lms_x)), int(max(lms_x))
            y1, y2 = int(min(lms_y)), int(max(lms_y))

            # --- Blink Detection ---
            avg_ear = (eye_aspect_ratio(face_landmarks.landmark, img_w, img_h, is_right=False) +
                       eye_aspect_ratio(face_landmarks.landmark, img_w, img_h, is_right=True)) / 2.0

            if avg_ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                # --- Verification on eye open after blink ---
                if blink_counter >= BLINK_CONSEC_FRAMES:
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        # 1. Liveness Check
                        face_liveness_img = cv2.resize(face_roi, liveness_img_size)
                        face_liveness_img = face_liveness_img.astype("float") / 255.0
                        face_liveness_img = np.expand_dims(face_liveness_img, axis=0)
                        liveness_pred = liveness_model.predict(face_liveness_img, verbose=0)[0][0]
                        is_real = liveness_pred < 0.5

                        if is_real:
                            # 2. Face Recognition
                            face_recog = cv2.resize(apply_clahe_color(face_roi), img_size)
                            live_embedding = facenet_model.embeddings([face_recog])[0]
                            best_match_name, best_similarity = "Unknown", 0

                            for name, stored_embedding in known_embeddings.items():
                                similarity = 1 - dist.cosine(live_embedding, stored_embedding)
                                if similarity > best_similarity:
                                    best_similarity, best_match_name = similarity, name

                            if best_similarity > SIMILARITY_THRESHOLD:
                                status_text = f"✅ {best_match_name}"
                                status_color = (0, 255, 0) # Green
                                authorized_person_in_frame = True
                                
                                if system_status == "LOCKED":
                                    system_status = "UNLOCKED"
                                    unlock_start_time = current_time
                                    if ser and last_arduino_command != "G":
                                        ser.write(b'G')
                                        last_arduino_command = "G"
                                        print(f"🟢 UNLOCKED for {best_match_name}! Green light ON.")
                                else: # If already unlocked, just reset the timer
                                     unlock_start_time = current_time

                            else:
                                status_text = "❓ Unknown (Live)"
                                status_color = (0, 255, 255) # Yellow
                                authorized_person_in_frame = False
                        else:
                            status_text = "🚫 SPOOF DETECTED"
                            status_color = (0, 0, 255) # Red
                            authorized_person_in_frame = False
                
                # Reset blink counter after check
                blink_counter = 0

            # Draw bounding box and status text on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
            cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # If no face is detected in the frame
        if not face_detected:
            status_text = "❓ Unknown"
            status_color = (0, 0, 255)
            blink_counter = 0
            authorized_person_in_frame = False
            
        # If authorized person leaves the frame, but still in unlock period
        if not authorized_person_in_frame and system_status == "UNLOCKED":
            remaining_time = UNLOCK_DURATION - (current_time - unlock_start_time)
            if remaining_time <= 0:
                system_status = "LOCKED"
                if ser and last_arduino_command != "R":
                    ser.write(b'R')
                    last_arduino_command = "R"
                    print("🔴 No authorized person in view. Locking. RED light ON.")
        
        # --- Display overall system status on screen ---
        system_status_text = "🔴 LOCKED"
        system_color = (0, 0, 255)
        if system_status == "UNLOCKED":
            remaining = max(0, UNLOCK_DURATION - (current_time - unlock_start_time))
            system_status_text = f"🟢 UNLOCKED ({remaining:.1f}s)"
            system_color = (0, 255, 0)
            
        cv2.putText(frame, f"System: {system_status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, system_color, 2)
        cv2.imshow('SINGLE-FACE Secure Recognition', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o') and ser: # Manual off
            ser.write(b'O')
            last_arduino_command = "O"
            print("⚫ All lights turned OFF manually.")
            
    # --- Cleanup ---
    if ser:
        ser.write(b'O')
        time.sleep(0.5)
        ser.close()
        print("🔌 Arduino connection closed.")
    cap.release()
    cv2.destroyAllWindows()

def list_persons():
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder does not exist.")
        return []
    persons = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    if not persons:
        print("🤷 No persons found in the database.")
    else:
        print("\n📋 Persons in database:")
        for idx, name in enumerate(persons, 1):
            print(f"{idx}. {name}")
    return persons

def delete_person_data():
    persons = list_persons()
    if not persons: return
    try:
        choice_str = input("Enter the number of the person to delete (or 'c' to cancel): ").strip()
        if choice_str.lower() == 'c':
            print("Action canceled.")
            return
        choice = int(choice_str)
        if 1 <= choice <= len(persons):
            person_name = persons[choice - 1]
            confirm = input(f"❓ Are you sure you want to delete all data for '{person_name}'? (y/n): ").strip().lower()
            if confirm == 'y':
                shutil.rmtree(os.path.join(dataset_path, person_name))
                print(f"✅ Successfully deleted data for '{person_name}'.")
                print("⚠️ Please run option '2. Generate Face Embeddings' again.")
            else:
                print("Deletion canceled.")
        else:
            print("❌ Invalid number.")
    except (ValueError, IndexError):
        print("❌ Invalid input. Please enter a valid number.")

def test_arduino_manual():
    """ทดสอบ Arduino ด้วยตนเอง พร้อมแก้ไขปัญหาแบบสมบูรณ์"""
    print("\n🔧 Arduino Connection & LED Test")
    print("=" * 50)
    ser = test_arduino_connection(force_reset=False)
    if ser is None:
        print("\n🆘 Connection failed. Choose recovery option:")
        print("1. 🔄 Try advanced recovery")
        print("2. 📋 Manual troubleshooting guide")
        print("3. ❌ Exit")
        choice = input("Choose option (1-3): ").strip()
        if choice == '1':
            print("\n🛠️ Starting advanced recovery...")
            ser = test_arduino_connection(force_reset=True)
        elif choice == '2':
            print("\n📋 Manual Troubleshooting Guide:")
            print("="*40)
            print("1. Close Arduino IDE, Serial Monitor, PuTTY, etc.")
            print("2. Reset USB: Win+R -> devmgmt.msc -> Ports -> Disable/Enable USB-SERIAL CH340")
            print("3. Unplug and replug Arduino USB cable.")
            retry = input("\nTry connecting again? (y/n): ")
            if retry.lower() == 'y':
                return test_arduino_manual()
            else:
                return
        else:
            return

    if ser is None:
        print("❌ Could not establish Arduino connection")
        return

    print("\n🎮 Manual Arduino Control (R, G, B, O, T=Test, S=Status, Q=Quit)")
    while True:
        cmd = input("Enter command: ").strip().upper()
        if cmd == 'Q': break
        elif cmd == 'T':
            print("🎨 Running auto test...")
            for color, command in [("RED", b'R'), ("GREEN", b'G'), ("BLUE", b'B'), ("OFF", b'O')]:
                print(f"  Testing {color}")
                try:
                    ser.write(command)
                    time.sleep(1.5)
                except Exception as e:
                    print(f"   ❌ Failed to send {color}: {e}")
        elif cmd == 'S':
            print(f"📊 Status: Port={ser.port}, Baud={ser.baudrate}, Open={ser.is_open}")
        elif cmd in ['R', 'G', 'B', 'O']:
            try:
                ser.write(cmd.encode())
                print(f"✅ Sent: {cmd}")
            except Exception as e:
                print(f"❌ Failed to send command: {e}")
        else:
            print("❌ Invalid command!")
    try:
        ser.write(b'O')
        time.sleep(0.5)
        ser.close()
        print("🔌 Arduino connection closed.")
    except Exception as e:
        print(f"⚠️ Error closing connection: {e}")

def main():
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(liveness_dataset_path, 'real'), exist_ok=True)
    os.makedirs(os.path.join(liveness_dataset_path, 'spoof'), exist_ok=True)

    admin_status = "✅ Administrator" if is_admin() else "⚠️ User"
    print(f"🔐 Running with {admin_status} privileges")

    while True:
        print("\n" + "="*60)
        print("   🔐 Ultimate Secure Recognition System + Arduino 🔐")
        print("="*60)
        print("--- Face Identity ---")
        print("1. Capture new faces")
        print("2. Generate Face Embeddings")
        print("3. Show persons in database")
        print("4. Delete a person's data")
        print("\n--- Liveness Anti-Spoofing ---")
        print("5. Capture LIVENESS data (Real)")
        print("6. Capture LIVENESS data (Spoof)")
        print("7. TRAIN Liveness Model")
        print("\n--- System Control ---")
        print("8. 🚀 Start secure recognition (SINGLE PERSON)") # Modified
        print("9. 🔧 Test Arduino connection & LEDs")
        print("10. Exit")
        print(f"\n💡 Status: {admin_status} | Keys: 'q'=Quit recognition")

        choice = input("\nEnter your choice (1-10): ").strip()

        if choice == '1': capture_faces()
        elif choice == '2': generate_embeddings()
        elif choice == '3': list_persons()
        elif choice == '4': delete_person_data()
        elif choice == '5': capture_liveness_data('real')
        elif choice == '6': capture_liveness_data('spoof')
        elif choice == '7': train_liveness_model()
        elif choice == '8': recognize_faces()
        elif choice == '9': test_arduino_manual()
        elif choice == '10':
            print("👋 Exiting...")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
