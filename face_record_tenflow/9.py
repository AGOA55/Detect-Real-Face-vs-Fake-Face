import cv2
import os
import numpy as np
import pickle
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial import distance as dist
import mediapipe as mp
from mtcnn import MTCNN
from keras_facenet import FaceNet
import serial
import serial.tools.list_ports
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp_process
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

# --- JSON Serializer Helper ---
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- GPU Configuration ---
def setup_gpu():
    """ตั้งค่า GPU สำหรับ TensorFlow"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # เปิดใช้ memory growth เพื่อไม่ให้ GPU memory overflow
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # ตั้งค่า mixed precision เพื่เพิ่มประสิทธิภาพ
            tf.config.optimizer.set_jit(True)
            
            print(f"✅ Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            print("✅ GPU memory growth enabled")
            return True
        except RuntimeError as e:
            print(f"⚠ GPU setup error: {e}")
            return False
    else:
        print("⚠ No GPU found, using CPU only")
        return False

# เรียกใช้การตั้งค่า GPU
GPU_AVAILABLE = setup_gpu()

# --- Global Settings ---
dataset_path = 'dataset'
embeddings_path = 'face_embeddings.pkl'
model_info_path = 'model_info.json'
liveness_model_path = 'DeepPixWeights.hdf5'  # เพิ่มเส้นทางโมเดล liveness
camera_index = 0
img_size = (160, 160)  # FaceNet input
liveness_size = (224, 224)  # Liveness model input
confidence_threshold = 0.75  # FaceNet similarity
liveness_threshold = 0.5    # DeepPix threshold

# --- Initializing Models ---
print("🔄 Loading models...")
detector = MTCNN()  # CPU-based face detection
facenet_model = FaceNet()

# Load liveness detection model if available
liveness_model = None
if os.path.exists(liveness_model_path):
    try:
        print("📂 Loading Liveness model...")
        liveness_model = load_model(liveness_model_path)
        print("✅ Liveness Detection Model Loaded")
    except Exception as e:
        print(f"⚠ Failed to load liveness model: {e}")
        liveness_model = None
else:
    print("⚠ Liveness model not found. Using MediaPipe liveness detection only.")

# ตั้งค่า FaceNet ให้ใช้ GPU
if GPU_AVAILABLE:
    with tf.device('/GPU:0'):
        # Warm up the model
        dummy_input = np.random.random((1, 160, 160, 3)).astype(np.float32)
        _ = facenet_model.embeddings([dummy_input])
        print("✅ FaceNet Model Loaded on GPU")
else:
    print("✅ FaceNet Model Loaded on CPU")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=4, refine_landmarks=True, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye indices
EYE_INDICES_LEFT = [362, 385, 387, 263, 373, 380]
EYE_INDICES_RIGHT = [33, 160, 158, 133, 153, 144]

# --- Liveness Detection Functions ---
def check_liveness_deeppix(face_img):
    """ตรวจสอบ liveness ด้วย DeepPix model"""
    if liveness_model is None:
        return True, 1.0  # fallback to True if no model
    
    try:
        face_resized = cv2.resize(face_img, liveness_size)
        face_resized = face_resized.astype("float32") / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        pred = liveness_model.predict(face_resized, verbose=0)[0][0]
        return pred > liveness_threshold, pred
    except Exception as e:
        print(f"DeepPix error: {e}")
        return True, 1.0

def check_liveness_combined(face_img, mediapipe_verified=False, ear_score=0.0):
    """Combined liveness detection using both DeepPix and MediaPipe"""
    # DeepPix liveness
    deeppix_real, deeppix_score = check_liveness_deeppix(face_img)
    
    # MediaPipe liveness (blink + movement)
    mediapipe_weight = 0.6 if mediapipe_verified else 0.2
    deeppix_weight = 0.7 if liveness_model else 0.4
    
    # Combined score
    combined_score = (deeppix_score * deeppix_weight) + (mediapipe_weight if mediapipe_verified else 0.0)
    
    # Final decision
    is_real = (deeppix_real and deeppix_score > liveness_threshold) or mediapipe_verified
    
    return is_real, combined_score, deeppix_score

# --- Advanced Image Augmentation ---
def advanced_augmentation(image):
    """การ augment รูปภาพแบบละเอียด"""
    augmented_images = [image]  # ภาพต้นฉบับ
    
    # 1. Brightness adjustment
    bright_img = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    dark_img = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
    augmented_images.extend([bright_img, dark_img])
    
    # 2. Contrast adjustment
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # High contrast
    clahe_high = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_high = clahe_high.apply(l)
    high_contrast = cv2.merge([l_high, a, b])
    high_contrast = cv2.cvtColor(high_contrast, cv2.COLOR_LAB2BGR)
    
    # Low contrast
    clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    l_low = clahe_low.apply(l)
    low_contrast = cv2.merge([l_low, a, b])
    low_contrast = cv2.cvtColor(low_contrast, cv2.COLOR_LAB2BGR)
    
    augmented_images.extend([high_contrast, low_contrast])
    
    # 3. Gaussian noise
    noise = np.random.randint(0, 25, image.shape, dtype=np.uint8)
    noisy_img = cv2.add(image, noise)
    augmented_images.append(noisy_img)
    
    # 4. Gaussian blur
    blurred_img = cv2.GaussianBlur(image, (3,3), 1.0)
    augmented_images.append(blurred_img)
    
    # 5. Histogram equalization
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    eq_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    augmented_images.append(eq_img)
    
    return augmented_images

def quality_check_image(image):
    """ตรวจสอบคุณภาพของรูปภาพ"""
    if image is None or image.size == 0:
        return False, "Empty image"
    
    # ตรวจสอบความสว่าง
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    if brightness < 30:
        return False, "Too dark"
    elif brightness > 220:
        return False, "Too bright"
    
    # ตรวจสอบความคมชัด (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return False, "Too blurry"
    
    return True, "Good quality"

# --- Face Tracker Class ---
class FaceTracker:
    """ติดตามหน้าแต่ละคนด้วย geometric matching"""
    def __init__(self, max_faces=4, max_disappeared=10):
        self.next_id = 0
        self.faces = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_faces = max_faces
    
    def register(self, centroid):
        """ลงทะเบียนหน้าใหม่"""
        self.faces[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, face_id):
        """ยกเลิกการลงทะเบียน"""
        del self.faces[face_id]
        del self.disappeared[face_id]
    
    def update(self, face_boxes):
        """อัพเดทตำแหน่งหน้า"""
        if len(face_boxes) == 0:
            # ไม่มีหน้า - เพิ่มการหายไป
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
            return {}
        
        # คำนวณ centroid ของหน้าแต่ละหน้า
        input_centroids = []
        for (x, y, w, h) in face_boxes:
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))
        
        # ถ้าไม่มีหน้าที่ติดตาม - ลงทะเบียนใหม่ทั้งหมด
        if len(self.faces) == 0:
            face_assignments = {}
            for i, centroid in enumerate(input_centroids):
                face_id = self.register(centroid)
                face_assignments[face_id] = i
            return face_assignments
        
        # คำนวณระยะห่างระหว่างหน้าเก่าและใหม่
        face_centroids = list(self.faces.values())
        face_ids = list(self.faces.keys())
        
        # Hungarian algorithm แบบง่าย
        D = dist.cdist(np.array(face_centroids), np.array(input_centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_row_indices = set()
        used_col_indices = set()
        face_assignments = {}
        
        # จับคู่หน้าที่มีอยู่
        for (row, col) in zip(rows, cols):
            if row in used_row_indices or col in used_col_indices:
                continue
            
            # ถ้าระยะห่างมากเกินไป ถือว่าเป็นหน้าใหม่
            if D[row, col] > 100:  # threshold สำหรับการติดตาม
                continue
            
            face_id = face_ids[row]
            self.faces[face_id] = input_centroids[col]
            self.disappeared[face_id] = 0
            face_assignments[face_id] = col
            
            used_row_indices.add(row)
            used_col_indices.add(col)
        
        # จัดการหน้าที่ไม่ได้จับคู่
        unused_rows = set(range(0, D.shape[0])).difference(used_row_indices)
        unused_cols = set(range(0, D.shape[1])).difference(used_col_indices)
        
        # เพิ่มการหายไปสำหรับหน้าที่ไม่ได้จับคู่
        if D.shape[0] >= D.shape[1]:
            for row in unused_rows:
                face_id = face_ids[row]
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
        
        # ลงทะเบียนหน้าใหม่
        else:
            for col in unused_cols:
                if len(self.faces) < self.max_faces:
                    face_id = self.register(input_centroids[col])
                    face_assignments[face_id] = col
        
        return face_assignments

# --- Background Recognition Worker with Improved Tracking ---
class BackgroundRecognitionWorker:
    """Background worker สำหรับ GPU recognition พร้อม confidence checking"""
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.input_queue = queue.Queue(maxsize=3)
        self.output_queue = queue.Queue(maxsize=3)
        self.running = False
        self.thread = None
        self.similarity_threshold = 0.75  # เพิ่ม threshold
        self.confidence_threshold = 0.8   # threshold สำหรับความเชื่อมั่น
        
        # สร้าง FaceNet model สำหรับ thread นี้
        if GPU_AVAILABLE:
            with tf.device('/GPU:0'):
                self.facenet_model = FaceNet()
        else:
            self.facenet_model = FaceNet()
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _worker(self):
        while self.running:
            try:
                data = self.input_queue.get(timeout=0.1)
                face_rois, face_track_ids = data
                
                results = {}
                if face_rois:
                    # GPU processing
                    if GPU_AVAILABLE:
                        with tf.device('/GPU:0'):
                            batch_embeddings = self.facenet_model.embeddings(face_rois)
                    else:
                        batch_embeddings = self.facenet_model.embeddings(face_rois)
                    
                    for embedding, track_id in zip(batch_embeddings, face_track_ids):
                        best_match_name = "Unknown"
                        best_similarity = 0
                        second_best_similarity = 0
                        
                        # หาค่า similarity ที่ดีที่สุดและรองลงมา
                        similarities = []
                        for name, stored_embedding in self.embeddings.items():
                            similarity = 1 - dist.cosine(embedding, stored_embedding)
                            similarities.append((similarity, name))
                        
                        # เรียง similarity จากมากไปน้อย
                        similarities.sort(reverse=True)
                        
                        if len(similarities) > 0:
                            best_similarity, best_match_name = similarities[0]
                            if len(similarities) > 1:
                                second_best_similarity = similarities[1][0]
                        
                        # เช็คหลายเงื่อนไข
                        confidence_gap = best_similarity - second_best_similarity
                        is_confident = (best_similarity > self.similarity_threshold and 
                                      confidence_gap > 0.1)  # ต้องห่างจากอันดับ 2 อย่างน้อย 0.1
                        
                        if is_confident and best_similarity > self.confidence_threshold:
                            status = "Pass"
                        else:
                            status = "Unknown"
                            best_match_name = "Unknown"
                            # ถ้า confidence ต่ำ ให้แสดง similarity ที่แท้จริง
                            if best_similarity > 0.7:  # มี similarity บ้างแต่ไม่พอ
                                best_match_name = f"Low_Conf_{best_similarity:.2f}"
                        
                        results[track_id] = {
                            'name': best_match_name,
                            'confidence': best_similarity,
                            'status': status,
                            'confidence_gap': confidence_gap
                        }
                
                self.output_queue.put(results)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Recognition error: {e}")
                self.output_queue.put({})
    
    def process_async(self, face_rois, face_track_ids):
        try:
            # ล้าง queue เก่าถ้าเต็ม
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.input_queue.put_nowait((face_rois, face_track_ids))
            return True
        except queue.Full:
            return False
    
    def get_result(self):
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return {}

# --- Helper Functions ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def match_face_to_landmark(face_box, landmarks_list, frame_shape):
    """จับคู่หน้าจาก MTCNN กับ landmarks จาก MediaPipe"""
    h, w = frame_shape[:2]
    face_center = np.array([face_box[0] + face_box[2]/2, face_box[1] + face_box[3]/2])
    
    best_match_idx = -1
    min_distance = float('inf')
    
    for idx, face_landmarks in enumerate(landmarks_list):
        nose_tip = np.array([face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h])
        distance = np.linalg.norm(face_center - nose_tip)
        
        if distance < min_distance:
            min_distance = distance
            best_match_idx = idx
    
    if min_distance > 100:
        return -1
    
    return best_match_idx

def apply_clahe_color(img):
    """Image preprocessing with CPU optimization"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl,a,b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    img_blur = cv2.GaussianBlur(img_clahe, (3,3), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_3ch = cv2.merge([edges, edges, edges])
    edges_3ch = cv2.normalize(edges_3ch, None, 0, 255, cv2.NORM_MINMAX)
    final_img = cv2.addWeighted(img_blur, 0.7, edges_3ch, 0.3, 0)
    
    return final_img

# --- 1. Enhanced Capture Faces (50 images per angle) ---
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
    
    # เพิ่มจำนวนรูปต่อมุมเป็น 50
    angles = ["Center", "Left", "Right", "Up", "Down"]
    samples_per_angle = 50  # เพิ่มจาก 30 เป็น 50
    
    # สถิติการเก็บข้อมูล
    total_captured = 0
    rejected_count = 0
    
    for angle in angles:
        print(f"\n📸 Look {angle}. Press 's' to skip angle, 'q' to quit.")
        print(f"Target: {samples_per_angle} high-quality images")
        count = 0
        frame_count = 0
        
        while count < samples_per_angle:
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1

            faces = detector.detect_faces(frame)
            if len(faces) > 0:
                x, y, w, h = faces[0]['box']
                x1, y1 = abs(x), abs(y)
                x2, y2 = x1 + w, y1 + h
                face_img = frame[y1:y2, x1:x2]

                if face_img.size > 0:
                    # ตรวจสอบคุณภาพรูปภาพ
                    is_good_quality, quality_msg = quality_check_image(face_img)
                    
                    # เก็บรูปเฉพาะเมื่อคุณภาพดีและทุก 3 เฟรม
                    if is_good_quality and frame_count % 3 == 0:
                        face_img_resized = cv2.resize(face_img, img_size)
                        angle_folder = os.path.join(save_path, angle)
                        if not os.path.exists(angle_folder):
                            os.makedirs(angle_folder)
                        
                        filename = f"{count+1:03d}.jpg"  # เปลี่ยนเป็น 3 หลัก
                        cv2.imwrite(os.path.join(angle_folder, filename), face_img_resized)
                        count += 1
                        total_captured += 1
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"{angle}: {count}/{samples_per_angle} ✓"
                    else:
                        rejected_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        text = f"{angle}: {count}/{samples_per_angle} - {quality_msg}"
                    
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # แสดงสถิติ
                    cv2.putText(frame, f"Captured: {total_captured} | Rejected: {rejected_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Quality: {quality_msg}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "No face detected - Please position your face", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, f"Captured: {total_captured} | Rejected: {rejected_count}", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Enhanced Face Capture', frame)
            
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):
                print(f"Skipping {angle} angle (captured {count} images)")
                break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Finished capturing faces for '{person_name}'")
    print(f"📊 Total captured: {total_captured} images")
    print(f"📊 Total rejected: {rejected_count} images")
    print(f"📊 Quality ratio: {total_captured/(total_captured+rejected_count)*100:.1f}%")

# --- 2. Advanced Generate Embeddings ---
def generate_embeddings():
    print("\n🔄 Advanced Face Embeddings Generation...")
    print("=" * 60)
    
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ Dataset folder is empty. Please capture faces first.")
        return

    # เก็บสถิติ
    stats = {
        'total_images': 0,
        'processed_images': 0,
        'augmented_images': 0,
        'persons': 0,
        'processing_time': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    start_time = time.time()
    embeddings = {}
    all_images = []
    all_labels = []
    person_image_counts = {}
    
    print("📁 Scanning dataset...")
    
    # นับจำนวนรูปทั้งหมดก่อน
    for folder_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(person_path): continue
        
        person_image_counts[folder_name] = 0
        stats['persons'] += 1
        
        for angle_folder in os.listdir(person_path):
            angle_path = os.path.join(person_path, angle_folder)
            if not os.path.isdir(angle_path): continue
            
            image_files = [f for f in os.listdir(angle_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            person_image_counts[folder_name] += len(image_files)
            stats['total_images'] += len(image_files)
    
    print(f"📊 Found {stats['persons']} persons with {stats['total_images']} total images")
    print("📊 Images per person:")
    for person, count in person_image_counts.items():
        print(f"   • {person}: {count} images")
    
    print("\n🔄 Processing images with advanced augmentation...")
    
    # ประมวลผลรูปภาพแต่ละคน
    for folder_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(person_path): continue

        person_images = []
        person_processed = 0
        
        print(f"\n👤 Processing {folder_name}...")
        
        for angle_folder in os.listdir(person_path):
            angle_path = os.path.join(person_path, angle_folder)
            if not os.path.isdir(angle_path): continue
            
            print(f"   📐 Processing {angle_folder} angle...")
            angle_processed = 0

            for filename in os.listdir(angle_path):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(angle_path, filename)
                image = cv2.imread(img_path)
                if image is None: continue

                # ตรวจสอบคุณภาพรูปภาพ
                is_good_quality, quality_msg = quality_check_image(image)
                if not is_good_quality:
                    print(f"      ⚠ Skipped {filename}: {quality_msg}")
                    continue

                # Apply advanced preprocessing
                image_clahe = apply_clahe_color(image)
                
                # Advanced augmentation (เฉพาะรูปที่คุณภาพดี)
                augmented_images = advanced_augmentation(image_clahe)
                
                for aug_img in augmented_images:
                    image_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                    all_images.append(image_rgb)
                    all_labels.append(folder_name)
                    person_images.append(image_rgb)
                
                angle_processed += 1
                person_processed += 1
                stats['processed_images'] += 1
                stats['augmented_images'] += len(augmented_images)
                
                # แสดงความคืบหน้า
                if angle_processed % 10 == 0:
                    print(f"      ✓ Processed {angle_processed} images from {angle_folder}")
        
        print(f"   ✅ {folder_name}: {person_processed} original images → {len(person_images)} total samples")

    if not all_images:
        print("❌ No valid images found to generate embeddings.")
        return

    print(f"\n🚀 Generating embeddings for {len(all_images)} total samples...")
    print(f"📊 Augmentation ratio: {stats['augmented_images']/stats['processed_images']:.1f}x")
    
    # Batch processing with progress tracking
    batch_size = 32 if GPU_AVAILABLE else 8
    all_embeddings = []
    
    if GPU_AVAILABLE:
        with tf.device('/GPU:0'):
            for i in range(0, len(all_images), batch_size):
                batch = all_images[i:i+batch_size]
                batch_embeddings = facenet_model.embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                
                progress = min(i+batch_size, len(all_images))
                percentage = (progress / len(all_images)) * 100
                print(f"   🔄 GPU Processing: {progress}/{len(all_images)} ({percentage:.1f}%)")
    else:
        for i, image in enumerate(all_images):
            embedding = facenet_model.embeddings([image])[0]
            all_embeddings.append(embedding)
            
            if (i + 1) % 50 == 0:
                percentage = ((i + 1) / len(all_images)) * 100
                print(f"   🔄 CPU Processing: {i+1}/{len(all_images)} ({percentage:.1f}%)")

    print("\n📊 Computing person embeddings with quality analysis...")
    
    # Group embeddings by person และวิเคราะห์คุณภาพ
    person_embeddings = {}
    person_stats = {}
    
    for embedding, label in zip(all_embeddings, all_labels):
        if label not in person_embeddings:
            person_embeddings[label] = []
            person_stats[label] = {'count': 0, 'avg_norm': 0.0, 'std_norm': 0.0}
        
        # Normalize embedding
        normalized_embedding = embedding / np.linalg.norm(embedding)
        person_embeddings[label].append(normalized_embedding)
        person_stats[label]['count'] += 1

    # คำนวณ final embeddings และสถิติ
    for person, emb_list in person_embeddings.items():
        embeddings[person] = np.mean(emb_list, axis=0)
        embeddings[person] /= np.linalg.norm(embeddings[person])
        
        # คำนวณสถิติ และแปลงเป็น Python native types
        norms = [np.linalg.norm(emb) for emb in emb_list]
        person_stats[person]['avg_norm'] = float(np.mean(norms))  # แปลงเป็น float
        person_stats[person]['std_norm'] = float(np.std(norms))   # แปลงเป็น float
        
        print(f"   👤 {person}: {person_stats[person]['count']} samples, "
              f"norm={person_stats[person]['avg_norm']:.3f}±{person_stats[person]['std_norm']:.3f}")

    # Cross-person similarity analysis
    print("\n🔍 Cross-person similarity analysis:")
    person_names = list(embeddings.keys())
    similarity_matrix = {}
    
    if len(person_names) > 1:
        for i in range(len(person_names)):
            for j in range(i+1, len(person_names)):
                similarity = float(1 - dist.cosine(embeddings[person_names[i]], embeddings[person_names[j]]))  # แปลงเป็น float
                similarity_key = f"{person_names[i]}_vs_{person_names[j]}"
                similarity_matrix[similarity_key] = similarity
                
                print(f"   📏 {person_names[i]} ↔ {person_names[j]}: {similarity:.3f}")
                
                if similarity > 0.7:
                    print(f"      ⚠ High similarity detected! May cause confusion.")

    # บันทึกข้อมูล
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)

    # บันทึกสถิติและข้อมูลโมเดล (แก้ไข JSON serialization)
    stats['processing_time'] = float(time.time() - start_time)  # แปลงเป็น float
    stats['person_stats'] = person_stats
    stats['similarity_matrix'] = similarity_matrix
    
    model_info = {
        'version': '2.0',
        'model_type': 'FaceNet + Advanced Augmentation + Liveness',
        'stats': stats,
        'settings': {
            'similarity_threshold': 0.75,
            'confidence_threshold': 0.82,
            'liveness_threshold': liveness_threshold,
            'augmentation_ratio': float(stats['augmented_images']/stats['processed_images']),  # แปลงเป็น float
            'gpu_enabled': GPU_AVAILABLE,
            'liveness_model_available': liveness_model is not None
        }
    }
    
    # ใช้ NumpyEncoder เพื่อจัดการ numpy types
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2, cls=NumpyEncoder)

    print("\n" + "="*60)
    print("✅ Advanced embeddings generation completed!")
    print(f"📊 Processing time: {stats['processing_time']:.1f} seconds")
    print(f"📊 Processing speed: {len(all_images)/stats['processing_time']:.1f} images/second")
    print(f"📁 Embeddings saved to: {embeddings_path}")
    print(f"📁 Model info saved to: {model_info_path}")

def find_arduino_port(baudrate=9600, timeout=2):
    keywords = ["arduino", "ch340", "usb serial", "usb-serial", "ch9102", "silicon labs"]
    ports = serial.tools.list_ports.comports()

    for port in ports:
        desc = port.description.lower()
        if any(keyword in desc for keyword in keywords):
            try:
                ser = serial.Serial(port.device, baudrate, timeout=1)
                ser.close()
                return port.device
            except Exception:
                pass

    for port in ports:
        try:
            ser = serial.Serial(port.device, baudrate, timeout=1)
            ser.close()
            return port.device
        except Exception:
            pass

    return None

# --- 3. Improved Face Recognition with Liveness Detection ---
def recognize_faces():
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
            
        # โหลดข้อมูลโมเดล
        model_info = {}
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                print(f"📊 Loaded model: {model_info.get('model_type', 'Unknown')}")
                print(f"📊 Dataset: {model_info.get('stats', {}).get('persons', 0)} persons, "
                      f"{model_info.get('stats', {}).get('total_images', 0)} images")
                print(f"📊 Liveness: {'DeepPix + MediaPipe' if liveness_model else 'MediaPipe Only'}")
    except FileNotFoundError:
        print("❌ Embeddings file not found. Please generate embeddings first.")
        return

    # Arduino setup
    arduino_port = find_arduino_port()
    if arduino_port:
        try:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            print(f"🔌 Connected to Arduino on {arduino_port}")
        except Exception as e:
            print(f"❌ Could not open serial port: {e}")
            ser = None
    else:
        print("⚠ No Arduino found.")
        ser = None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    # Initialize trackers
    face_tracker = FaceTracker(max_faces=4, max_disappeared=20)
    recognition_worker = BackgroundRecognitionWorker(embeddings)
    recognition_worker.start()

    print(f"\n🔄 Advanced recognition with liveness detection started.")
    print(f"🔧 Processing: {'GPU+CPU' if GPU_AVAILABLE else 'CPU Only'}")
    print(f"🔧 Liveness: {'DeepPix + MediaPipe' if liveness_model else 'MediaPipe Only'}")
    
    # State tracking ใช้ track_id แทน face_idx
    EAR_THRESHOLD, EAR_CONSEC_FRAMES = 0.2, 3
    MOVE_THRESHOLD = 7
    
    person_states = {}
    recognition_cache = {}
    liveness_cache = {}  # เพิ่ม cache สำหรับ liveness
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            h, w, _ = frame.shape
            
            # --- Synchronous Processing ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = face_mesh.process(image_rgb)
            faces = detector.detect_faces(frame)

            if not faces:
                # ไม่มีหน้า - ล้างการติดตาม
                face_assignments = face_tracker.update([])
                cv2.putText(frame, "No faces detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if ser:
                    ser.write("No\n".encode())
                
                cv2.imshow('Advanced Face Recognition + Liveness', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # อัพเดท face tracking
            face_boxes = [face['box'] for face in faces]
            face_assignments = face_tracker.update(face_boxes)

            # เตรียมข้อมูลสำหรับ background recognition
            if frame_count % 5 == 0 and face_assignments:
                face_rois = []
                face_track_ids = []
                
                for track_id, face_idx in face_assignments.items():
                    if face_idx < len(faces):
                        x_min, y_min, w_mtcnn, h_mtcnn = faces[face_idx]['box']
                        x_max, y_max = x_min + w_mtcnn, y_min + h_mtcnn
                        
                        face_roi = frame[y_min:y_max, x_min:x_max]
                        if face_roi.size > 0:
                            face_roi_processed = apply_clahe_color(face_roi)
                            face_roi_resized = cv2.resize(face_roi_processed, img_size)
                            face_roi_rgb = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2RGB)
                            
                            face_rois.append(face_roi_rgb)
                            face_track_ids.append(track_id)
                
                if face_rois:
                    recognition_worker.process_async(face_rois, face_track_ids)

            # รับผลลัพธ์จาก background worker
            new_results = recognition_worker.get_result()
            if new_results:
                # อัพเดท cache แต่เฉพาะ track_id ที่ยังมีอยู่
                valid_track_ids = set(face_assignments.keys())
                # ลบ cache ของ track_id ที่หายไป
                recognition_cache = {tid: result for tid, result in recognition_cache.items() 
                                   if tid in valid_track_ids}
                liveness_cache = {tid: result for tid, result in liveness_cache.items() 
                                if tid in valid_track_ids}
                # อัพเดท cache ใหม่
                recognition_cache.update(new_results)

            # Process และแสดงผล
            current_detections = []
            
            if mp_results and mp_results.multi_face_landmarks:
                for track_id, face_idx in face_assignments.items():
                    if face_idx >= len(faces):
                        continue
                        
                    x_min, y_min, w_mtcnn, h_mtcnn = faces[face_idx]['box']
                    x_max, y_max = x_min + w_mtcnn, y_min + h_mtcnn
                    face_roi = frame[y_min:y_max, x_min:x_max]
                    
                    # จับคู่ landmarks
                    landmark_idx = match_face_to_landmark(faces[face_idx]['box'], mp_results.multi_face_landmarks, (h, w))
                    
                    # สร้าง state สำหรับ track_id ใหม่
                    if track_id not in person_states:
                        person_states[track_id] = {
                            'blink_counter': 0,
                            'blink_verified': False,
                            'movement_verified': False,
                            'prev_nose_pos': None,
                            'verified': False,
                            'mediapipe_liveness': False
                        }
                    
                    # Liveness Detection
                    if face_roi.size > 0 and frame_count % 10 == 0:  # ทุก 10 เฟรมสำหรับ liveness
                        mediapipe_verified = person_states[track_id]['verified']
                        is_real, combined_score, deeppix_score = check_liveness_combined(
                            face_roi, mediapipe_verified, 0.0)
                        
                        liveness_cache[track_id] = {
                            'is_real': is_real,
                            'combined_score': combined_score,
                            'deeppix_score': deeppix_score,
                            'mediapipe_verified': mediapipe_verified
                        }
                    
                    if landmark_idx >= 0:
                        face_landmarks = mp_results.multi_face_landmarks[landmark_idx]
                        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=int)
                        
                        # MediaPipe Liveness detection
                        left_eye = np.array([landmarks[i] for i in EYE_INDICES_LEFT])
                        right_eye = np.array([landmarks[i] for i in EYE_INDICES_RIGHT])
                        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                        
                        # Blink detection
                        if ear < EAR_THRESHOLD:
                            person_states[track_id]['blink_counter'] += 1
                        else:
                            if person_states[track_id]['blink_counter'] >= EAR_CONSEC_FRAMES:
                                person_states[track_id]['blink_verified'] = True
                            person_states[track_id]['blink_counter'] = 0
                        
                        # Movement detection
                        nose_tip = landmarks[1]
                        if person_states[track_id]['prev_nose_pos'] is not None:
                            move_dist = np.linalg.norm(np.array(nose_tip) - np.array(person_states[track_id]['prev_nose_pos']))
                            if move_dist > MOVE_THRESHOLD:
                                person_states[track_id]['movement_verified'] = True
                        person_states[track_id]['prev_nose_pos'] = nose_tip
                        
                        # MediaPipe Verification
                        person_states[track_id]['verified'] = (person_states[track_id]['blink_verified'] and 
                                                               person_states[track_id]['movement_verified'])
                        person_states[track_id]['mediapipe_liveness'] = person_states[track_id]['verified']
                        
                        # Combined Display Logic
                        display_text = "Blink & Move to Verify"
                        display_color = (0, 165, 255)  # Orange for verification
                        status = "No"
                        
                        # Get liveness result
                        liveness_result = liveness_cache.get(track_id, {
                            'is_real': False, 'combined_score': 0.0, 'deeppix_score': 0.0, 'mediapipe_verified': False
                        })
                        
                        if person_states[track_id]['verified'] and track_id in recognition_cache:
                            recognition_result = recognition_cache[track_id]
                            
                            # Combine recognition and liveness results
                            is_live = liveness_result['is_real']
                            live_score = liveness_result['combined_score']
                            
                            if recognition_result['status'] == "Pass" and is_live:
                                display_color = (0, 255, 0)  # Green for pass
                                status = "Pass"
                                display_text = f"{recognition_result['name']} ✓Live ({live_score:.2f})"
                            elif recognition_result['status'] == "Pass" and not is_live:
                                display_color = (0, 0, 255)  # Red for fake
                                status = "Fake"
                                display_text = f"{recognition_result['name']} ✗Fake ({live_score:.2f})"
                            elif not is_live:
                                display_color = (0, 0, 255)  # Red for fake
                                status = "Fake"
                                display_text = f"Fake Detected ({live_score:.2f})"
                            else:
                                display_color = (0, 0, 255)  # Blue for unknown but live
                                status = "Unknown"
                                display_text = f"Unknown ✓Live ({live_score:.2f})"
                        
                        current_detections.append({
                            'box': (x_min, y_min, x_max, y_max),
                            'text': display_text,
                            'color': display_color,
                            'status': status,
                            'track_id': track_id,
                            'liveness_info': liveness_result
                        })

            # แสดงผล
            for detection in current_detections:
                x_min, y_min, x_max, y_max = detection['box']
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), detection['color'], 2)
                cv2.putText(frame, detection['text'], (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection['color'], 2)
                cv2.putText(frame, f"ID:{detection['track_id']}", (x_min, y_max + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection['color'], 1)
                
                # แสดงข้อมูล liveness เพิ่มเติม
                liveness_info = detection['liveness_info']
                if liveness_model:
                    liveness_detail = f"DeepPix:{liveness_info['deeppix_score']:.2f} MP:{liveness_info['mediapipe_verified']}"
                    cv2.putText(frame, liveness_detail, (x_min, y_max + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, detection['color'], 1)

            # Send to Arduino
            if ser:
                status_list = [d['status'] for d in current_detections] or ["No"]
                status_msg = ",".join(status_list) + "\n"
                ser.write(status_msg.encode())

            # Performance info
            gpu_status = "GPU+CPU" if GPU_AVAILABLE else "CPU Only"
            liveness_status = "DeepPix+MP" if liveness_model else "MP Only"
            cv2.putText(frame, f"Mode: {gpu_status} + {liveness_status} | Faces: {len(current_detections)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Active Tracks: {len(face_tracker.faces)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Advanced Face Recognition + Liveness', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        print("🛑 Stopping recognition...")
        recognition_worker.stop()
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        if ser:
            ser.close()

# --- ฟังก์ชันแสดงรายชื่อคนในฐานข้อมูล ---
def list_persons():
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder does not exist.")
        return []
    
    persons = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    if not persons:
        print("❌ No persons found in the dataset.")
        return []
    
    print("\n📋 Persons in database:")
    total_images = 0
    
    for idx, name in enumerate(persons, 1):
        person_path = os.path.join(dataset_path, name)
        image_count = 0
        
        # นับจำนวนรูปในแต่ละมุม
        for angle_folder in os.listdir(person_path):
            angle_path = os.path.join(person_path, angle_folder)
            if os.path.isdir(angle_path):
                image_files = [f for f in os.listdir(angle_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                image_count += len(image_files)
        
        total_images += image_count
        print(f"{idx}. {name} ({image_count} images)")
    
    print(f"\n📊 Total: {len(persons)} persons, {total_images} images")
    
    # แสดงข้อมูลโมเดลถ้ามี
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
            print(f"📊 Model: {model_info.get('model_type', 'Unknown')}")
            print(f"📊 Last training: {model_info.get('stats', {}).get('timestamp', 'Unknown')}")
            print(f"📊 Liveness: {'Available' if model_info.get('settings', {}).get('liveness_model_available', False) else 'MediaPipe Only'}")
    
    return persons

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

def main():
    if not os.path.exists(dataset_path): 
        os.makedirs(dataset_path)
    
    while True:
        print("\n" + "="*60 + "\n   Advanced Face Recognition + Liveness System v2.0\n" + "="*60)
        print(f"🔧 Hardware: {'GPU+CPU' if GPU_AVAILABLE else 'CPU Only'}")
        print(f"🔧 Liveness: {'DeepPix + MediaPipe' if liveness_model else 'MediaPipe Only'}")
        print("1. Capture new faces (50 images per angle)")
        print("2. Generate Advanced Embeddings (GPU + Augmentation)")
        print("3. Start recognition with liveness detection")
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
