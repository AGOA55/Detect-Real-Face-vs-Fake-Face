import cv2
import os
import numpy as np
import pickle
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import time

# --- Global Settings ---
dataset_path = r'E:\Project By Tawan\dectect face\face_record_tenflow\data02' # เปลี่ยนตำแหน่ง path ที่ต้องการ
camera_index = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_size = (100, 100)

# Global threshold variables (เข้มงวดขึ้น)
CONFIDENCE_THRESHOLD = 88.0
MIN_CONFIDENCE_GAP = 15.0
MIN_STABILITY_FRAMES = 3

class FaceTracker:
    """คลาสสำหรับติดตามใบหน้าแต่ละคน"""
    def __init__(self, face_id, initial_bbox, max_missing_frames=10):
        self.face_id = face_id
        self.bbox = initial_bbox  # (x, y, w, h)
        self.center = self._get_center(initial_bbox)
        self.prediction_buffer = []
        self.missing_frames = 0
        self.max_missing_frames = max_missing_frames
        self.last_seen = time.time()
        self.is_stable = False
        self.confirmed_identity = None
        self.confidence_history = []
        
    def _get_center(self, bbox):
        x, y, w, h = bbox
        return (x + w//2, y + h//2)
    
    def update(self, new_bbox):
        """อัพเดทตำแหน่งใบหน้า"""
        self.bbox = new_bbox
        self.center = self._get_center(new_bbox)
        self.missing_frames = 0
        self.last_seen = time.time()
    
    def add_prediction(self, label_idx, confidence, confidence_gap):
        """เพิ่มการทำนายใหม่"""
        self.prediction_buffer.append({
            'label': label_idx,
            'confidence': confidence,
            'gap': confidence_gap,
            'timestamp': time.time()
        })
        
        # เก็บแค่ 10 ครั้งล่าสุด
        if len(self.prediction_buffer) > 10:
            self.prediction_buffer.pop(0)
        
        # ตรวจสอบความเสถียร
        self._check_stability()
    
    def _check_stability(self):
        """ตรวจสอบความเสถียรของการทำนาย"""
        if len(self.prediction_buffer) < MIN_STABILITY_FRAMES:
            return
        
        recent_predictions = self.prediction_buffer[-MIN_STABILITY_FRAMES:]
        
        # ตรวจสอบว่าการทำนายล่าสุดสอดคล้องกันหรือไม่
        labels = [p['label'] for p in recent_predictions]
        confidences = [p['confidence'] for p in recent_predictions]
        gaps = [p['gap'] for p in recent_predictions]
        
        # นับความถี่ของแต่ละ label
        from collections import Counter
        label_counts = Counter(labels)
        most_common_label, count = label_counts.most_common(1)[0]
        
        # ตรวจสอบเงื่อนไขความเสถียร
        stability_ratio = count / len(recent_predictions)
        avg_confidence = np.mean([c for i, c in enumerate(confidences) if labels[i] == most_common_label])
        avg_gap = np.mean([g for i, g in enumerate(gaps) if labels[i] == most_common_label])
        
        # เงื่อนไขสำหรับการยืนยันตัวตน
        self.is_stable = (
            stability_ratio >= 0.7 and  # อย่างน้อย 70% ต้องเป็น label เดียวกัน
            avg_confidence >= CONFIDENCE_THRESHOLD and
            avg_gap >= MIN_CONFIDENCE_GAP
        )
        
        if self.is_stable:
            self.confirmed_identity = most_common_label
            self.confidence_history = confidences[-3:]
        else:
            self.confirmed_identity = None
    
    def get_display_info(self, le):
        """ได้ข้อมูลสำหรับแสดงผล"""
        if not self.is_stable or self.confirmed_identity is None:
            return "Unknown", (0, 0, 255), 0.0
        
        name = le.inverse_transform([self.confirmed_identity])[0]
        avg_confidence = np.mean(self.confidence_history)
        color = (0, 255, 0)
        
        return name, color, avg_confidence
    
    def distance_to(self, bbox):
        """คำนวณระยะห่างจากใบหน้าใหม่"""
        new_center = self._get_center(bbox)
        return np.sqrt((self.center[0] - new_center[0])**2 + (self.center[1] - new_center[1])**2)

def is_good_quality_image(image, min_variance=80):
    """ตรวจสอบคุณภาพภาพ"""
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance > min_variance

def capture_faces():
    person_name = input("Enter the person's name (no spaces): ").strip()
    if not person_name:
        print("❌ Invalid name.")
        return

    save_path = os.path.join(dataset_path, person_name)
    
    try:
        os.makedirs(save_path, exist_ok=True)
        print(f"📁 Directory ready: {save_path}")
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print(f"\n📷 Capturing faces for '{person_name}'...")
    print("💡 Tips: Move your head slightly, change expressions, different angles")
    print("Press 'q' to quit early")
    
    count = 0
    samples_to_take = 300
    skip_frames = 0

    while count < samples_to_take:
        ret, frame = cap.read()
        if not ret:
            break

        skip_frames += 1
        if skip_frames < 3:
            continue
        skip_frames = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            (x, y, w, h) = faces[0]
            
            face_roi = gray[y:y+h, x:x+w]
            
            if is_good_quality_image(face_roi, min_variance=60):
                face_img = cv2.resize(face_roi, img_size)
                face_img = cv2.equalizeHist(face_img)
                
                file_path = os.path.join(save_path, f"{count+1:03d}.jpg")
                cv2.imwrite(file_path, face_img)
                count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured: {count}/{samples_to_take}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.putText(frame, "Poor quality", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        progress = int((count / samples_to_take) * 100)
        cv2.putText(frame, f"Progress: {progress}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Collected {count} images for '{person_name}'.")

def train_model():
    print("\n🔄 Loading dataset and training model...")
    X, y = [], []

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ Dataset is empty. Please capture faces first.")
        return

    print("📊 Loading images...")
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        folder_images = 0
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, img_size)
            image = cv2.equalizeHist(image)
            X.append(image)
            y.append(folder_name)
            folder_images += 1
        
        print(f"✅ Loaded {folder_images} images for '{folder_name}'")

    if len(X) < 100:
        print("❌ Need at least 100 images total for training.")
        return

    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1).astype('float32') / 255.0

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.25, random_state=42, stratify=y_encoded
    )

    print(f"📈 Training data: {len(X_train)}, Validation data: {len(X_val)}")

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1),
               kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu',
               kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    print("🚀 Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=8),
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✅ Final Validation Accuracy: {val_acc*100:.2f}%")

    if val_acc >= 0.90:
        print("🎉 Excellent accuracy!")
    elif val_acc >= 0.80:
        print("👍 Good accuracy!")
    else:
        print("⚠️  Consider collecting more varied data.")

    model.save("face_recognition_model.keras")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("✅ Training complete. Model saved.")

def recognize_faces():
    """ฟังก์ชันตรวจจับใบหน้าแบบใหม่ - รองรับหลายคนพร้อมกัน"""
    global CONFIDENCE_THRESHOLD, MIN_CONFIDENCE_GAP
    
    try:
        model = load_model("face_recognition_model.keras")
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("❌ Model not found. Train the model first.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🟢 Enhanced multi-person recognition started!")
    print(f"🎚️  Confidence threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"🎚️  Confidence gap threshold: {MIN_CONFIDENCE_GAP}%")
    print("Press 'q' to quit")
    
    face_trackers = []
    next_face_id = 0
    max_distance_threshold = 80  # พิกเซล สำหรับการจับคู่ใบหน้า
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        current_time = time.time()
        
        # อัพเดท missing frames สำหรับ tracker ทั้งหมด
        for tracker in face_trackers[:]:
            tracker.missing_frames += 1
            if tracker.missing_frames > tracker.max_missing_frames:
                face_trackers.remove(tracker)
        
        # จับคู่ใบหน้าที่ตรวจพบกับ tracker ที่มีอยู่
        matched_faces = set()
        
        for face_bbox in faces:
            best_tracker = None
            min_distance = float('inf')
            
            # หา tracker ที่ใกล้ที่สุด
            for tracker in face_trackers:
                distance = tracker.distance_to(face_bbox)
                if distance < min_distance and distance < max_distance_threshold:
                    min_distance = distance
                    best_tracker = tracker
            
            if best_tracker:
                # อัพเดท tracker ที่มีอยู่
                best_tracker.update(face_bbox)
                matched_faces.add(id(best_tracker))
            else:
                # สร้าง tracker ใหม่
                new_tracker = FaceTracker(next_face_id, face_bbox)
                face_trackers.append(new_tracker)
                next_face_id += 1
        
        # ประมวลผลใบหน้าแต่ละคน
        for tracker in face_trackers:
            if tracker.missing_frames > 0:  # ข้าม tracker ที่ไม่พบในเฟรมนี้
                continue
                
            x, y, w, h = tracker.bbox
            face_roi = gray[y:y+h, x:x+w]
            
            # ตรวจสอบคุณภาพ
            if not is_good_quality_image(face_roi, min_variance=50):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.putText(frame, f"Poor Quality (ID:{tracker.face_id})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                continue

            # เตรียมข้อมูลสำหรับ model
            face_roi = cv2.resize(face_roi, img_size)
            face_roi = cv2.equalizeHist(face_roi)
            face_roi = face_roi.reshape(1, img_size[0], img_size[1], 1).astype('float32') / 255.0

            # ทำนายผล
            predictions = model.predict(face_roi, verbose=0)
            confidence = np.max(predictions) * 100
            label_idx = np.argmax(predictions)
            
            # คำนวณ confidence gap
            sorted_predictions = np.sort(predictions[0])[::-1]
            confidence_gap = (sorted_predictions[0] - sorted_predictions[1]) * 100 if len(sorted_predictions) > 1 else confidence
            
            # เพิ่มการทำนายลง tracker
            tracker.add_prediction(label_idx, confidence, confidence_gap)
            
            # ได้ข้อมูลสำหรับแสดงผล
            name, color, avg_confidence = tracker.get_display_info(le)
            
            # วาดกรอบและข้อมูล
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            if name == "Unknown":
                label = f"Unknown (ID:{tracker.face_id})"
                detail = f"Conf: {confidence:.1f}% | Gap: {confidence_gap:.1f}%"
            else:
                label = f"{name} (ID:{tracker.face_id})"
                detail = f"Avg: {avg_confidence:.1f}% | Stable: {'✓' if tracker.is_stable else '✗'}"
            
            cv2.putText(frame, label, (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, detail, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # แสดงข้อมูลระบบ
        cv2.putText(frame, f"Active Trackers: {len(face_trackers)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD}% | Gap: {MIN_CONFIDENCE_GAP}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Enhanced Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🔴 Recognition stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()

def delete_person_data():
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ Dataset is empty.")
        return
    
    print("\n📁 Available persons:")
    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    for i, person in enumerate(persons, 1):
        img_count = len([f for f in os.listdir(os.path.join(dataset_path, person)) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{i}. {person} ({img_count} images)")
    
    person_name = input("\nEnter the person's name to delete: ").strip()
    person_path = os.path.join(dataset_path, person_name)

    if os.path.exists(person_path):
        confirm = input(f"⚠️  Are you sure you want to delete '{person_name}'? (y/n): ")
        if confirm.lower() == 'y':
            try:
                shutil.rmtree(person_path)
                print(f"✅ Deleted data for '{person_name}'. Please retrain the model.")
                
                if os.path.exists("face_recognition_model.keras"):
                    delete_model = input("Delete trained model too? (y/n): ")
                    if delete_model.lower() == 'y':
                        os.remove("face_recognition_model.keras")
                        if os.path.exists("label_encoder.pkl"):
                            os.remove("label_encoder.pkl")
                        print("🗑️  Model files deleted too.")
                        
            except Exception as e:
                print(f"❌ Error deleting: {e}")
        else:
            print("❌ Deletion cancelled.")
    else:
        print("❌ Person not found in dataset.")

def view_dataset():
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ Dataset is empty.")
        return
    
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    
    total_images = 0
    total_persons = 0
    min_images = float('inf')
    max_images = 0
    
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            img_count = len([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            status = ""
            if img_count >= 200:
                status = "🟢 Excellent"
            elif img_count >= 100:
                status = "🟡 Good"
            else:
                status = "🔴 Need more"
            
            print(f"👤 {folder_name:15} : {img_count:3d} images {status}")
            
            total_images += img_count
            total_persons += 1
            min_images = min(min_images, img_count)
            max_images = max(max_images, img_count)
    
    print("="*60)
    print(f"📈 Total persons  : {total_persons}")
    print(f"📈 Total images   : {total_images}")
    if total_persons > 0:
        print(f"📊 Average/person : {total_images//total_persons}")
        print(f"📊 Min images     : {min_images}")
        print(f"📊 Max images     : {max_images}")
    
    print("="*60)
    
    if total_images < 200:
        print("💡 Recommendation: Collect at least 200+ images per person")
    elif any(len([f for f in os.listdir(os.path.join(dataset_path, d)) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) < 100 
            for d in os.listdir(dataset_path) 
            if os.path.isdir(os.path.join(dataset_path, d))):
        print("💡 Recommendation: Some persons need more images")
    else:
        print("✅ Dataset looks good for training!")

def test_model():
    """ทดสอบ model กับรูปที่บันทึกไว้"""
    try:
        model = load_model("face_recognition_model.keras")
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("❌ Model not found. Train the model first.")
        return

    if not os.path.exists(dataset_path):
        print("❌ Dataset not found.")
        return
    
    print("\n🧪 Testing model with dataset images...")
    
    correct = 0
    total = 0
    
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
            
        person_correct = 0
        person_total = 0
        
        images = [f for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        test_images = np.random.choice(images, min(10, len(images)), replace=False)
        
        for img_name in test_images:
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                continue
                
            image = cv2.resize(image, img_size)
            image = cv2.equalizeHist(image)
            image = image.reshape(1, img_size[0], img_size[1], 1).astype('float32') / 255.0
            
            predictions = model.predict(image, verbose=0)
            confidence = np.max(predictions) * 100
            predicted_idx = np.argmax(predictions)
            predicted_name = le.inverse_transform([predicted_idx])[0]
            
            # ใช้เงื่อนไขเดียวกับระบบจริง
            if predicted_name == person_name and confidence >= CONFIDENCE_THRESHOLD:
                correct += 1
                person_correct += 1
            
            total += 1
            person_total += 1
        
        accuracy = (person_correct/person_total)*100 if person_total > 0 else 0
        print(f"👤 {person_name:15} : {person_correct:2d}/{person_total:2d} = {accuracy:5.1f}%")
    
    overall_accuracy = (correct/total)*100 if total > 0 else 0
    print("="*50)
    print(f"🎯 Overall Test Accuracy: {correct}/{total} = {overall_accuracy:.1f}%")
    
    if overall_accuracy >= 90:
        print("🎉 Excellent performance!")
    elif overall_accuracy >= 80:
        print("👍 Good performance!")
    else:
        print("⚠️  Model needs improvement. Consider:")
        print("   • Collecting more varied images")
        print("   • Retraining with different parameters")

def adjust_threshold():
    """ปรับค่า threshold สำหรับการตรวจจับ"""
    global CONFIDENCE_THRESHOLD, MIN_CONFIDENCE_GAP
    
    print("\n⚙️  THRESHOLD ADJUSTMENT")
    print("="*50)
    print("Higher confidence threshold = More strict recognition")
    print("Higher gap threshold = Better unknown detection")
    print("="*50)
    
    print(f"📊 Current confidence threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"📊 Current confidence gap threshold: {MIN_CONFIDENCE_GAP}%")
    
    try:
        print("\n1. Adjust confidence threshold")
        print("2. Adjust confidence gap threshold") 
        print("3. Reset to defaults")
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            new_threshold = input(f"Enter new confidence threshold (70-99) or 'q' to cancel: ").strip()
            if new_threshold.lower() != 'q':
                new_threshold = float(new_threshold)
                if 70 <= new_threshold <= 99:
                    CONFIDENCE_THRESHOLD = new_threshold
                    print(f"✅ Confidence threshold updated to {CONFIDENCE_THRESHOLD}%")
                else:
                    print("❌ Invalid threshold. Must be between 70-99")
                    
        elif choice == '2':
            new_gap = input(f"Enter new confidence gap threshold (5-30) or 'q' to cancel: ").strip()
            if new_gap.lower() != 'q':
                new_gap = float(new_gap)
                if 5 <= new_gap <= 30:
                    MIN_CONFIDENCE_GAP = new_gap
                    print(f"✅ Confidence gap threshold updated to {MIN_CONFIDENCE_GAP}%")
                else:
                    print("❌ Invalid gap threshold. Must be between 5-30")
                    
        elif choice == '3':
            CONFIDENCE_THRESHOLD = 88.0
            MIN_CONFIDENCE_GAP = 15.0
            print("✅ Thresholds reset to defaults")
            
    except ValueError:
        print("❌ Invalid input. Please enter a number.")

def live_adjustment():
    """ปรับ threshold แบบ real-time ขณะรันการตรวจจับ"""
    global CONFIDENCE_THRESHOLD, MIN_CONFIDENCE_GAP
    
    try:
        model = load_model("face_recognition_model.keras")
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("❌ Model not found. Train the model first.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🎚️  LIVE THRESHOLD ADJUSTMENT")
    print("Controls:")
    print("  W/S = Confidence threshold ↑↓")
    print("  A/D = Gap threshold ↑↓") 
    print("  Q = Quit")
    
    face_trackers = []
    next_face_id = 0
    max_distance_threshold = 80
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        # อัพเดท missing frames
        for tracker in face_trackers[:]:
            tracker.missing_frames += 1
            if tracker.missing_frames > tracker.max_missing_frames:
                face_trackers.remove(tracker)
        
        # จับคู่ใบหน้า
        for face_bbox in faces:
            best_tracker = None
            min_distance = float('inf')
            
            for tracker in face_trackers:
                distance = tracker.distance_to(face_bbox)
                if distance < min_distance and distance < max_distance_threshold:
                    min_distance = distance
                    best_tracker = tracker
            
            if best_tracker:
                best_tracker.update(face_bbox)
            else:
                new_tracker = FaceTracker(next_face_id, face_bbox)
                face_trackers.append(new_tracker)
                next_face_id += 1
        
        # ประมวลผลแต่ละใบหน้า
        for tracker in face_trackers:
            if tracker.missing_frames > 0:
                continue
                
            x, y, w, h = tracker.bbox
            face_roi = gray[y:y+h, x:x+w]
            
            if is_good_quality_image(face_roi, min_variance=50):
                face_roi = cv2.resize(face_roi, img_size)
                face_roi = cv2.equalizeHist(face_roi)
                face_roi = face_roi.reshape(1, img_size[0], img_size[1], 1).astype('float32') / 255.0

                predictions = model.predict(face_roi, verbose=0)
                confidence = np.max(predictions) * 100
                label_idx = np.argmax(predictions)
                
                sorted_predictions = np.sort(predictions[0])[::-1]
                confidence_gap = (sorted_predictions[0] - sorted_predictions[1]) * 100 if len(sorted_predictions) > 1 else confidence
                
                tracker.add_prediction(label_idx, confidence, confidence_gap)
                name, color, avg_confidence = tracker.get_display_info(le)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                if name == "Unknown":
                    label = f"Unknown (ID:{tracker.face_id})"
                    detail = f"C:{confidence:.1f}% G:{confidence_gap:.1f}%"
                else:
                    label = f"{name} (ID:{tracker.face_id})"
                    detail = f"Avg:{avg_confidence:.1f}%"
                
                cv2.putText(frame, label, (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, detail, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # แสดงข้อมูลระบบ
        cv2.putText(frame, f"Conf Threshold: {CONFIDENCE_THRESHOLD:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Gap Threshold: {MIN_CONFIDENCE_GAP:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "W/S=Conf, A/D=Gap, Q=Quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Live Threshold Adjustment", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w') and CONFIDENCE_THRESHOLD < 99:
            CONFIDENCE_THRESHOLD += 1.0
            print(f"🔺 Confidence Threshold: {CONFIDENCE_THRESHOLD}%")
        elif key == ord('s') and CONFIDENCE_THRESHOLD > 70:
            CONFIDENCE_THRESHOLD -= 1.0
            print(f"🔻 Confidence Threshold: {CONFIDENCE_THRESHOLD}%")
        elif key == ord('d') and MIN_CONFIDENCE_GAP < 30:
            MIN_CONFIDENCE_GAP += 1.0
            print(f"🔺 Gap Threshold: {MIN_CONFIDENCE_GAP}%")
        elif key == ord('a') and MIN_CONFIDENCE_GAP > 5:
            MIN_CONFIDENCE_GAP -= 1.0
            print(f"🔻 Gap Threshold: {MIN_CONFIDENCE_GAP}%")

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Thresholds set to: Confidence={CONFIDENCE_THRESHOLD}%, Gap={MIN_CONFIDENCE_GAP}%")

def advanced_test():
    """ทดสอบระบบแบบ advanced พร้อมแสดง confusion matrix"""
    try:
        model = load_model("face_recognition_model.keras")
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("❌ Model not found. Train the model first.")
        return

    if not os.path.exists(dataset_path):
        print("❌ Dataset not found.")
        return
    
    print("\n🔬 ADVANCED MODEL TESTING")
    print("="*50)
    
    all_true_labels = []
    all_pred_labels = []
    all_confidences = []
    
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
            
        images = [f for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # ทดสอบ 20 รูปต่อคน
        test_images = np.random.choice(images, min(20, len(images)), replace=False)
        
        for img_name in test_images:
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                continue
                
            image = cv2.resize(image, img_size)
            image = cv2.equalizeHist(image)
            image = image.reshape(1, img_size[0], img_size[1], 1).astype('float32') / 255.0
            
            predictions = model.predict(image, verbose=0)
            confidence = np.max(predictions) * 100
            predicted_idx = np.argmax(predictions)
            predicted_name = le.inverse_transform([predicted_idx])[0]
            
            # เพิ่มเงื่อนไข confidence gap
            sorted_predictions = np.sort(predictions[0])[::-1]
            confidence_gap = (sorted_predictions[0] - sorted_predictions[1]) * 100 if len(sorted_predictions) > 1 else confidence
            
            # ใช้เงื่อนไขเข้มงวด
            if confidence >= CONFIDENCE_THRESHOLD and confidence_gap >= MIN_CONFIDENCE_GAP:
                final_prediction = predicted_name
            else:
                final_prediction = "Unknown"
            
            all_true_labels.append(person_name)
            all_pred_labels.append(final_prediction)
            all_confidences.append(confidence)
    
    # คำนวณ metrics
    from collections import defaultdict
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for true_label, pred_label in zip(all_true_labels, all_pred_labels):
        confusion_matrix[true_label][pred_label] += 1
    
    # แสดงผล
    all_persons = list(set(all_true_labels))
    print(f"\n📊 Confusion Matrix (Confidence≥{CONFIDENCE_THRESHOLD}%, Gap≥{MIN_CONFIDENCE_GAP}%):")
    print("="*60)
    
    total_correct = 0
    total_samples = 0
    
    for person in all_persons:
        correct = confusion_matrix[person][person]
        unknown = confusion_matrix[person]["Unknown"]
        wrong = sum(confusion_matrix[person][other] for other in confusion_matrix[person] 
                   if other != person and other != "Unknown")
        
        total = correct + unknown + wrong
        accuracy = (correct/total)*100 if total > 0 else 0
        
        print(f"👤 {person:12} | Correct: {correct:2d} | Unknown: {unknown:2d} | Wrong: {wrong:2d} | Acc: {accuracy:5.1f}%")
        
        total_correct += correct
        total_samples += total
    
    overall_accuracy = (total_correct/total_samples)*100 if total_samples > 0 else 0
    
    print("="*60)
    print(f"🎯 Overall Recognition Rate: {total_correct}/{total_samples} = {overall_accuracy:.1f}%")
    print(f"📈 Average Confidence: {np.mean(all_confidences):.1f}%")
    
    # คำแนะนำ
    if overall_accuracy >= 95:
        print("🎉 Excellent! System is ready for production.")
    elif overall_accuracy >= 85:
        print("👍 Good performance. Consider fine-tuning thresholds.")
    else:
        print("⚠️  Needs improvement:")
        print("   • Collect more diverse training images")
        print("   • Adjust confidence/gap thresholds") 
        print("   • Retrain with better parameters")

def main():
    global CONFIDENCE_THRESHOLD, MIN_CONFIDENCE_GAP
    
    # สร้าง dataset directory
    try:
        os.makedirs(dataset_path, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating dataset directory: {e}")
        return

    while True:
        print("\n" + "="*70)
        print("🤖 ENHANCED MULTI-PERSON FACE RECOGNITION SYSTEM")
        print("="*70)
        print("1. 📷 Capture new faces")
        print("2. 🧠 Train the model")
        print("3. 🔍 Multi-person recognition (Enhanced)")
        print("4. 🧪 Basic model test")
        print("5. 🔬 Advanced model test")
        print("6. 📊 View dataset summary")
        print("7. ⚙️  Adjust thresholds")
        print("8. 🎚️  Live threshold adjustment")
        print("9. 🗑️  Delete person data")
        print("10. 🚪 Exit")
        print("="*70)
        print(f"Current Settings - Confidence: {CONFIDENCE_THRESHOLD}% | Gap: {MIN_CONFIDENCE_GAP}%")
        
        choice = input("Enter your choice (1-10): ").strip()

        choice = input("Enter your choice (1-10): ").strip()

        if choice == '1':
            capture_faces()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            test_model()
        elif choice == '5':
            advanced_test()
        elif choice == '6':
            view_dataset()
        elif choice == '7':
            adjust_threshold()
        elif choice == '8':
            live_adjustment()
        elif choice == '9':
            delete_person_data()
        elif choice == '10':
            print("👋 Thank you for using Enhanced Face Recognition System!")
            print("\n🔥 SYSTEM PERFORMANCE TIPS:")
            print("="*50)
            print("📸 Data Collection:")
            print("   • Collect 300+ images per person")
            print("   • Use varied lighting conditions")
            print("   • Include different facial expressions")
            print("   • Capture from multiple angles")
            print("   • Ensure consistent image quality")
            print("\n⚙️  Threshold Settings:")
            print("   • Higher confidence = More strict recognition")
            print("   • Higher gap = Better unknown detection")
            print("   • Balance based on your environment")
            print("\n🎯 Multi-Person Features:")
            print("   • System tracks up to 10 faces simultaneously")
            print("   • Each person gets unique ID for tracking")
            print("   • Stability checking prevents false positives")
            print("   • Individual prediction buffers per person")
            print(f"\n💾 Your final settings:")
            print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}%")
            print(f"   Confidence Gap Threshold: {MIN_CONFIDENCE_GAP}%")
            print("\n✨ System ready for production use!")
            break
        else:
            print("❌ Invalid input. Please enter a number 1–10.")

def system_info():
    """แสดงข้อมูลระบบและการตั้งค่า"""
    print("\n🖥️  SYSTEM INFORMATION")
    print("="*50)
    
    # ตรวจสอบ OpenCV
    try:
        print(f"✅ OpenCV version: {cv2.__version__}")
    except:
        print("❌ OpenCV not found!")
    
    # ตรวจสอบ TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
    except:
        print("❌ TensorFlow not found!")
    
    # ตรวจสอบ Camera
    cap_test = cv2.VideoCapture(camera_index)
    if cap_test.isOpened():
        print(f"✅ Camera {camera_index} available")
        cap_test.release()
    else:
        print(f"❌ Camera {camera_index} not available")
    
    # ตรวจสอบ Haar Cascade
    if os.path.exists(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
        print("✅ Haar Cascade face detector loaded")
    else:
        print("❌ Haar Cascade face detector not found!")
    
    # ตรวจสอบ Dataset
    if os.path.exists(dataset_path):
        persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        total_images = sum([len([f for f in os.listdir(os.path.join(dataset_path, p)) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) 
                           for p in persons])
        print(f"✅ Dataset: {len(persons)} persons, {total_images} images")
    else:
        print("❌ Dataset directory not found")
    
    # ตรวจสอบ Model
    if os.path.exists("face_recognition_model.keras"):
        print("✅ Trained model found")
    else:
        print("❌ No trained model found")
    
    if os.path.exists("label_encoder.pkl"):
        print("✅ Label encoder found")
    else:
        print("❌ No label encoder found")
    
    print("="*50)
    print("🎚️  Current Thresholds:")
    print(f"   Confidence: {CONFIDENCE_THRESHOLD}%")
    print(f"   Gap: {MIN_CONFIDENCE_GAP}%")
    print(f"   Stability frames: {MIN_STABILITY_FRAMES}")
    print("="*50)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Face Recognition System...")
    print("🔧 Performing system checks...")
    
    # ตรวจสอบระบบ
    system_requirements_ok = True
    
    # ตรวจสอบ OpenCV
    try:
        cv2_version = cv2.__version__
        print(f"✅ OpenCV {cv2_version} detected")
    except:
        print("❌ OpenCV not found! Please install: pip install opencv-python")
        system_requirements_ok = False
    
    # ตรวจสอบ TensorFlow
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        print(f"✅ TensorFlow {tf_version} detected")
    except:
        print("❌ TensorFlow not found! Please install: pip install tensorflow")
        system_requirements_ok = False
    
    # ตรวจสอบ sklearn
    try:
        import sklearn
        print(f"✅ scikit-learn {sklearn.__version__} detected")
    except:
        print("❌ scikit-learn not found! Please install: pip install scikit-learn")
        system_requirements_ok = False
    
    if not system_requirements_ok:
        print("❌ System requirements not met. Please install missing packages.")
        exit(1)
    
    # ตรวจสอบ camera
    print("📹 Testing camera...")
    cap_test = cv2.VideoCapture(camera_index)
    if cap_test.isOpened():
        print(f"✅ Camera {camera_index} is working")
        cap_test.release()
    else:
        print(f"⚠️  Camera {camera_index} not available")
        print("   Face capture will not work, but you can still train/test with existing data")
    
    # ตรวจสอบ Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        print("✅ Face detection model loaded successfully")
    else:
        print("❌ Haar Cascade file not found!")
        system_requirements_ok = False
    
    # สร้าง dataset directory
    try:
        os.makedirs(dataset_path, exist_ok=True)
        print(f"✅ Dataset directory ready: {dataset_path}")
    except Exception as e:
        print(f"❌ Cannot create dataset directory: {e}")
        print("Please check path permissions and try again.")
        system_requirements_ok = False
    
    if not system_requirements_ok:
        print("❌ Critical system requirements not met. Exiting...")
        exit(1)
    
    # แสดงข้อมูลระบบ
    existing_persons = 0
    existing_images = 0
    
    if os.path.exists(dataset_path):
        persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        existing_persons = len(persons)
        existing_images = sum([len([f for f in os.listdir(os.path.join(dataset_path, p)) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) 
                              for p in persons])
    
    if existing_persons > 0:
        print(f"📊 Found existing dataset: {existing_persons} persons, {existing_images} images")
    else:
        print("📊 No existing dataset found - ready for new data collection")
    
    if os.path.exists("face_recognition_model.keras"):
        print("🧠 Trained model found - ready for recognition")
    else:
        print("🧠 No trained model - you'll need to train after collecting data")
    
    print("\n" + "="*70)
    print("🎯 ENHANCED FEATURES:")
    print("✨ Multi-person simultaneous recognition")
    print("🔍 Advanced unknown detection with confidence gap analysis") 
    print("📊 Individual face tracking with stability checking")
    print("⚙️  Real-time threshold adjustment")
    print("🧪 Comprehensive model testing and analysis")
    print("="*70)
    
    print(f"\n🎚️  Default Settings:")
    print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"   Confidence Gap Threshold: {MIN_CONFIDENCE_GAP}%")
    print(f"   Stability Frames Required: {MIN_STABILITY_FRAMES}")
    
    print("\n🚀 System initialization complete! Starting main menu...")
    
    # เรียกใช้ main function
    main()
