
# ปรับอีกนิดน่อย มีอาการแหลกน่อยๆๆ

import cv2
import os
import numpy as np
import pickle
import shutil
import time
import random
from collections import deque

from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
import mediapipe as mp
from mtcnn import MTCNN
from keras_facenet import FaceNet

# ---------------------- User-tuneable parameters ----------------------
dataset_path = 'dataset'
embeddings_path = 'face_embeddings.pkl'
camera_index = 0
img_size = (160, 160)

# Preprocessing params
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAMMA_TARGET = 110.0  # target mean brightness (0-255) to normalize toward
GAMMA_MAX = 2.0
GAMMA_MIN = 0.5
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Brightness adaptive bounds (initial)
BRIGHTNESS_EMA_ALPHA = 0.05  # smooth factor for brightness running mean
BRIGHTNESS_STD_EMA_ALPHA = 0.03

# Image clarity thresholds (base)
CLARITY_BASE_THRESHOLD = 80.0

# Liveness / motion thresholds (scaled by face size)
BASE_MOTION_THRESHOLD = 3.0
BASE_Z_THRESHOLD = 0.0018
BASE_EAR_THRESHOLD = 0.20
BASE_MAR_THRESHOLD = 0.35

# Movement buffer length
MOVEMENT_BUFFER_LEN = 12

# FaceNet similarity
SIMILARITY_THRESHOLD = 0.75

# Tracking parameters after verification
TRACK_SIMILARITY_HYSTERESIS = 0.70  # หาก similarity ตกลงมาต่ำกว่าค่านี้บ่อยๆ ให้ถือว่าหลุด
TRACK_MISS_FRAMES_MAX = 6         # กี่เฟรมที่หลุดก่อนหยุดการติดตาม
TRACK_MIN_FRAMES = 5              # อย่างน้อยต้องติดตามกี่เฟรมก่อนนับเป็นการติดตามสำเร็จ

# -----------------------------------------------------------------------

# --- Init models ---
detector = MTCNN()
facenet_model = FaceNet()
print("✅ FaceNet Model Loaded.")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

EYE_INDICES_LEFT = [362, 385, 387, 263, 373, 380]
EYE_INDICES_RIGHT = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14, 78, 308]

# Running stats for brightness
running_brightness_mean = None
running_brightness_std = None

# movement smoothing
movement_buffer = deque(maxlen=MOVEMENT_BUFFER_LEN)

# ---------- helper math ----------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(mouth):
    vertical = dist.euclidean(mouth[0], mouth[1])
    horizontal = dist.euclidean(mouth[2], mouth[3])
    return vertical / (horizontal + 1e-6)

# ---------- preprocessing ----------
def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_frame(frame):
    """
    Enhance frame for detection under varying lighting:
    - Convert to YCrCb, apply CLAHE on Y channel
    - Adjust gamma to push global brightness toward GAMMA_TARGET (adaptive)
    - Bilateral filter to reduce noise while keeping edges
    Returns enhanced BGR image.
    """
    # convert to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # CLAHE on Y
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    y_eq = clahe.apply(y)

    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    enhanced = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # compute current mean brightness on L channel (use Y channel original or eq)
    global running_brightness_mean, running_brightness_std
    cur_mean = float(np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)))
    cur_std = float(np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)))
    if running_brightness_mean is None:
        running_brightness_mean = cur_mean
        running_brightness_std = cur_std
    else:
        running_brightness_mean = (1 - BRIGHTNESS_EMA_ALPHA) * running_brightness_mean + BRIGHTNESS_EMA_ALPHA * cur_mean
        running_brightness_std = (1 - BRIGHTNESS_STD_EMA_ALPHA) * running_brightness_std + BRIGHTNESS_STD_EMA_ALPHA * cur_std

    # adaptive gamma: bring current toward GAMMA_TARGET
    if running_brightness_mean > 0:
        ratio = GAMMA_TARGET / (running_brightness_mean + 1e-6)
        gamma = np.clip(ratio ** 0.5, GAMMA_MIN, GAMMA_MAX)  # moderate gamma
    else:
        gamma = 1.0
    enhanced = adjust_gamma(enhanced, gamma)

    # bilateral filter (edge-preserving denoise)
    enhanced = cv2.bilateralFilter(enhanced, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

    return enhanced

# ---------- adaptive brightness check ----------
def is_brightness_ok_adaptive(frame, margin_low=45, margin_high=45):
    """
    Use running brightness stats to decide if frame is within acceptable brightness.
    margin_low/high are pixel-value margins around running mean.
    """
    global running_brightness_mean, running_brightness_std
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    if running_brightness_mean is None:
        # warm-up: accept reasonable range
        return 40 <= mean_val <= 200
    low = running_brightness_mean - margin_low
    high = running_brightness_mean + margin_high
    return low <= mean_val <= high

def is_image_clear_adaptive(img, face_width_pixel=None, base_thresh=CLARITY_BASE_THRESHOLD):
    """
    Adaptive Laplacian variance threshold. If face ROI small, reduce threshold.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if face_width_pixel is None or face_width_pixel <= 0:
        return var > base_thresh
    # scale threshold inversely with face size (bigger face -> need more detail -> higher threshold)
    scale = face_width_pixel / 200.0  # 200 pixel face considered "normal"
    adapt_thresh = max(20.0, base_thresh * scale)
    return var > adapt_thresh

def resize_rgb_for_facenet(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, img_size)
    return resized

# --- capture faces (same as original but with preprocess) ---
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
        print("❌ Error: Could not open webcam.")
        return

    # Try to enable camera auto exposure/white balance (may or may not work)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # try enabling auto exposure
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    print("\n📸 เตรียมตัวเก็บภาพ โปรดทำตามคำสั่งบนจอ")
    count = 0
    samples_per_pose = 10
    poses = ["หันหน้า ตรง", "หันหน้า ซ้าย", "หันหน้า ขวา", "เงยหน้า", "ก้มหน้า"]

    for pose_idx, pose in enumerate(poses):
        print(f"\n➡️ {pose}")
        target_count = samples_per_pose * (pose_idx + 1)
        while count < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            enhanced = preprocess_frame(frame)

            if not is_brightness_ok_adaptive(enhanced):
                cv2.putText(enhanced, "Adjust lighting!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Capturing Faces', enhanced)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
                continue

            faces = detector.detect_faces(enhanced)
            if len(faces) > 0:
                x, y, w, h = faces[0]['box']
                x1, y1 = abs(x), abs(y)
                x2, y2 = x1 + w, y1 + h
                face_img = enhanced[y1:y2, x1:x2]

                if face_img.size > 0 and is_image_clear_adaptive(face_img, face_width_pixel=w):
                    face_img_resized = cv2.resize(face_img, img_size)
                    cv2.imwrite(os.path.join(save_path, f"{count+1}.jpg"), face_img_resized)
                    count += 1
                    cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(enhanced, f"{pose} - Saved: {count}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(enhanced, "Face too blurry", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(enhanced, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow('Capturing Faces', enhanced)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Finished capturing {count} images for '{person_name}'.")

# --- generate embeddings (same but uses preprocess for robustness) ---
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

            img_for_net = resize_rgb_for_facenet(image)
            face_embedding = facenet_model.embeddings([img_for_net])[0]
            person_embeddings.append(face_embedding)

        if person_embeddings:
            embeddings[folder_name] = np.mean(person_embeddings, axis=0)

    if not embeddings:
        print("❌ No valid images found to generate embeddings.")
        return

    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print("✅ Embeddings generated and saved to 'face_embeddings.pkl'.")

# --- recognition: updated with preprocess, adaptive thresholds, and tracking state ---
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

    # try enabling camera auto-exposure / auto-WB if available
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    print("\nStarting secure recognition (FaceNet). Follow the on-screen challenge.")

    # liveness / blink trackers
    EAR_CONSEC_FRAMES = 3
    blink_counter = 0
    blink_verified = False
    last_blink_time = None

    motion_counter = 0
    motion_verified = False

    prev_landmarks = None
    movement_buffer.clear()

    challenges = ["blink", "turn_left", "turn_right", "open_mouth"]
    current_challenge = random.choice(challenges)
    challenge_start = time.time()
    CHALLENGE_TIMEOUT = 6.0

    prev_center = None

    # state machine: 'idle' (awaiting verification), 'verifying' (challenge in progress), 'tracking' (person verified and tracked)
    state = 'verifying'
    current_name = None
    tracked_name = None
    tracked_similarity = 0.0
    tracked_frames = 0
    tracked_miss = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame to normalize lighting
        enhanced = preprocess_frame(frame)

        h, w, _ = enhanced.shape
        image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        display_text = f"Challenge: {current_challenge}"
        display_color = (0, 165, 255)

        faces = detector.detect_faces(enhanced)

        # rotate challenge if timeout (only when verifying)
        if state == 'verifying' and (time.time() - challenge_start > CHALLENGE_TIMEOUT):
            current_challenge = random.choice(challenges)
            challenge_start = time.time()

        if results.multi_face_landmarks and faces:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = []
            for lm in face_landmarks.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pz = lm.z
                landmarks.append((px, py, pz))

            # derive face bounding box from MTCNN for size scaling
            x_min, y_min, w_mtcnn, h_mtcnn = faces[0]['box']
            face_w = max(1, w_mtcnn)

            # scale thresholds by face size (bigger face -> require more absolute movement)
            size_scale = face_w / 200.0
            MOTION_THRESHOLD = max(1.2, BASE_MOTION_THRESHOLD * (1.0 / size_scale))  # small faces -> less threshold
            Z_MOVEMENT_THRESHOLD = max(0.0008, BASE_Z_THRESHOLD * (1.0 / (size_scale * 1.0)))
            EAR_THRESHOLD = max(0.14, BASE_EAR_THRESHOLD * (0.9))  # keep ear around same
            MAR_THRESHOLD = BASE_MAR_THRESHOLD

            # EAR (blink)
            try:
                left_eye = np.array([(landmarks[i][0], landmarks[i][1]) for i in EYE_INDICES_LEFT])
                right_eye = np.array([(landmarks[i][0], landmarks[i][1]) for i in EYE_INDICES_RIGHT])
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
            except Exception:
                ear = 1.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EAR_CONSEC_FRAMES:
                    current_time = time.time()
                    if last_blink_time is None or (current_time - last_blink_time > 0.9):
                        blink_verified = True
                        last_blink_time = current_time
                blink_counter = 0

            # mouth MAR
            try:
                mouth_pts = np.array([(landmarks[i][0], landmarks[i][1]) for i in MOUTH_INDICES])
                mar = mouth_aspect_ratio(mouth_pts)
            except Exception:
                mar = 0.0
            mouth_verified = mar > MAR_THRESHOLD

            # movement detection using landmarks (xy + z)
            if prev_landmarks is not None and len(prev_landmarks) == len(landmarks):
                dists_xy = [np.linalg.norm(np.array(landmarks[i][:2]) - np.array(prev_landmarks[i][:2])) for i in range(len(landmarks))]
                mean_xy = float(np.mean(dists_xy))
                dists_z = [abs(landmarks[i][2] - prev_landmarks[i][2]) for i in range(len(landmarks))]
                mean_z = float(np.mean(dists_z))
                combined_movement = mean_xy + (mean_z * 1000.0)
                movement_buffer.append(combined_movement)
                avg_movement = float(np.mean(movement_buffer))
                var_movement = float(np.var(movement_buffer))

                if avg_movement > MOTION_THRESHOLD or mean_z > Z_MOVEMENT_THRESHOLD:
                    motion_counter += 1
                else:
                    motion_counter = 0

                # detect video-like stable loop (very low variance)
                if len(movement_buffer) == movement_buffer.maxlen and var_movement < 0.035:
                    display_text = "Stable movement -> possible replay attack"
                    display_color = (0, 0, 255)
                # if motion_counter large enough -> verified movement
                if motion_counter >= max(4, int(6.0 / max(0.5, size_scale))):
                    motion_verified = True
                    movement_buffer.clear()
                    motion_counter = 0

            prev_landmarks = landmarks

            # Evaluate challenge (only relevant in 'verifying' state)
            challenge_ok = False
            if state == 'verifying':
                if current_challenge == "blink" and blink_verified:
                    challenge_ok = True
                elif current_challenge == "open_mouth" and mouth_verified:
                    challenge_ok = True
                elif current_challenge in ["turn_left", "turn_right"]:
                    xs = [p[0] for p in landmarks]
                    center_x = np.mean(xs)
                    if prev_center is not None:
                        dx = center_x - prev_center
                        # threshold scaled by face width
                        turn_thresh = max(8, face_w * 0.04)
                        if current_challenge == "turn_left" and dx < -turn_thresh:
                            challenge_ok = True
                        if current_challenge == "turn_right" and dx > turn_thresh:
                            challenge_ok = True
                    prev_center = center_x

            # face roi from MTCNN
            x_min, y_min, w_mtcnn, h_mtcnn = faces[0]['box']
            x_max, y_max = x_min + w_mtcnn, y_min + h_mtcnn
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
            face_roi = enhanced[y_min:y_max, x_min:x_max]

            # If currently tracking someone, then prefer tracking logic
            if state == 'tracking' and tracked_name is not None:
                # keep verifying that it's still the same person by computing embedding similarity
                if face_roi.size == 0:
                    tracked_miss += 1
                else:
                    # brightness & clarity check still matters
                    if (not is_brightness_ok_adaptive(face_roi, margin_low=50, margin_high=60)
                            or not is_image_clear_adaptive(face_roi, face_width_pixel=w_mtcnn)):
                        # do not immediately count as miss, but increase miss
                        tracked_miss += 1
                    else:
                        face_roi_resized = resize_rgb_for_facenet(face_roi)
                        try:
                            live_embedding = facenet_model.embeddings([face_roi_resized])[0]
                            stored_embedding = embeddings.get(tracked_name, None)
                            if stored_embedding is not None:
                                sim = 1 - dist.cosine(live_embedding, stored_embedding)
                                tracked_similarity = sim
                                # if similarity above hysteresis, reset miss count
                                if sim >= TRACK_SIMILARITY_HYSTERESIS:
                                    tracked_miss = 0
                                else:
                                    tracked_miss += 1
                            else:
                                tracked_miss += 1
                        except Exception:
                            tracked_miss += 1

                tracked_frames += 1
                display_text = f"{tracked_name} (tracking) {tracked_similarity:.2f}"
                display_color = (0, 255, 0)

                # if too many consecutive misses -> stop tracking and go back to verifying (new person)
                if tracked_miss >= TRACK_MISS_FRAMES_MAX:
                    # reset state
                    print(f"[INFO] Tracking for '{tracked_name}' ended (missed {tracked_miss} frames).")
                    state = 'verifying'
                    current_challenge = random.choice(challenges)
                    challenge_start = time.time()
                    tracked_name = None
                    tracked_similarity = 0.0
                    tracked_frames = 0
                    tracked_miss = 0
                    prev_center = None
                # otherwise continue tracking and don't require challenge
                # draw box and landmarks below
            else:
                # state == 'verifying' or 'idle' (we keep 'verifying' as default)
                verified = False
                if state == 'verifying' and challenge_ok:
                    # require at least one liveness signal + challenge to avoid recorded videos that mimic challenge
                    if (blink_verified or motion_verified or mouth_verified):
                        verified = True

                if verified:
                    # check ROI and quality first
                    if face_roi.size == 0:
                        display_text = "Face ROI invalid"
                        display_color = (0, 0, 255)
                    else:
                        if (not is_brightness_ok_adaptive(face_roi, margin_low=50, margin_high=60)
                                or not is_image_clear_adaptive(face_roi, face_width_pixel=w_mtcnn)):
                            display_text = "Adjust Lighting / Hold Still"
                            display_color = (0, 165, 255)
                        else:
                            # compute embedding and compare
                            face_roi_resized = resize_rgb_for_facenet(face_roi)
                            live_embedding = facenet_model.embeddings([face_roi_resized])[0]

                            best_match_name = "Unknown"
                            best_similarity = 0.0
                            for name, stored_embedding in embeddings.items():
                                similarity = 1 - dist.cosine(live_embedding, stored_embedding)
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match_name = name

                            if best_similarity > SIMILARITY_THRESHOLD:
                                # VERIFIED -> enter tracking state for this person
                                tracked_name = best_match_name
                                tracked_similarity = best_similarity
                                tracked_frames = 1
                                tracked_miss = 0
                                state = 'tracking'
                                display_text = f"{tracked_name} ({best_similarity:.2f}) - VERIFIED & TRACKING"
                                display_color = (0, 255, 0)
                                print(f"[INFO] Verified and start tracking: {tracked_name} ({best_similarity:.2f})")
                            else:
                                display_text = "Unknown - try again"
                                display_color = (0, 0, 255)

                    # reset challenge-related flags regardless
                    blink_verified = False
                    motion_verified = False
                    movement_buffer.clear()
                    motion_counter = 0
                    blink_counter = 0
                    current_challenge = random.choice(challenges)
                    challenge_start = time.time()
                    prev_center = None
                else:
                    # Not verified yet -> show hints (verifying mode)
                    if blink_counter > 0:
                        display_text = f"Blinking... ({blink_counter}) | Do: {current_challenge}"
                        display_color = (0, 165, 255)
                    elif motion_counter > 0:
                        display_text = f"Moving... ({motion_counter}) | Do: {current_challenge}"
                        display_color = (0, 165, 255)
                    else:
                        display_text = f"Do: {current_challenge}"
                        display_color = (0, 165, 255)

            # draw
            cv2.rectangle(enhanced, (x_min, y_min), (x_max, y_max), display_color, 2)
            mp_draw.draw_landmarks(enhanced, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1))
            cv2.putText(enhanced, display_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        else:
            # no face
            prev_landmarks = None
            movement_buffer.clear()
            blink_counter = 0
            motion_counter = 0

            # if we were tracking someone and now there's no face, count as a miss
            if state == 'tracking':
                tracked_miss += 1
                if tracked_miss >= TRACK_MISS_FRAMES_MAX:
                    print(f"[INFO] Tracking ended for '{tracked_name}' due to lost face.")
                    state = 'verifying'
                    current_challenge = random.choice(challenges)
                    challenge_start = time.time()
                    tracked_name = None
                    tracked_similarity = 0.0
                    tracked_frames = 0
                    tracked_miss = 0
            display_text = "No face detected"
            display_color = (0, 0, 255)
            cv2.putText(enhanced, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, display_color, 2)

        cv2.imshow('Secure Face Recognition (FaceNet + Robust Liveness + Tracking)', enhanced)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

# --- delete person data ---
def delete_person_data():
    person_name = input("Enter name of person's data to delete: ").strip()
    person_path = os.path.join(dataset_path, person_name)
    if os.path.exists(person_path):
        try:
            shutil.rmtree(person_path)
            print(f"✅ Successfully deleted '{person_name}'. Please generate embeddings again.")
        except OSError as e:
            print(f"❌ Error deleting folder: {e.strerror}")
    else:
        print(f"❌ Dataset for '{person_name}' not found.")

# --- main menu ---
def main():
    if not os.path.exists(dataset_path): os.makedirs(dataset_path)
    while True:
        print("\n" + "="*30 + "\n   Secure Face Recognition System\n" + "="*30)
        print("1. Capture new faces")
        print("2. Generate Face Embeddings")
        print("3. Start secure recognition")
        print("4. Delete a person's data")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ").strip()
        if choice == '1': capture_faces()
        elif choice == '2': generate_embeddings()
        elif choice == '3': recognize_faces()
        elif choice == '4': delete_person_data()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
