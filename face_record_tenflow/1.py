import cv2
import os
import time

# --- การตั้งค่าที่ปรับปรุงใหม่ ---
# ชื่อของบุคคลที่ต้องการเก็บภาพ
person_name = "Tawan"
output_folder = f"train/{person_name}"

# จำนวนรูปภาพที่ต้องการเก็บ
max_images = 50
saved_count = 0

# (ปรับปรุง) ตั้งค่าขนาดขั้นต่ำของใบหน้าที่จะตรวจจับ (เป็น pixel)
# ช่วยกรองใบหน้าที่อยู่ไกลหรือเล็กเกินไปออกไป
MIN_FACE_SIZE = (120, 120)

# (ปรับปรุง) ตั้งค่าการหน่วงเวลาระหว่างการบันทึกแต่ละภาพ (เป็นวินาที)
# เพื่อให้มีเวลาขยับใบหน้าเล็กน้อย ทำให้ได้มุมที่หลากหลายขึ้น
CAPTURE_DELAY = 0.2  # 0.2 วินาที
last_capture_time = time.time()

# --- โหลดตัวตรวจจับใบหน้าที่แม่นยำขึ้น ---
# ใช้ haarcascade_frontalface_default.xml ซึ่งมาพร้อมกับ OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ตรวจสอบว่าไฟล์ cascade ถูกโหลดหรือไม่
if face_cascade.empty():
    print("❌ ERROR: Failed to load cascade file. Please check your OpenCV installation.")
    exit()

# --- เริ่มการทำงาน ---
print("กำลังเปิดกล้อง...")
cap = cv2.VideoCapture(0)

# (ปรับปรุง) ตรวจสอบว่าเปิดกล้องสำเร็จหรือไม่
if not cap.isOpened():
    print("❌ ERROR: ไม่สามารถเปิดกล้องได้, กรุณาตรวจสอบการเชื่อมต่อหรือสิทธิ์การเข้าถึง")
    exit()

# สร้างโฟลเดอร์สำหรับเก็บภาพ ถ้ายังไม่มี
os.makedirs(output_folder, exist_ok=True)
print("✅ เปิดกล้องสำเร็จ, เริ่มการจับภาพ...")
print(f"คำแนะนำ: ค่อยๆ หันใบหน้าไปทางซ้าย-ขวา, ขึ้น-ลง เล็กน้อยเพื่อให้ได้ภาพที่หลากหลาย")

while saved_count < max_images:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: ไม่สามารถอ่านเฟรมจากกล้องได้")
        break

    # พลิกภาพซ้าย-ขวา (เหมือนส่องกระจก) เพื่อให้ใช้งานง่ายขึ้น
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=MIN_FACE_SIZE
    )

    # (ปรับปรุง) ประมวลผลเฉพาะใบหน้าที่ใหญ่ที่สุดที่เจอในเฟรม
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]  # เลือกใช้เฉพาะใบหน้าที่ใหญ่ที่สุด

        current_time = time.time()
        # (ปรับปรุง) ตรวจสอบเงื่อนไขการหน่วงเวลาก่อนบันทึก
        if current_time - last_capture_time > CAPTURE_DELAY:
            # ครอบเฉพาะใบหน้า
            face_img = frame[y:y + h, x:x + w]

            # (ปรับปรุง) ตรวจสอบคุณภาพของภาพ (ความเบลอ)
            laplacian_var = cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            
            # ตั้งค่าเกณฑ์ความเบลอ (ปรับค่าได้ตามความเหมาะสม)
            BLUR_THRESHOLD = 50 
            
            if laplacian_var > BLUR_THRESHOLD:
                # บันทึกรูปภาพ
                file_path = os.path.join(output_folder, f"{saved_count + 1}.jpg")
                cv2.imwrite(file_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                saved_count += 1
                last_capture_time = current_time  # อัปเดตเวลาที่บันทึกล่าสุด

                # แสดงสถานะบนใบหน้าว่า "บันทึกแล้ว!"
                cv2.putText(frame, "SAVED!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # แสดงสถานะว่า "เบลอ"
                cv2.putText(frame, "BLURRY", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # วาดกรอบสี่เหลี่ยมรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # --- แสดงข้อมูลบนหน้าจอ ---
    # แถบแสดงความคืบหน้า
    progress_percentage = saved_count / max_images if max_images > 0 else 0
    bar_x = 10
    bar_y = frame.shape[0] - 30  # ตำแหน่งด้านล่างของจอ
    bar_width = frame.shape[1] - 20
    bar_height = 20

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), cv2.FILLED)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress_percentage), bar_y + bar_height), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, f"{int(progress_percentage * 100)}%", (frame.shape[1] // 2 - 20, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.putText(frame, f"Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Capturing Faces - By Tawan", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\nผู้ใช้ยกเลิกการทำงาน")
        break

# --- สิ้นสุดการทำงาน ---
cap.release()
cv2.destroyAllWindows()

if saved_count >= max_images:
    print(f"\n✅ เก็บภาพครบแล้ว: {saved_count} ภาพ ที่โฟลเดอร์ {output_folder}")
else:
    print(f"\nสิ้นสุดการทำงาน, เก็บภาพได้ทั้งหมด: {saved_count} ภาพ")
