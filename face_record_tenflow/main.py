import face_recognition
import cv2
import os
import numpy as np
import pickle

# --- ⚙️ Main Configurations ⚙️ ---
# Path to the pre-computed encodings file.
ENCODINGS_FILE = "encodings.pkl"

# Tolerance for face matching. A lower value means a stricter match.
# This value is now more reliable because our known encodings are averaged.
TOLERANCE = 0.45

# Resize factor for real-time processing.
RESIZE_FACTOR = 4

# --- Main Logic ---
print("Loading pre-computed face encodings...")
if not os.path.exists(ENCODINGS_FILE):
    print(f"❌ ERROR: Encoding file '{ENCODINGS_FILE}' not found.")
    print("Please run the 'create_encodings.py' script first to generate it.")
    exit()

with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

print("✅ Encodings loaded successfully. Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not open webcam.")
    exit()

print("✅ Starting real-time face recognition. (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=1/RESIZE_FACTOR, fy=1/RESIZE_FACTOR)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect all faces in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
        name = "Unknown"

        # ⭐️ Improved Matching Logic: Use face_distance to find the best match ⭐️
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            
            # Check if the best match is within the tolerance
            if matches[best_match_index] and face_distances[best_match_index] < TOLERANCE:
                name = known_names[best_match_index]
        
        face_names.append(name)

    # Draw results on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= RESIZE_FACTOR
        right *= RESIZE_FACTOR
        bottom *= RESIZE_FACTOR
        left *= RESIZE_FACTOR

        # Set box color based on the result
        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Robust Face Recognition - By Tawan', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program terminated.")