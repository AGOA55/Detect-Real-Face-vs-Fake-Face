import face_recognition
import os
import pickle
import numpy as np

# --- ⚙️ Configurations ⚙️ ---
# Path to the parent folder containing subfolders of individuals.
TRAIN_DATA_DIR = r"E:\Project By Tawan\dectect face\train"

# The output file where the generated encodings will be stored.
ENCODINGS_OUTPUT_FILE = "encodings.pkl"

# --- Main Logic ---
print("Starting to process faces...")
known_encodings = []
known_names = []

# Iterate through each person's folder in the training directory
for person_name in os.listdir(TRAIN_DATA_DIR):
    person_dir = os.path.join(TRAIN_DATA_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"Processing person: {person_name}")
    person_encodings = []

    # Iterate through each image of the person
    for filename in os.listdir(person_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(person_dir, filename)
            
            try:
                # Load image and find face encodings
                # Using the 'large' model can provide better accuracy.
                image = face_recognition.load_image_file(file_path)
                # Use model="cnn" for higher accuracy, but it's slower and needs dlib with CUDA support for speed.
                # 'hog' is faster and works well on CPU.
                face_locations = face_recognition.face_locations(image, model="hog")

                if face_locations:
                    # We assume there's only one primary face per training image
                    encoding = face_recognition.face_encodings(image, face_locations, num_jitters=5)[0]
                    person_encodings.append(encoding)
                    print(f"  -> Encoded {filename}")
                else:
                    print(f"  -> ⚠️ Warning: No face found in {filename}. Skipping.")
            except Exception as e:
                print(f"  -> ❌ Error processing {filename}: {e}")
    
    # After processing all images for a person, calculate the average encoding
    if person_encodings:
        # Calculate the mean of all found encodings for this person
        average_encoding = np.mean(person_encodings, axis=0)
        known_encodings.append(average_encoding)
        known_names.append(person_name)
        print(f"✅ Created a robust 'Master Encoding' for {person_name} from {len(person_encodings)} images.\n")

# Save the generated encodings to a file
print(f"Saving encodings to file: {ENCODINGS_OUTPUT_FILE}")
with open(ENCODINGS_OUTPUT_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)


print("\n🎉 All done! Encodings have been saved successfully.")
