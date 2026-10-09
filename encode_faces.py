import face_recognition
import os
import cv2
import pickle

criminals_folder = "criminals"

known_encodings = []
known_names = []

print("Encoding faces...")

for filename in os.listdir(criminals_folder):

    if filename.endswith(".jpg") or filename.endswith(".png"):

        path = os.path.join(criminals_folder, filename)

        image = face_recognition.load_image_file(path)
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)

        face_locations = face_recognition.face_locations(image, model="hog")

        encodings = face_recognition.face_encodings(image, face_locations)

        name = filename.split("_")[0]

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

        print("Encoded:", filename)

# Save encodings
data = {"encodings": known_encodings, "names": known_names}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Encoding complete. Saved to encodings.pkl")