import face_recognition
import os
import cv2

def load_criminal_faces():

    criminals_folder = "criminals"

    known_encodings = []
    known_names = []

    print("Loading criminal images...")

    for filename in os.listdir(criminals_folder):

        if filename.endswith(".jpg") or filename.endswith(".png"):

            image_path = os.path.join(criminals_folder, filename)

            image = face_recognition.load_image_file(image_path)

            # Resize image (improves detection)
            image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)

            # Stronger detection
            face_locations = face_recognition.face_locations(image, model="cnn")

            if len(face_locations) == 0:
                print("No face found in", filename)
                continue

            encodings = face_recognition.face_encodings(image, face_locations)

            if len(encodings) > 0:

                known_encodings.append(encodings[0])

                name = filename.split("_")[0]
                known_names.append(name)

                print("Loaded:", filename)

    print("Total Faces Loaded:", len(known_encodings))

    return known_encodings, known_names