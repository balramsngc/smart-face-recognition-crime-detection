import face_recognition
import numpy as np
from face_loader import load_criminal_faces

print("Loading criminal database...")

known_encodings, known_names = load_criminal_faces()

print("Total Faces Loaded:", len(known_names))

# Load test image
print("\nChecking test image...")

test_image = face_recognition.load_image_file("test.jpg")

# detect face
face_locations = face_recognition.face_locations(test_image, model="hog")

if len(face_locations) == 0:
    print("❌ No face found in test image")
    exit()

# encode detected face
test_encodings = face_recognition.face_encodings(test_image, face_locations)

for test_encoding in test_encodings:

    # compare with database
    distances = face_recognition.face_distance(known_encodings, test_encoding)

    best_match_index = np.argmin(distances)

    distance = distances[best_match_index]

    # mask friendly threshold
    if distance < 0.70:

        name = known_names[best_match_index]

        accuracy = (1 - distance) * 100

        print("\n🚨 Criminal Match Found!")
        print("Name:", name)
        print("Accuracy:", round(accuracy, 2), "%")

    else:

        print("\n✅ No criminal match found")