import cv2
import face_recognition
import numpy as np
import pickle

# Load pre-encoded data (FAST)
print("Loading encoded data...")

try:
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    print("✅ Encodings loaded instantly!")
    print("Loaded names:", known_names)

except FileNotFoundError:
    print("❌ encodings.pkl not found! Run encode_faces.py first")
    exit()

# Start webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("❌ Could not access camera")
    exit()

# Masked faces usually need slightly higher tolerance
MATCH_TOLERANCE = 0.45

while True:
    ret, frame = video.read()
    if not ret:
        print("❌ Camera error")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces using CNN (better for masked faces)
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

    # Encode faces with num_jitters=5 for stability
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=5)

    for encoding, location in zip(face_encodings, face_locations):
        name = "Unknown"
        accuracy = 0.0

        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(distances)
            matches = face_recognition.compare_faces(
                known_encodings, encoding, tolerance=MATCH_TOLERANCE
            )

            if matches[best_match_index]:
                name = known_names[best_match_index]
                accuracy = (1 - distances[best_match_index]) * 100

        # Unpack face location
        top, right, bottom, left = location
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} ({accuracy:.2f}%)", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if name != "Unknown":
            print("\n🚨 Criminal Detected:", name)
            print("Accuracy:", round(accuracy, 2), "%")
            print("Distances:", distances)

    cv2.imshow("Smart Face Crime System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()