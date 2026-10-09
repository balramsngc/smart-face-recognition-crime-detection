from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
from face_loader import load_criminal_faces

app = Flask(__name__)

# Load criminal database
known_encodings, known_names = load_criminal_faces()

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for encoding, location in zip(face_encodings, face_locations):

            name = "Unknown"

            if len(known_encodings) > 0:
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.75:
                    name = known_names[best_match_index]

            top, right, bottom, left = location

            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

            cv2.putText(frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)