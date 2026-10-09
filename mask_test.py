import face_recognition
import cv2

image = face_recognition.load_image_file("criminals/balram_4.jpg")

# try stronger detection
face_locations = face_recognition.face_locations(image, model="cnn")

print("Faces detected:", len(face_locations))