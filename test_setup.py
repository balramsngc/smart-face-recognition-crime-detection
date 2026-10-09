import face_recognition
from PIL import Image, ImageDraw

image = face_recognition.load_image_file("sample.jpg")
face_locations = face_recognition.face_locations(image)

print(f"Found {len(face_locations)} face(s).")

pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left) in face_locations:
    draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)

pil_image.show()
