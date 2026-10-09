from face_loader import load_criminal_faces

encodings, names = load_criminal_faces()

print("Loaded Faces:", names)
print("Total Faces Loaded:", len(names))