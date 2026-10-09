from database import load_criminal_database

print("Loading criminal database...")

encodings, names = load_criminal_database()

print("Loaded Names:", names)
print("Total Faces Loaded:", len(encodings))
