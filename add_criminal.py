import sqlite3

conn = sqlite3.connect("criminals.db")
cursor = conn.cursor()

cursor.execute("""
INSERT INTO criminals (name, crime, image_path)
VALUES (?, ?, ?)
""", ("John Doe", "Robbery", "criminals/sample.jpg"))

conn.commit()
conn.close()

print("Criminal added successfully.")
