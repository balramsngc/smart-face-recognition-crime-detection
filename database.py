import sqlite3

conn = sqlite3.connect("criminals.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS criminals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    crime TEXT,
    image_path TEXT
)
""")

conn.commit()
conn.close()

print("Criminal database created successfully.")
