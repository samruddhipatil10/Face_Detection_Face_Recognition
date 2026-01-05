import sqlite3
import os

os.makedirs("database", exist_ok=True)

conn = sqlite3.connect("database/face_recognition.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")

conn.commit()
conn.close()

print("✅ Database and table created successfully")
