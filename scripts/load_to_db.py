import sqlite3
import json
import numpy as np

FILE_PATH = r"C:\SMART_ATTENDANCE\data\processed_faces.json"
DB_PATH = r"C:\SMART_ATTENDANCE\db\face_embeddings.db"

def load_to_sql():
    # 1. Read the file
    with open(FILE_PATH, 'r') as f:
        data = json.load(f)

    # 2. Connect to SQL
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, roll_no TEXT UNIQUE, name TEXT, embedding BLOB)")

    print(f"📥 Loading {len(data)} student(s) from file to database...")

    for student in data:
        # Convert the list of numbers back to binary bytes for the DB
        vector_bytes = np.array(student['vector'], dtype=np.float32).tobytes()
        
        cursor.execute(
            "INSERT OR REPLACE INTO students (roll_no, name, embedding) VALUES (?, ?, ?)",
            (student['roll_no'], student['name'], vector_bytes)
        )
        print(f"✅ Migrated: {student['name']}")

    conn.commit()
    conn.close()
    print("\n🎉 Database is now fully synchronized with the file!")

if __name__ == "__main__":
    load_to_sql()