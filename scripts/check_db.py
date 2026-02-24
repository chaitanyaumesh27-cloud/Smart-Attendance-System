import sqlite3
import numpy as np
import os

# Path to your database
DB_PATH = r"C:\SMART_ATTENDANCE\db\face_embeddings.db"

def check_database():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found at: {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get all students
        cursor.execute("SELECT roll_no, name, embedding FROM students")
        rows = cursor.fetchall()

        print(f"\n--- 🏛️ Database Audit: {len(rows)} Student(s) Found ---")
        
        for roll, name, blob in rows:
            # Convert the binary BLOB back into a numpy array (512 numbers)
            vector = np.frombuffer(blob, dtype=np.float32)
            
            print(f"📍 Roll: {roll} | Name: {name}")
            print(f"   🧬 Vector Stats: {len(vector)} dimensions")
            print(f"   🔢 Preview: {vector[:3]} ... {vector[-3:]}")
            print("-" * 40)

        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")

if __name__ == "__main__":
    check_database()