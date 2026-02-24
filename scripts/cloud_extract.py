import os
import cv2
import sqlite3
import numpy as np
import shutil
import sys

# Link to your engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.face_engine import AttendanceEngine

# --- UPDATED PATH FOR YOUR EXTRACTED FILES ---
# Pointing directly to the deep folder you just created
MANUAL_PATH = r"C:\SMART_ATTENDANCE\data\temp_cloud\Smart_Attendance -20260223T052459Z-1-001\Smart_Attendance"
DB_PATH = r"db/face_embeddings.db"

def process_manual_extract():
    if not os.path.exists(MANUAL_PATH):
        print(f"❌ Path not found: {MANUAL_PATH}")
        return

    print("🚀 Initializing AI Engine...")
    engine = AttendanceEngine()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, roll_no TEXT UNIQUE, name TEXT, embedding BLOB)")

    print("\n🧬 Converting Manual Photos to 512-D Vectors...")
    success_count = 0

    # Look through the folders: 23151050_Shubharaj, 23151087_Pratik, etc.
    for folder in os.listdir(MANUAL_PATH):
        folder_path = os.path.join(MANUAL_PATH, folder)
        if not os.path.isdir(folder_path): continue

        try:
            roll, name = folder.split("_", 1)
        except ValueError: continue

        embeddings = []
        print(f"👤 Processing: {name}")

        for img_name in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, img_name))
            if img is None: continue
            
            # Extract 512-D math DNA
            faces = engine.detect_and_extract(img)
            if faces:
                embeddings.append(faces[0].embedding)

        if embeddings:
            # Average the math for a stable anchor
            mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
            cursor.execute("INSERT OR REPLACE INTO students (roll_no, name, embedding) VALUES (?, ?, ?)",
                           (roll, name, mean_emb.tobytes()))
            success_count += 1
            print(f"   ✅ Saved {len(embeddings)} images for {name}")

    conn.commit()
    conn.close()

    # THE PURGE: Deleting the whole temp_cloud folder to clean your disk
    print(f"\n🗑️ Deleting temporary folder: C:\\SMART_ATTENDANCE\\data\\temp_cloud")
    shutil.rmtree(r"C:\SMART_ATTENDANCE\data\temp_cloud")
    print("✨ PC is clean. Only the math DNA remains in SQL.")

if __name__ == "__main__":
    process_manual_extract()