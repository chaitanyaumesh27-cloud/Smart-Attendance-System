import cv2
import sqlite3
import numpy as np
import os
from datetime import datetime
from engine.face_engine import AttendanceEngine

# --- CONFIGURATION ---
DB_PATH = r"db/face_embeddings.db"
# Since your image is in 'data/', we point directly there
DATA_DIR = r"data"
OUTPUT_DIR = r"data/output_results"
LOG_FILE = "Daily_Attendance.csv"
THRESHOLD = 0.55  # Match sensitivity

def load_vault():
    """Fetches 512-D vectors from your SQL vault."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT roll_no, name, embedding FROM students")
    # Convert binary back to numpy arrays
    data = [{"roll": r, "name": n, "emb": np.frombuffer(e, dtype=np.float32)} 
            for r, n, e in cursor.fetchall()]
    conn.close()
    return data

def cosine_similarity(v1, v2):
    """Measures how closely two faces match."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def log_attendance(roll, name):
    """Appends to the CSV log."""
    now = datetime.now()
    dt, tm = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if header: f.write("Date,Time,Roll_No,Name\n")
        f.write(f"{dt},{tm},{roll},{name}\n")

def process_image(filename):
    print("🔍 Loading AI Engine...")
    engine = AttendanceEngine()
    student_vault = load_vault()
    
    # Target path: C:\SMART_ATTENDANCE\data\test.jpg
    img_path = os.path.join(DATA_DIR, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Error: Could not find '{filename}' in {DATA_DIR}")
        return

    # Scan the image for faces
    faces = engine.detect_and_extract(img)
    print(f"🚀 AI Found {len(faces)} face(s). Comparing with SQL DNA...")

    for face in faces:
        best_match, max_sim = None, -1.0
        for student in student_vault:
            sim = cosine_similarity(face.embedding, student['emb'])
            if sim > max_sim:
                max_sim, best_match = sim, student

        if best_match and max_sim > THRESHOLD:
            name, roll = best_match['name'], best_match['roll']
            label, color = f"{name} ({max_sim*100:.1f}%)", (0, 255, 0)
            log_attendance(roll, name) #
            print(f"✅ Recognized: {name}")
        else:
            label, color = "Unknown", (0, 0, 255)

        # Draw box and name
        box = face.bbox.astype(int)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
        cv2.putText(img, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
    cv2.imwrite(out_path, img)
    print(f"✨ DONE! Labeled result saved to: {out_path}")

if __name__ == "__main__":
    # Ensure this matches your file in the 'data' folder
    TEST_IMAGE = "test.jpg" 
    process_image(TEST_IMAGE)