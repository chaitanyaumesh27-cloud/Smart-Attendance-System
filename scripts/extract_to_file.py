import os
import cv2
import json
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.face_engine import AttendanceEngine

DATASET_DIR = r"C:\SMART_ATTENDANCE\data\enrollment"
OUTPUT_FILE = r"C:\SMART_ATTENDANCE\data\processed_faces.json"
SUMMARY_FILE = r"C:\SMART_ATTENDANCE\data\face_summary.txt"

print("🔍 Extracting Face DNA...")
engine = AttendanceEngine()
extracted_data = []

# Added encoding='utf-8' here to fix the UnicodeEncodeError
with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    f.write("--- 🧬 FACE DNA SUMMARY REPORT ---\n\n")

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path): continue

    try:
        roll_no, name = folder.split("_", 1)
    except ValueError: continue

    embeddings = []
    print(f"👤 Processing: {name}")

    for img_name in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, img_name))
        if img is None: continue
        
        faces = engine.detect_and_extract(img)
        if faces:
            embeddings.append(faces[0].embedding)

    if embeddings:
        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        
        extracted_data.append({
            "roll_no": roll_no,
            "name": name,
            "vector": mean_emb.tolist()
        })

        # Added encoding='utf-8' here as well
        with open(SUMMARY_FILE, 'a', encoding='utf-8') as f:
            f.write(f"STUDENT: {name} (Roll: {roll_no})\n")
            f.write(f"DIMENSIONS: {len(mean_emb)}\n")
            f.write(f"VECTOR SNIPPET: {mean_emb[:5]} ... {mean_emb[-5:]}\n")
            f.write("-" * 40 + "\n")

# Save the big JSON
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f)

print(f"\n✅ FULL DATA saved to: {OUTPUT_FILE}")
print(f"✅ READABLE SUMMARY saved to: {SUMMARY_FILE}")