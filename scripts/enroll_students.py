import os
import cv2
import chromadb
import sys

# Crucial: Tells the script to look for the 'engine' folder in the main directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.face_engine import AttendanceEngine

def run_enrollment():
    print("🚀 Initializing AI Engine...")
    engine = AttendanceEngine()
    
    # Setup Memory with CORRECT math (Cosine Similarity)
    client = chromadb.PersistentClient(path="db/chromadb")
    
    # We use get_or_create, but specifying the metadata ensures the math is correct
    collection = client.get_or_create_collection(
        name="students",
        metadata={"hnsw:space": "cosine"} # <--- This fixes the 'Distance: 800+' issue
    )

    base_path = "data/enrollment"
    
    if not os.path.exists(base_path):
        print(f"❌ Error: {base_path} folder not found!")
        return

    for student_name in os.listdir(base_path):
        student_dir = os.path.join(base_path, student_name)
        if not os.path.isdir(student_dir): continue
        
        print(f"\n👤 Enrolling: {student_name}")
        
        for img_name in os.listdir(student_dir):
            img_path = os.path.join(student_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None: 
                print(f"   ⚠️ Could not read {img_name}")
                continue

            # Extract math vector (The "Fingerprint")
            faces = engine.detect_and_extract(img)
            
            if faces:
                # We take the largest face detected in the photo
                face = faces[0] 
                collection.add(
                    embeddings=[face.embedding.tolist()],
                    metadatas=[{"name": student_name}],
                    ids=[f"{student_name}_{img_name}"]
                )
                print(f"   ✅ Added {img_name}")
            else:
                print(f"   ⚠️ No face detected in {img_name}")

    print(f"\n✨ Done! Total faces in Database: {collection.count()}")

if __name__ == "__main__":
    run_enrollment()