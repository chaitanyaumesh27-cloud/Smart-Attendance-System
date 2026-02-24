# Smart Classroom Attendance System 🎓🤖

An automated attendance system leveraging Deep Learning and Computer Vision to identify students via face recognition and log attendance in real-time.

## 🚀 Overview
This project replaces traditional manual attendance with a "Smart Classroom" approach. It uses a **Face DNA** extraction method, converting facial features into 512-dimensional embeddings for high accuracy and privacy-conscious storage.

### Key Features:
* **Deep Learning Pipeline:** Utilizes Mediapipe/OpenCV for face detection and feature extraction.
* **Storage Efficient:** Does not store raw images locally. It converts faces into mathematical "embeddings" stored in a local SQLite database.
* **Cloud Integration:** Dynamically pulls enrollment photos from Google Drive for processing.

---

## 🏗️ Project Structure
* **engine/**: Core AI logic (Face detection & embedding extraction)
* **scripts/**: Utility scripts (Cloud sync, DB initialization)
* **db/**: Local SQLite database (face_embeddings.db)
* **run_attendance.py**: Main execution script

---

## 🛠️ Installation & Setup
1. **Clone the repository:**
   `git clone git@github.com:chaitanyaumesh27-cloud/Smart-Attendance-System.git`
2. **Install Dependencies:**
   `pip install -r requirements.txt`
3. **Run Enrollment:**
   `python scripts/enroll_students.py`

---

## 📝 Author
* **Chaitanya** - *College Project 2026*
