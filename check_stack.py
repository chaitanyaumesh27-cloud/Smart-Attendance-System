import insightface
import onnxruntime as ort
import chromadb
import cv2
import sys

print(f"--- 🛠️ System Health Check ---")
print(f"Python Version: {sys.version}")

# 1. Check InsightFace & ONNX
try:
    print(f"✅ InsightFace Version: {insightface.__version__}")
    print(f"✅ ONNX Runtime Device: {ort.get_device()}") 
    # This will say 'GPU' if your NVIDIA drivers are correct, or 'CPU' otherwise.
except Exception as e:
    print(f"❌ InsightFace/ONNX Error: {e}")

# 2. Check ChromaDB (The Memory)
try:
    client = chromadb.PersistentClient(path="db/check_test")
    print(f"✅ ChromaDB: Ready and persistent.")
except Exception as e:
    print(f"❌ ChromaDB Error: {e}")

# 3. Check OpenCV (The Eyes)
try:
    print(f"✅ OpenCV Version: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV Error: {e}")

print(f"------------------------------")