import insightface
from insightface.app import FaceAnalysis
import cv2

class AttendanceEngine:
    def __init__(self, model_name='buffalo_l'):
        """
        Initializes RetinaFace (Detection) and ArcFace (Recognition).
        'buffalo_l' is the high-accuracy model pack.
        """
        # We use CPU for now to ensure it works on all laptops
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        
        # Prepare the model with a standard detection resolution
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("🚀 AI Engine is Loaded and Ready!")

    def detect_and_extract(self, frame):
        """Finds faces and generates 512-dim math vectors."""
        faces = self.app.get(frame)
        return faces