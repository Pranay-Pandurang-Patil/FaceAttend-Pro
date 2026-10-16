"""Download the official OpenCV Zoo YuNet and SFace ONNX model files."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_attendance.recognition import FaceEngine
if __name__ == "__main__":
    engine = FaceEngine(Path(__file__).resolve().parent)
    engine.load()
    print("Both face models downloaded and loaded successfully.")
    print(engine.status())
