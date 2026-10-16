from __future__ import annotations
from pathlib import Path
import urllib.request
import threading
import cv2
import numpy as np

MODEL_SOURCES = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "minimum_bytes": 100_000,
    },
    "sface": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "minimum_bytes": 5_000_000,
    },
}

class FaceModelError(RuntimeError):
    pass

class FaceEngine:
    """OpenCV Zoo YuNet face detector + SFace face embedding model."""
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.detector = None
        self.recognizer = None
        self._load_lock = threading.RLock()
        self._infer_lock = threading.RLock()

    def status(self):
        return {key: {"filename": spec["filename"], "ready": (self.model_dir/spec["filename"]).is_file() and (self.model_dir/spec["filename"]).stat().st_size >= spec["minimum_bytes"]}
                for key, spec in MODEL_SOURCES.items()} | {"loaded": self.detector is not None and self.recognizer is not None}

    def _download(self, key):
        spec = MODEL_SOURCES[key]
        target = self.model_dir / spec["filename"]
        if target.exists() and target.stat().st_size >= spec["minimum_bytes"]:
            return target
        partial = target.with_suffix(target.suffix + ".download")
        try:
            req = urllib.request.Request(spec["url"], headers={"User-Agent": "FaceAttendPro/1.0 (local academic prototype)"})
            with urllib.request.urlopen(req, timeout=90) as response, open(partial, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk: break
                    out.write(chunk)
            size = partial.stat().st_size
            if size < spec["minimum_bytes"]:
                raise FaceModelError(f"Downloaded {spec['filename']} is only {size:,} bytes. Delete the incomplete file and retry. Model URL: {spec['url']}")
            partial.replace(target)
            return target
        except FaceModelError:
            partial.unlink(missing_ok=True)
            raise
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise FaceModelError(
                f"Could not download {spec['filename']}: {exc}. Check internet access and download it manually from {spec['url']} into {self.model_dir}."
            ) from exc

    def load(self):
        with self._load_lock:
            if self.detector is not None and self.recognizer is not None:
                return
            if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
                raise FaceModelError("OpenCV face APIs are missing. Install requirements.txt; use opencv-contrib-python, not multiple conflicting OpenCV packages.")
            det_path = self._download("yunet")
            rec_path = self._download("sface")
            try:
                self.detector = cv2.FaceDetectorYN_create(str(det_path), "", (320, 320), 0.85, 0.30, 5000)
                self.recognizer = cv2.FaceRecognizerSF_create(str(rec_path), "")
            except Exception as exc:
                self.detector = self.recognizer = None
                raise FaceModelError(f"OpenCV could not load a model. Confirm both ONNX files are complete and compatible. Details: {exc}") from exc

    def detect(self, image):
        self.load()
        if image is None or image.size == 0 or image.ndim != 3:
            raise ValueError("Invalid image. Capture a new webcam frame and try again.")
        h, w = image.shape[:2]
        with self._infer_lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
        if faces is None:
            return []
        return sorted([np.asarray(row, dtype=np.float32).copy() for row in faces], key=lambda f: float(f[2]*f[3]), reverse=True)

    def embedding(self, image, face):
        self.load()
        with self._infer_lock:
            aligned = self.recognizer.alignCrop(image, face)
            feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(feature).all() or norm < 1e-8:
            raise ValueError("The face embedding was invalid; improve lighting and capture the face again.")
        return feature / norm
