# Third-party ML models

This project downloads the following pretrained ONNX models from the official OpenCV Zoo repository at first use:

- YuNet face detector: `models/face_detection_yunet/face_detection_yunet_2023mar.onnx` in OpenCV Zoo.
- SFace face recognizer: `models/face_recognition_sface/face_recognition_sface_2021dec.onnx` in OpenCV Zoo.

The project source code does not bundle these model binaries. Check the model-directory license/readme and confirm the allowed terms for the way you intend to use them before redistribution or commercial deployment.
