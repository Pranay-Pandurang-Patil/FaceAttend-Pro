# FaceAttend Pro — Student Face Attendance System

A local-first academic prototype for enrolling consenting students from live webcam captures and recording attendance by matching face embeddings against enrolled references. It includes a polished dashboard, multiple-reference enrollment, attendance history, and CSV export.

## What the ML does

- **YuNet** detects faces in a camera image.
- **SFace** aligns a detected face and creates a numeric face embedding.
- During enrollment, the app captures 3–5 webcam frames and stores their embeddings in SQLite (not the original photos).
- During an attendance scan, the app detects every visible face, compares its embedding to enrolled templates using cosine similarity, and records a student only when the best similarity clears the configured threshold and is sufficiently separated from the second-best student's score.
- A student is marked at most once per local calendar date. Unknown and ambiguous faces are not marked.

The model weights are pretrained. The app does **not** retrain the face-recognition neural network for every student. Instead, enrollment creates database reference embeddings for identity matching; this is the appropriate approach for a roster that changes over time.

## Features

- Live webcam enrollment, 3–5 varied reference captures per student.
- Live webcam attendance snapshots; multiple faces may be matched in one frame.
- Adjustable cosine-similarity threshold (default 0.42) and conservative ambiguity handling.
- Annotated result preview: green = marked/recognized, amber = already present or needs review, red = unknown / not marked.
- SQLite student, reference-template, and attendance storage.
- Student archive immediately deletes that student's biometric templates while retaining historical attendance rows.
- Date-filtered attendance table and CSV export.
- Local-host-only Flask server, avoiding uploading face data to a cloud service.
- Unit tests for storage, idempotent attendance, validation, image decoding, matching, ambiguity rejection, and annotations.

## Windows setup

Use Python 3.11 if available. Extract this folder and open it in VS Code. In Command Prompt or PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If `py -3.11` is unavailable, use `py -m venv .venv`.

Open **http://127.0.0.1:5000**. Grant browser camera access. The site should be opened through `localhost` because browsers restrict webcam access on non-secure origins.

### First run — download model files

Click **Check / download models** in the sidebar. The first download is about 40 MB and requires internet access. The application can also download lazily during enrollment/scan. The two ONNX files are stored in `models/`:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

If automatic download is blocked, download the official files manually from the OpenCV Zoo URLs printed in `face_attendance/recognition.py` and place them in `models/`. Do not leave Git LFS pointer files in place of the actual ONNX weights; the app checks minimum file sizes.

If OpenCV face APIs are missing, avoid mixing multiple OpenCV wheels. In the active venv, run:

```powershell
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
pip install -r requirements.txt
```

## How to use

1. Start the server and open `http://127.0.0.1:5000`.
2. Click **Check / download models**, wait for success.
3. Open **Enroll students** and enter a unique student ID, name, and course.
4. Start the webcam. Capture at least 3 clear views: facing forward, a slight turn left, and a slight turn right. A capture must contain exactly one sufficiently large face. Check the consent confirmation and save.
5. Open **Live attendance**, start the camera, choose a matching threshold, and click **Scan & mark attendance**. The processed frame is shown with per-face status.
6. Check **Attendance records**, select a date, and export a CSV.
7. To remove biometric reference data, archive a student from the enrolled-student list. Historical attendance rows remain for reporting.

## Project structure

```text
FaceAttendPro/
├── app.py                         # Flask endpoints and app startup
├── requirements.txt
├── run_windows.bat                # Optional one-click environment setup
├── face_attendance/
│   ├── database.py                # SQLite schema and persistence
│   ├── recognition.py             # YuNet/SFace model loading and inference
│   └── service.py                 # input validation, embeddings and matching
├── models/
│   └── download_models.py         # fetch and validate model weights
├── templates/index.html           # dashboard, enrollment, camera and reports UI
├── static/style.css               # professional UI styles
├── static/app.js                  # camera, captures, scan flow and reports
├── tests/
└── data/                          # SQLite database created at runtime
```

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The test suite intentionally uses synthetic embeddings for deterministic matching/storage tests; it does not pretend to prove real-world face recognition accuracy. For deployment validation, test the full webcam workflow on the target machine and a consented validation set that includes different lighting, angles, and people not enrolled in the roster.

## Threshold and accuracy

The default threshold (0.42) is a starting point, not a universal accuracy guarantee. A higher threshold generally reduces false matches but increases false rejections. Calibrate using a held-out, consented test set from your own camera conditions. Track false acceptance and false rejection rates; do not use training/enrollment images as your only evaluation set.

## Privacy, security, and limitations

- Use only with informed consent and according to institution policy. For minors, follow applicable guardian/institution requirements.
- This is an academic prototype, not a production attendance or access-control system. It has no anti-spoof/liveness detection; a photo or screen shown to the camera may fool it. Keep a teacher present and provide a manual correction path.
- Face similarity can be wrong because of pose, lighting, image quality, camera quality, or model limitations. Never treat the result as definitive without appropriate human oversight.
- The server binds only to `127.0.0.1`. Do not expose it to a public/LAN network without real authentication, TLS, access controls, retention rules and a security review.
- The app stores face embeddings in local SQLite. Embeddings are biometric data too. Protect the machine, back up carefully, and archive students when their data should be removed.
- This repository does not bundle the third-party ONNX weights. Review the licenses/terms in the official [OpenCV Zoo](https://github.com/opencv/opencv_zoo) model directories before use beyond permitted academic/personal testing.

## Troubleshooting

- **Camera button does nothing:** open `http://127.0.0.1:5000`, grant camera permission in the browser, and make sure another app is not holding the webcam.
- **Could not download models:** check internet/firewall; see the printed official model URLs and download into `models/`.
- **No face detected:** use good, front lighting, move closer, and remove extreme blur/backlighting.
- **Enrollment says multiple faces:** ensure only one person appears in each capture.
- **Student is unknown:** retake varied enrollment frames, inspect camera lighting, and calibrate the threshold on held-out consented images.
- **False matches:** raise the threshold and review the second-best margin; do not simply lower the threshold to force a match.
- **Attendance recorded twice:** the database enforces one record per student per local date. Repeat scans return “already present”.
