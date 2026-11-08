# FaceAttend Pro — Student Face Attendance System

A local-first academic prototype for enrolling consenting students from live webcam captures and recording attendance by matching face embeddings against enrolled references. It includes a polished dashboard, multiple-reference enrollment, attendance history, and CSV export.

---

## Table of Contents

- [Overview](#overview)
- [What the ML Does](#what-the-ml-does)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [First Run — Download Model Files](#first-run--download-model-files)
- [How to Use](#how-to-use)
- [Run Tests](#run-tests)
- [Threshold and Accuracy](#threshold-and-accuracy)
- [Privacy, Security, and Limitations](#privacy-security-and-limitations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

FaceAttend Pro is a face-recognition-based student attendance management system designed as a local-first academic prototype.

The application uses live webcam captures to enroll consenting students and subsequently recognize them during attendance sessions.

It uses pretrained face detection and recognition models to generate numerical face embeddings, which are stored locally in an SQLite database for identity matching.

The system supports multiple student references, multi-face attendance scanning, attendance history, and CSV export.

The application runs locally using Flask and does not upload face data to a cloud service.

## What the ML Does

The application uses two pretrained models:

- **YuNet:** Detects faces in a camera image.
- **SFace:** Aligns a detected face and creates a numeric face embedding.

### Enrollment Process

1. The application captures 3–5 webcam frames for each student.
2. Each frame is processed using YuNet and SFace.
3. The generated face embeddings are stored in SQLite.
4. Original enrollment photographs are not stored.

### Attendance Recognition Process

1. The application captures a live webcam image.
2. YuNet detects every visible face.
3. SFace generates an embedding for each detected face.
4. The generated embedding is compared against enrolled templates using cosine similarity.
5. A student is recognized only when the best similarity clears the configured threshold and is sufficiently separated from the second-best student's score.
6. Attendance is recorded for successfully recognized students.

A student is marked at most once per local calendar date.

Unknown and ambiguous faces are not marked.

**Important:** The model weights are pretrained. The application does **not** retrain the face-recognition neural network for every student. Instead, enrollment creates database reference embeddings for identity matching; this is the appropriate approach for a roster that changes over time.

---

## Features

- Live webcam enrollment with 3–5 varied reference captures per student.
- Live webcam attendance snapshots.
- Multiple faces can be matched in one frame.
- Adjustable cosine-similarity threshold (default 0.42).
- Conservative ambiguity handling.
- Annotated attendance result preview:
  - 🟢 Green: Marked / recognized.
  - 🟠 Amber: Already present or needs review.
  - 🔴 Red: Unknown / not marked.
- SQLite student, reference-template, and attendance storage.
- Student archive immediately deletes that student's biometric templates while retaining historical attendance rows.
- Date-filtered attendance table.
- CSV attendance export.
- Local-host-only Flask server, avoiding uploading face data to a cloud service.
- Unit tests for:
  - Storage.
  - Idempotent attendance.
  - Validation.
  - Image decoding.
  - Matching.
  - Ambiguity rejection.
  - Annotations.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Flask | Backend and local web server |
| OpenCV | Image processing and face recognition |
| YuNet | Face detection |
| SFace | Face alignment and embedding generation |
| ONNX | Pretrained model format |
| SQLite | Local database storage |
| HTML | Dashboard structure |
| CSS | Dashboard styling |
| JavaScript | Webcam integration and frontend interactions |
| Unittest | Automated testing |

---

## Project Structure

```text
FaceAttendPro/
├── app.py                         # Flask endpoints and app startup
├── requirements.txt
├── run_windows.bat                # Optional one-click environment setup
├── face_attendance/
│   ├── database.py                # SQLite schema and persistence
│   ├── recognition.py             # YuNet/SFace model loading and inference
│   └── service.py                 # Input validation, embeddings and matching
├── models/
│   └── download_models.py         # Fetch and validate model weights
├── templates/
│   └── index.html                 # Dashboard, enrollment, camera and reports UI
├── static/
│   ├── style.css                  # Professional UI styles
│   └── app.js                     # Camera, captures, scan flow and reports
├── tests/                         # Unit tests
└── data/                          # SQLite database created at runtime
```

---

## Installation and Setup

### Prerequisites

- Windows 10/11.
- Python 3.11 (recommended).
- VS Code or another Python-compatible IDE.
- Webcam.
- Internet connection for the initial model download.

### Step 1: Clone or Extract the Repository

Extract the project folder and open it in VS Code.

### Step 2: Create a Virtual Environment

Open Command Prompt or PowerShell inside the project directory.

```powershell
py -3.11 -m venv .venv
```

If `py -3.11` is unavailable, use:

```powershell
py -m venv .venv
```

### Step 3: Activate the Virtual Environment

```powershell
.venv\Scripts\activate
```

### Step 4: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 5: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 6: Run the Application

```powershell
python app.py
```

### Step 7: Open the Application

Open the following address in your browser:

**http://127.0.0.1:5000**

Grant browser camera access when prompted.

The site should be opened through `localhost` because browsers restrict webcam access on non-secure origins.

---

## First Run — Download Model Files

Click **Check / download models** in the sidebar.

The first download is about 40 MB and requires internet access.

The application can also download models lazily during enrollment or attendance scanning.

The two ONNX files are stored in the `models/` directory:

```text
models/
├── face_detection_yunet_2023mar.onnx
└── face_recognition_sface_2021dec.onnx
```

If automatic download is blocked:

1. Download the official files manually from the OpenCV Zoo URLs printed in `face_attendance/recognition.py`.
2. Place the downloaded files inside the `models/` directory.
3. Ensure the files are actual ONNX model weights.

**Note:** Do not leave Git LFS pointer files in place of the actual ONNX weights; the app checks minimum file sizes.

### OpenCV Face API Troubleshooting

If OpenCV face APIs are missing, avoid mixing multiple OpenCV wheels.

In the active virtual environment, run:

```powershell
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
pip install -r requirements.txt
```

---

## How to Use

### 1. Start the Application

Start the Flask server and open:

```text
http://127.0.0.1:5000
```

### 2. Download Models

Click **Check / download models** and wait for successful model initialization.

### 3. Enroll Students

Open **Enroll students** and enter:

- Unique student ID.
- Student name.
- Course.

### 4. Capture Student References

Start the webcam.

Capture at least 3 clear views:

- Facing forward.
- Slight turn left.
- Slight turn right.

A capture must contain exactly one sufficiently large face.

Check the consent confirmation and save.

### 5. Mark Live Attendance

Open **Live attendance**.

1. Start the camera.
2. Choose a matching threshold.
3. Click **Scan & mark attendance**.
4. Review the processed frame with per-face status annotations.

### 6. View Attendance Records

Open **Attendance records**.

Select a date to view attendance records and export the data as a CSV file.

### 7. Archive Students

To remove biometric reference data, archive a student from the enrolled-student list.

Historical attendance rows remain available for reporting.

---

## Run Tests

The project includes unit tests for storage, attendance handling, validation, image decoding, matching, ambiguity rejection, and annotations.

Run the complete test suite using:

```powershell
python -m unittest discover -s tests -v
```

The test suite intentionally uses synthetic embeddings for deterministic matching/storage tests; it does not pretend to prove real-world face recognition accuracy.

For deployment validation, test the full webcam workflow on the target machine and a consented validation set that includes different lighting, angles, and people not enrolled in the roster.

---

## Threshold and Accuracy

The default cosine-similarity threshold is:

```text
0.42
```

This is a starting point, not a universal accuracy guarantee.

A higher threshold generally:

- Reduces false matches.
- Increases false rejections.

A lower threshold generally:

- Makes recognition less restrictive.
- Can increase false matches.

Calibrate using a held-out, consented test set from your own camera conditions.

Track:

- False Acceptance Rate (FAR).
- False Rejection Rate (FRR).

Do not use training/enrollment images as your only evaluation set.

---

## Privacy, Security, and Limitations

- Use only with informed consent and according to institution policy. For minors, follow applicable guardian/institution requirements.
- This is an academic prototype, not a production attendance or access-control system.
- It has no anti-spoof/liveness detection; a photo or screen shown to the camera may fool it. Keep a teacher present and provide a manual correction path.
- Face similarity can be wrong because of pose, lighting, image quality, camera quality, or model limitations. Never treat the result as definitive without appropriate human oversight.
- The server binds only to `127.0.0.1`. Do not expose it to a public/LAN network without real authentication, TLS, access controls, retention rules and a security review.
- The app stores face embeddings in local SQLite. Embeddings are biometric data too. Protect the machine, back up carefully, and archive students when their data should be removed.
- This repository does not bundle the third-party ONNX weights. Review the licenses/terms in the official [OpenCV Zoo](https://github.com/opencv/opencv_zoo) model directories before use beyond permitted academic/personal testing.

---

## Troubleshooting

| Problem | Possible Solution |
|---|---|
| Camera button does nothing | Open `http://127.0.0.1:5000`, grant camera permission in the browser, and make sure another app is not holding the webcam. |
| Could not download models | Check internet/firewall; see the printed official model URLs and download into `models/`. |
| No face detected | Use good, front lighting, move closer, and remove extreme blur/backlighting. |
| Enrollment says multiple faces | Ensure only one person appears in each capture. |
| Student is unknown | Retake varied enrollment frames, inspect camera lighting, and calibrate the threshold on held-out consented images. |
| False matches | Raise the threshold and review the second-best margin; do not simply lower the threshold to force a match. |
| Attendance recorded twice | The database enforces one record per student per local date. Repeat scans return "already present". |

---

## License

License information has not yet been specified.

Add the appropriate license to the repository before distributing the project.

---

## Disclaimer

FaceAttend Pro is developed as an academic prototype to demonstrate the practical application of computer vision, machine learning, face embeddings, and database-based attendance management.

It is intended for educational and controlled testing purposes with informed consent and appropriate human oversight.
