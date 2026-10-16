from __future__ import annotations
import base64, csv, io, json, re
from datetime import date
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response
from face_attendance.database import Database
from face_attendance.recognition import FaceEngine, FaceModelError
from face_attendance.service import decode_image_data_url, draw_results, match_face, validate_student_id

ROOT = Path(__file__).resolve().parent
INSTANCE = ROOT / "data"
INSTANCE.mkdir(exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
DB = Database(INSTANCE / "attendance.sqlite3")
ENGINE = FaceEngine(ROOT / "models")
DEFAULT_THRESHOLD = 0.42

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "app": "FaceAttend Pro", "models": ENGINE.status()})

@app.post("/api/model-check")
def model_check():
    try:
        ENGINE.load()
        return jsonify({"ok": True, "models": ENGINE.status(), "message": "Face detection and recognition models are ready."})
    except FaceModelError as exc:
        return jsonify({"ok": False, "error": str(exc), "models": ENGINE.status()}), 503
    except Exception as exc:
        app.logger.exception("Could not initialize face models")
        return jsonify({"ok": False, "error": f"Model initialization failed: {exc}"}), 500

@app.get("/api/summary")
def summary():
    return jsonify(DB.summary())

@app.get("/api/students")
def students():
    return jsonify({"students": DB.list_students(active_only=True)})

@app.post("/api/students")
def enroll_student():
    payload = request.get_json(silent=True) or {}
    student_id = validate_student_id(payload.get("student_id", ""))
    full_name = str(payload.get("full_name", "")).strip()
    class_name = str(payload.get("class_name", "")).strip()
    photos = payload.get("photos", [])
    consent = payload.get("consent") is True
    if not student_id:
        return jsonify(error="Student ID may contain letters, numbers, hyphens and underscores (2–32 characters)."), 400
    if not full_name or len(full_name) > 100:
        return jsonify(error="Enter a student name up to 100 characters."), 400
    if len(class_name) > 100:
        return jsonify(error="Class/course must be 100 characters or fewer."), 400
    if not consent:
        return jsonify(error="Confirm that the student has been informed and has consented to face-based attendance."), 400
    if not isinstance(photos, list) or not 3 <= len(photos) <= 5:
        return jsonify(error="Capture 3 to 5 clear webcam photos: front, slight left, and slight right."), 400
    try:
        ENGINE.load()
        templates = []
        for index, data_url in enumerate(photos, start=1):
            image = decode_image_data_url(data_url)
            faces = ENGINE.detect(image)
            if len(faces) != 1:
                return jsonify(error=f"Capture {index}: expected exactly one face, but detected {len(faces)}. Retake it with one face centered in the frame."), 400
            face = faces[0]
            x, y, w, h = map(int, face[:4])
            if min(w, h) < 75:
                return jsonify(error=f"Capture {index}: face is too small. Move closer to the camera and retake it."), 400
            if float(face[14]) < 0.75:
                return jsonify(error=f"Capture {index}: face detection confidence is low. Improve lighting and retake it."), 400
            templates.append({"embedding": ENGINE.embedding(image, face), "quality": float(face[14])})
        DB.add_student(student_id, full_name, class_name, templates)
        return jsonify({"ok": True, "message": f"{full_name} enrolled with {len(templates)} reference templates.", "student": DB.get_student(student_id)}), 201
    except FaceModelError as exc:
        return jsonify(error=str(exc)), 503
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            return jsonify(error=f"Student ID {student_id} already exists. Use a unique ID."), 409
        app.logger.exception("Enrollment failed")
        return jsonify(error=f"Enrollment failed: {exc}"), 500

@app.delete("/api/students/<student_id>")
def archive_student(student_id):
    if not DB.archive_student(student_id):
        return jsonify(error="Active student not found."), 404
    return jsonify({"ok": True, "message": "Student archived and face templates deleted. Historical attendance records are retained."})

@app.post("/api/attendance/scan")
def scan_attendance():
    payload = request.get_json(silent=True) or {}
    try:
        image = decode_image_data_url(payload.get("image", ""))
        threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
        if not 0.30 <= threshold <= 0.70:
            raise ValueError("Matching threshold must be between 0.30 and 0.70.")
        ENGINE.load()
        templates = DB.all_templates()
        if not templates:
            return jsonify(error="No active students are enrolled yet. Enroll students first."), 400
        detections = ENGINE.detect(image)
        results = []
        for face in detections:
            face_box = [int(v) for v in face[:4]]
            if min(face_box[2], face_box[3]) < 45:
                results.append({"box": face_box, "student_id": None, "full_name": "Face too small",
                                "class_name": "", "similarity": 0.0, "status": "low_quality",
                                "attendance": "low_quality"})
                continue
            embedding = ENGINE.embedding(image, face)
            match = match_face(embedding, templates, threshold)
            item = {"box": face_box, **match}
            if match["status"] == "matched":
                record = DB.mark_attendance(match["student_id"], match["similarity"])
                item["attendance"] = "marked" if record["created"] else "already_marked"
                item["marked_at"] = record["marked_at"]
            else:
                item["attendance"] = match["status"]
            results.append(item)
        annotated = draw_results(image, results)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Could not encode the annotated camera frame.")
        return jsonify({"ok": True, "faces_detected": len(detections), "results": results,
                        "image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
                        "summary": _scan_summary(results)})
    except FaceModelError as exc:
        return jsonify(error=str(exc)), 503
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        app.logger.exception("Attendance scan failed")
        return jsonify(error=f"Attendance scan failed: {exc}"), 500

def _scan_summary(results):
    marked = sum(r.get("attendance") == "marked" for r in results)
    already = sum(r.get("attendance") == "already_marked" for r in results)
    unknown = sum(r.get("attendance") in {"unknown", "ambiguous", "low_quality"} for r in results)
    return {"marked": marked, "already_present": already, "unmatched": unknown, "faces": len(results)}

@app.get("/api/attendance")
def attendance():
    day = request.args.get("date", date.today().isoformat())
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return jsonify(error="Date must be YYYY-MM-DD."), 400
    return jsonify({"date": day, "records": DB.list_attendance(day)})

@app.get("/api/attendance/export.csv")
def export_attendance():
    day = request.args.get("date", date.today().isoformat())
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return jsonify(error="Date must be YYYY-MM-DD."), 400
    records = DB.list_attendance(day, include_archived=True)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Student Name", "Class / Course", "Attendance Date", "Marked At", "Similarity"])
    writer.writerows([[r["student_id"], r["full_name"], r["class_name"], r["attendance_date"], r["marked_at"], f"{r['similarity']:.4f}"] for r in records])
    return Response("\ufeff" + output.getvalue(), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename=attendance-{day}.csv"})

if __name__ == "__main__":
    print("\nFaceAttend Pro is starting. Open http://127.0.0.1:5000")
    print("For privacy, keep this server local; do not expose it to a public network.\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
