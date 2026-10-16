from __future__ import annotations
import base64, binascii, re
import cv2
import numpy as np

STUDENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")

def validate_student_id(value):
    value = str(value or "").strip().upper()
    return value if STUDENT_ID_RE.fullmatch(value) else ""

def decode_image_data_url(value):
    if not isinstance(value, str) or "," not in value:
        raise ValueError("Missing webcam image. Start the camera and capture a frame.")
    header, encoded = value.split(",", 1)
    if not header.startswith("data:image/") or len(encoded) > 14_000_000:
        raise ValueError("Invalid or overly large image data. Capture a smaller webcam frame.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image data is corrupted; capture the photo again.") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The camera image could not be decoded.")
    if image.shape[0] > 3000 or image.shape[1] > 3000:
        raise ValueError("Image dimensions exceed 3000 px. Use the built-in webcam capture.")
    return image

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size or a.size == 0:
        return -1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else -1.0

def match_face(embedding, templates, threshold=0.42, ambiguity_margin=0.04):
    """Compare an embedding with enrolled templates; never force a low-confidence identity."""
    if not templates:
        return {"status": "unknown", "student_id": None, "full_name": "Unknown", "similarity": 0.0}
    best_by_student = {}
    for item in templates:
        score = cosine_similarity(embedding, item["embedding"])
        sid = item["student_id"]
        if sid not in best_by_student or score > best_by_student[sid]["similarity"]:
            best_by_student[sid] = {"student_id": sid, "full_name": item["full_name"],
                                    "class_name": item.get("class_name", ""), "similarity": score}
    ranked = sorted(best_by_student.values(), key=lambda x: x["similarity"], reverse=True)
    best = ranked[0]
    if best["similarity"] < threshold:
        # Do not reveal the nearest enrolled identity when the match is rejected.
        return {"student_id": None, "full_name": "Unknown", "class_name": "",
                "similarity": best["similarity"], "status": "unknown"}
    if len(ranked) > 1 and (best["similarity"] - ranked[1]["similarity"]) < ambiguity_margin:
        # Avoid presenting a low-margin identity as confirmed.
        return {"student_id": None, "full_name": "Needs review", "class_name": "",
                "similarity": best["similarity"], "status": "ambiguous",
                "message": "Two student templates scored similarly; verify manually."}
    return {**best, "status": "matched"}

def draw_results(image, results):
    canvas = image.copy()
    for r in results:
        x, y, w, h = r["box"]
        status = r.get("attendance") or r.get("status")
        if status == "marked": color = (45, 190, 100)
        elif status == "already_marked": color = (0, 190, 240)
        elif status == "matched": color = (45, 190, 100)
        elif status == "ambiguous": color = (0, 165, 255)
        else: color = (45, 55, 230)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 3)
        score = r.get("similarity")
        text = r.get("full_name", "Unknown")
        if score is not None: text += f" {score:.2f}"
        text = text[:48]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = max(0.5, canvas.shape[1]/1300), 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        ty = max(th+8, y-8)
        cv2.rectangle(canvas, (x, ty-th-7), (min(canvas.shape[1]-1, x+tw+10), ty+baseline), color, -1)
        cv2.putText(canvas, text, (x+5, ty-3), font, scale, (255,255,255), thickness, cv2.LINE_AA)
    return canvas
