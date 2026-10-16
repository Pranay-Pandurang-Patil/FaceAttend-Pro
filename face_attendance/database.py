from __future__ import annotations
import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np

class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _initialize(self):
        with self.connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS students (
              student_id TEXT PRIMARY KEY, full_name TEXT NOT NULL, class_name TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL, consent_at TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS face_templates (
              id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL REFERENCES students(student_id),
              embedding BLOB NOT NULL, dimensions INTEGER NOT NULL, quality REAL NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_templates_student ON face_templates(student_id);
            CREATE TABLE IF NOT EXISTS attendance (
              id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL REFERENCES students(student_id),
              attendance_date TEXT NOT NULL, marked_at TEXT NOT NULL, similarity REAL NOT NULL,
              UNIQUE(student_id, attendance_date)
            );
            CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(attendance_date);
            """)

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")

    def add_student(self, student_id, full_name, class_name, templates):
        now = self._now()
        with self.connect() as c:
            c.execute("INSERT INTO students(student_id,full_name,class_name,created_at,consent_at,is_active) VALUES(?,?,?,?,?,1)",
                      (student_id, full_name, class_name, now, now))
            for item in templates:
                vector = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
                if vector.size < 16 or not np.isfinite(vector).all():
                    raise ValueError("The face model returned an invalid feature vector.")
                c.execute("INSERT INTO face_templates(student_id,embedding,dimensions,quality,created_at) VALUES(?,?,?,?,?)",
                          (student_id, vector.tobytes(), int(vector.size), float(item["quality"]), now))

    def list_students(self, active_only=True):
        query = """SELECT s.student_id,s.full_name,s.class_name,s.created_at,s.is_active,
                   COUNT(t.id) AS template_count FROM students s LEFT JOIN face_templates t USING(student_id)"""
        if active_only:
            query += " WHERE s.is_active=1"
        query += " GROUP BY s.student_id ORDER BY s.full_name COLLATE NOCASE"
        with self.connect() as c:
            return [dict(r) for r in c.execute(query).fetchall()]

    def get_student(self, student_id):
        with self.connect() as c:
            row = c.execute("SELECT student_id,full_name,class_name,created_at,is_active FROM students WHERE student_id=?", (student_id,)).fetchone()
            return dict(row) if row else None

    def all_templates(self):
        with self.connect() as c:
            rows = c.execute("""SELECT t.student_id,t.embedding,t.dimensions,s.full_name,s.class_name
                               FROM face_templates t JOIN students s USING(student_id)
                               WHERE s.is_active=1 ORDER BY t.student_id""").fetchall()
        return [{"student_id": r["student_id"], "full_name": r["full_name"], "class_name": r["class_name"],
                 "embedding": np.frombuffer(r["embedding"], dtype=np.float32, count=r["dimensions"]).copy()} for r in rows]

    def archive_student(self, student_id):
        now = self._now()
        with self.connect() as c:
            cur = c.execute("UPDATE students SET is_active=0 WHERE student_id=? AND is_active=1", (student_id,))
            if cur.rowcount:
                # Biometric templates are deleted immediately; historical attendance remains.
                c.execute("DELETE FROM face_templates WHERE student_id=?", (student_id,))
            return cur.rowcount > 0

    def mark_attendance(self, student_id, similarity):
        now = self._now()
        day = datetime.now().astimezone().date().isoformat()
        with self.connect() as c:
            cur = c.execute("INSERT OR IGNORE INTO attendance(student_id,attendance_date,marked_at,similarity) VALUES(?,?,?,?)",
                            (student_id, day, now, float(similarity)))
            row = c.execute("SELECT marked_at FROM attendance WHERE student_id=? AND attendance_date=?", (student_id, day)).fetchone()
            return {"created": cur.rowcount == 1, "marked_at": row["marked_at"] if row else now}

    def list_attendance(self, day, include_archived=False):
        query = """SELECT a.student_id,s.full_name,s.class_name,a.attendance_date,a.marked_at,a.similarity,s.is_active
                   FROM attendance a JOIN students s USING(student_id) WHERE a.attendance_date=?"""
        if not include_archived:
            query += " AND s.is_active=1"
        query += " ORDER BY a.marked_at DESC"
        with self.connect() as c:
            return [dict(r) for r in c.execute(query, (day,)).fetchall()]

    def summary(self):
        day = datetime.now().astimezone().date().isoformat()
        with self.connect() as c:
            students = c.execute("SELECT COUNT(*) FROM students WHERE is_active=1").fetchone()[0]
            present = c.execute("SELECT COUNT(*) FROM attendance a JOIN students s USING(student_id) WHERE a.attendance_date=? AND s.is_active=1", (day,)).fetchone()[0]
            templates = c.execute("SELECT COUNT(*) FROM face_templates t JOIN students s USING(student_id) WHERE s.is_active=1").fetchone()[0]
        return {"active_students": int(students), "present_today": int(present), "absent_today": max(0, int(students)-int(present)), "face_templates": int(templates), "date": day}
