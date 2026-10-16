import tempfile
import unittest
from pathlib import Path
import numpy as np
from face_attendance.database import Database

class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / 'test.sqlite3')
        self.embedding = np.arange(1, 129, dtype=np.float32)
        self.embedding /= np.linalg.norm(self.embedding)
        self.db.add_student('STU-001', 'Test Student', 'CSE 3rd Year', [
            {'embedding': self.embedding, 'quality': .94},
            {'embedding': self.embedding, 'quality': .91},
            {'embedding': self.embedding, 'quality': .93},
        ])

    def tearDown(self): self.tmp.cleanup()

    def test_enrollment_writes_three_templates(self):
        students = self.db.list_students()
        self.assertEqual(len(students), 1)
        self.assertEqual(students[0]['template_count'], 3)
        self.assertEqual(students[0]['student_id'], 'STU-001')

    def test_template_embedding_roundtrip(self):
        templates = self.db.all_templates()
        self.assertEqual(len(templates), 3)
        self.assertEqual(templates[0]['embedding'].shape, (128,))
        self.assertAlmostEqual(float(np.dot(templates[0]['embedding'], self.embedding)), 1.0, places=5)

    def test_duplicate_attendance_is_idempotent(self):
        a = self.db.mark_attendance('STU-001', .81)
        b = self.db.mark_attendance('STU-001', .89)
        self.assertTrue(a['created'])
        self.assertFalse(b['created'])
        records = self.db.list_attendance(self.db.summary()['date'])
        self.assertEqual(len(records), 1)

    def test_archive_removes_biometrics_but_preserves_history(self):
        self.db.mark_attendance('STU-001', .81)
        self.assertTrue(self.db.archive_student('STU-001'))
        self.assertEqual(self.db.list_students(), [])
        self.assertEqual(self.db.all_templates(), [])
        records = self.db.list_attendance(self.db.summary()['date'], include_archived=True)
        self.assertEqual(len(records), 1)

    def test_invalid_embedding_is_rejected(self):
        with self.assertRaises(ValueError):
            self.db.add_student('STU-002', 'Bad', '', [{'embedding': [float('nan')] * 20, 'quality': .9}])

if __name__ == '__main__': unittest.main()
