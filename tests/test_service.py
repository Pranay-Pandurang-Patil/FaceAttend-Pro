import base64
import unittest
import cv2
import numpy as np
from face_attendance.service import (validate_student_id, decode_image_data_url,
                                     cosine_similarity, match_face, draw_results)

class ServiceTests(unittest.TestCase):
    def test_student_id_validation_and_normalization(self):
        self.assertEqual(validate_student_id('git2026-01'), 'GIT2026-01')
        self.assertEqual(validate_student_id('x'), '')
        self.assertEqual(validate_student_id('../admin'), '')
        self.assertEqual(validate_student_id('a' * 33), '')

    def test_data_url_image_decode(self):
        original = np.full((80, 120, 3), 127, dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', original)
        data = 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode()
        decoded = decode_image_data_url(data)
        self.assertEqual(decoded.shape, original.shape)

    def test_rejects_bad_data_url(self):
        with self.assertRaises(ValueError): decode_image_data_url('not-an-image')
        with self.assertRaises(ValueError): decode_image_data_url('data:image/jpeg;base64,???')

    def test_cosine_similarity(self):
        a = np.array([1, 0, 0], dtype=np.float32)
        b = np.array([2, 0, 0], dtype=np.float32)
        c = np.array([0, 1, 0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=6)
        self.assertAlmostEqual(cosine_similarity(a, c), 0.0, places=6)

    def test_match_accepts_clear_identity(self):
        q = np.array([1, 0, 0], dtype=np.float32)
        templates = [{'student_id':'S1','full_name':'Student One','class_name':'CSE','embedding':q},
                     {'student_id':'S2','full_name':'Student Two','class_name':'CSE','embedding':np.array([0,1,0],dtype=np.float32)}]
        r = match_face(q, templates, threshold=.42)
        self.assertEqual(r['status'], 'matched')
        self.assertEqual(r['student_id'], 'S1')

    def test_low_score_does_not_reveal_nearest_identity(self):
        q = np.array([1, 0, 0], dtype=np.float32)
        templates = [{'student_id':'S1','full_name':'Secret Name','embedding':np.array([0,1,0],dtype=np.float32)}]
        r = match_face(q, templates, threshold=.42)
        self.assertEqual(r['status'], 'unknown')
        self.assertEqual(r['full_name'], 'Unknown')
        self.assertIsNone(r['student_id'])

    def test_ambiguous_near_scores_are_rejected(self):
        q = np.array([1, 0], dtype=np.float32)
        a = np.array([.99, .141067], dtype=np.float32)
        b = np.array([.99, -.141067], dtype=np.float32)
        r = match_face(q, [{'student_id':'A1','full_name':'A','embedding':a},
                            {'student_id':'B1','full_name':'B','embedding':b}], threshold=.5, ambiguity_margin=.04)
        self.assertEqual(r['status'], 'ambiguous')
        self.assertIsNone(r['student_id'])

    def test_draw_results_does_not_change_input_shape(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        result = draw_results(image, [{'box':[10,10,40,40], 'attendance':'marked', 'full_name':'Test', 'similarity':.81}])
        self.assertEqual(result.shape, image.shape)
        self.assertGreater(int(result.sum()), 0)

if __name__ == '__main__': unittest.main()
