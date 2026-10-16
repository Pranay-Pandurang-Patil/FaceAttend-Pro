import unittest
from face_attendance.recognition import MODEL_SOURCES

class ModelConfigTests(unittest.TestCase):
    def test_expected_model_files_are_configured(self):
        self.assertIn('yunet', MODEL_SOURCES)
        self.assertIn('sface', MODEL_SOURCES)
        self.assertGreater(MODEL_SOURCES['yunet']['minimum_bytes'], 100_000 - 1)
        self.assertGreater(MODEL_SOURCES['sface']['minimum_bytes'], 5_000_000 - 1)
        self.assertTrue(MODEL_SOURCES['yunet']['url'].endswith('.onnx'))
        self.assertTrue(MODEL_SOURCES['sface']['url'].endswith('.onnx'))

if __name__ == '__main__': unittest.main()
