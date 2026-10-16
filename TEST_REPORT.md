# FaceAttend Pro — local test report

Checks performed in the available build environment:

- Python `compileall`: passed.
- JavaScript `node --check static/app.js`: passed.
- Unit tests: 14 passed per run.
- Repeated unit-test run: 10 consecutive runs passed (140 test executions).
- SQLite: enrollment/template persistence, vector round-trip, per-day duplicate prevention, and biometric template deletion on archive tested.
- Matching logic: clear match, below-threshold unknown, ambiguous match rejection, cosine similarity, input validation, image decode, and annotation tested.
- ZIP integrity: verified when packaging.

**Not claimed as tested:** end-to-end Flask/browser/camera integration or real biometric accuracy. Flask is not installed in the isolated build environment and network access prevented installing project dependencies or downloading the ONNX models here. Those final checks must be run after installing `requirements.txt` on the target Windows machine. The unit tests use deterministic synthetic feature vectors and are not an accuracy benchmark.
