"""
Comprehensive test suite verifying two-mode workspace and all endpoints on port 3099:
1. Demo simulation mode with different drift factors & sample sizes
2. CSV upload and analysis for Model batches (genuine drift detection)
3. CSV upload and analysis for General tabular files (dataset profiling)
4. Retraining loop and auth routes
"""
import subprocess
import time
import urllib.request
import json
import sys
import os

TEST_PORT = 3099
BASE_URL = f"http://localhost:{TEST_PORT}"

def build_multipart_form(field_name, filename, file_bytes):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    content_type = f"multipart/form-data; boundary={boundary}"
    return content_type, body

def test_two_mode_system():
    print(f"🚀 Starting server.py on port {TEST_PORT} for two-mode verification...")
    env = dict(os.environ)
    env["PORT"] = str(TEST_PORT)
    proc = subprocess.Popen([sys.executable, "server.py"], env=env)
    time.sleep(2)

    try:
        # 1. Test HTML contains Two-Mode Cards and Upload Elements
        print("\n[TEST 1] Testing HTML contains Two-Mode Cards and Upload Elements...")
        with urllib.request.urlopen(f"{BASE_URL}/") as resp:
            html = resp.read().decode('utf-8')
            assert resp.status == 200
            assert "DEMO SIMULATION" in html
            assert "UPLOAD &amp; ANALYZE DATASET" in html or "UPLOAD & ANALYZE DATASET" in html
            assert "csv-dropzone" in html
            assert "sim-sample-size" in html
            assert "Model Drift Detection vs. General Dataset Profiling" in html
            print("✓ HTML verified: Contains prominent Two-Mode cards, dropzone, sample size selector, and educational callout.")

        # 2. Test Demo Simulation with different drift factors & sample sizes
        print("\n[TEST 2] Testing Demo Simulation (Drift: 0.60, N: 1,000)...")
        req = urllib.request.Request(f"{BASE_URL}/api/simulate?drift=0.60&n=1000&seed=7", method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            assert resp.status == 200
            assert data["n_rows"] == 1000
            assert data["is_simulated"] is True
            assert "score" in data
            assert len(data["features"]) == 7
            print(f"✓ Demo Simulation passed: Score={data['score']}, Level={data['level']}, Rows={data['n_rows']}, Simulated={data['is_simulated']}")

        # 3. Test Upload & Analyze Dataset with Model Batch (reference_2000.csv)
        print("\n[TEST 3] Testing Upload & Analyze with Model Batch (reference_2000.csv)...")
        with open("artifacts/reference_2000.csv", "rb") as f:
            ref_bytes = f.read()

        ctype, mbody = build_multipart_form("file", "reference_batch_q3.csv", ref_bytes)
        req = urllib.request.Request(f"{BASE_URL}/api/analyze", data=mbody, headers={"Content-Type": ctype}, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            assert resp.status == 200
            assert data.get("is_simulated") is False
            assert "features" in data
            assert len(data["features"]) == 7
            print(f"✓ Model Batch Upload passed: Detected Model Schema, Features={len(data['features'])}, Simulated={data.get('is_simulated')}")

        # 4. Test Upload & Analyze Dataset with General CSV (sample_test.csv)
        print("\n[TEST 4] Testing Upload & Analyze with General CSV (sample_test.csv)...")
        with open("sample_test.csv", "rb") as f:
            test_bytes = f.read()

        ctype, mbody = build_multipart_form("file", "student_scores.csv", test_bytes)
        req = urllib.request.Request(f"{BASE_URL}/api/analyze", data=mbody, headers={"Content-Type": ctype}, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            assert resp.status == 200
            assert data.get("mode") == "profile"
            assert data.get("column_count") == 4
            assert len(data.get("columns", [])) == 4
            print(f"✓ General CSV Profiling passed: Mode={data['mode']}, Columns={data['column_count']}, Rows={data['rows']}")

        # 5. Test Retraining Loop
        print("\n[TEST 5] Testing Model Retraining...")
        req = urllib.request.Request(f"{BASE_URL}/api/retrain", method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            assert resp.status == 200
            assert "new_baseline" in data
            print(f"✓ Retrain loop passed: New AUC={data['new_baseline']['auc']}")

        # 6. Test Auth endpoints
        print("\n[TEST 6] Testing Auth Database & Login...")
        login_payload = json.dumps({"email": "sarah.lin@mlops.corp", "password": "DemoSecure2026!"}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/auth/login", data=login_payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            assert resp.status == 200
            assert data["success"] is True

        with urllib.request.urlopen(f"{BASE_URL}/api/auth/users") as resp:
            data = json.loads(resp.read().decode('utf-8'))
            assert resp.status == 200
            assert len(data["users"]) >= 3
            print(f"✓ Auth endpoints passed: Logged in successfully, Users={len(data['users'])}")

        print("\n=======================================================")
        print("🎉 ALL TWO-MODE WORKSPACE & API TESTS PASSED 100%!")
        print("=======================================================")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test_two_mode_system()
