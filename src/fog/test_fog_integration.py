"""
src/fog/test_fog_integration.py

Integration test: spins up the fog server in a background thread,
then runs the edge detector simulation against it and verifies responses.

Run:
    python src/fog/test_fog_integration.py
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import requests
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "fog"))
sys.path.insert(0, str(ROOT / "src" / "edge"))

FOG_URL  = "http://localhost:8765"
FOG_PORT = 8765


def start_fog_server():
    import fog_server
    config = uvicorn.Config(fog_server.app, host="0.0.0.0", port=FOG_PORT, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def wait_for_fog(timeout=15):
    for _ in range(timeout * 10):
        try:
            r = requests.get(f"{FOG_URL}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def run_tests():
    log.info("Starting fog server in background...")
    thread = threading.Thread(target=start_fog_server, daemon=True)
    thread.start()

    if not wait_for_fog():
        log.error("Fog server did not start in time")
        sys.exit(1)

    log.info(f"Fog server ready at {FOG_URL}")
    log.info("=" * 60)

    # Test 1: Health check
    r = requests.get(f"{FOG_URL}/health")
    health = r.json()
    assert r.status_code == 200
    assert health["xgb_loaded"] is True
    log.info(f"[PASS] Health check — XGBoost loaded: {health['xgb_loaded']}, LSTM: {health['lstm_loaded']}")

    # Test 2: Normal patient
    normal_payload = {
        "hadm_id":   1001,
        "timestamp": "2026-01-01T10:00:00",
        "level":     "NORMAL",
        "level_int": 0,
        "triggers":  [],
        "reading": {
            "hadm_id": 1001, "age": 45, "gender": 1, "losdays": 2.0,
            "numchartevents": 80, "numlabs": 30, "numprocs": 3,
            "numinput": 10, "numoutput": 8, "numtransfers": 0,
            "numrx": 5, "numnotes": 3, "numdiagnosis": 4,
            "numcallouts": 0, "numcptevents": 5, "nummicrolabs": 1,
            "numprocevents": 2, "totalnuminteract": 50,
            "admit_type": "ELECTIVE", "acuity_score": 60.0,
            "dx_sepsis": 0, "dx_cardiac": 0, "dx_respiratory": 0, "dx_trauma": 0,
        },
        "forwarded": False,
    }
    r = requests.post(f"{FOG_URL}/predict", json=normal_payload)
    assert r.status_code == 200
    pred = r.json()
    log.info(f"[PASS] Normal patient — ensemble_risk={pred['ensemble_risk']:.3f}, alert={pred['alert_level']}")

    # Test 3: High-risk patient
    critical_payload = {
        "hadm_id":   2002,
        "timestamp": "2026-01-01T10:01:00",
        "level":     "CRITICAL",
        "level_int": 3,
        "triggers":  ["dx_sepsis=1", "numchartevents=620 > 500", "numtransfers=7 > 5"],
        "reading": {
            "hadm_id": 2002, "age": 82, "gender": 0, "losdays": 35.0,
            "numchartevents": 620, "numlabs": 250, "numprocs": 25,
            "numinput": 180, "numoutput": 100, "numtransfers": 7,
            "numrx": 35, "numnotes": 40, "numdiagnosis": 18,
            "numcallouts": 3, "numcptevents": 40, "nummicrolabs": 20,
            "numprocevents": 30, "totalnuminteract": 800,
            "admit_type": "EMERGENCY", "acuity_score": 850.0,
            "dx_sepsis": 1, "dx_cardiac": 1, "dx_respiratory": 0, "dx_trauma": 0,
        },
        "forwarded": False,
    }
    r = requests.post(f"{FOG_URL}/predict", json=critical_payload)
    assert r.status_code == 200
    pred = r.json()
    log.info(f"[PASS] Critical patient — ensemble_risk={pred['ensemble_risk']:.3f}, alert={pred['alert_level']}")
    assert pred["ensemble_risk"] > 0.4, "High-risk patient should score > 0.4"

    # Test 4: Multiple readings
    for i in range(5):
        critical_payload["timestamp"] = f"2026-01-01T10:0{i+2}:00"
        r = requests.post(f"{FOG_URL}/predict", json=critical_payload)
        assert r.status_code == 200
    pred = r.json()
    assert pred["window_size"] > 1
    log.info(f"[PASS] Rolling window — window_size={pred['window_size']}")

    # Test 5: Patient list
    r = requests.get(f"{FOG_URL}/patients")
    assert r.status_code == 200
    patients = r.json()
    assert 2002 in patients or "2002" in [str(k) for k in patients.keys()]
    log.info(f"[PASS] Patient list — {len(patients)} active patients")

    # Test 6: Discharge
    r = requests.delete(f"{FOG_URL}/patients/2002")
    assert r.status_code == 200
    log.info("[PASS] Patient discharge")

    log.info("=" * 60)
    log.info("All integration tests passed.")


if __name__ == "__main__":
    run_tests()