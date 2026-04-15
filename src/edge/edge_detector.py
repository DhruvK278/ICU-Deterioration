"""
src/edge/edge_detector.py

Lightweight rule-based anomaly detector that runs on the bedside device
(Raspberry Pi / edge gateway). No ML model needed — pure threshold logic
so it works offline with <5ms latency.

Responsibilities:
  - Receive a single patient vitals reading
  - Flag immediate life-threatening anomalies instantly
  - Forward ALL readings to the fog server for LSTM risk scoring
  - If fog is unreachable, queue locally and retry

In this dataset we don't have real vitals, so the edge layer uses the
administrative features (acuity proxies) as stand-ins for demo purposes.
In a real deployment this would connect to bedside IoT sensors.

Usage:
    python src/edge/edge_detector.py                  # run demo simulation
    python src/edge/edge_detector.py --fog-url http://localhost:8000
"""

import argparse
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [EDGE]  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

THRESHOLDS = {
    "numlabs_high":        200,    # >200 lab orders → very high acuity
    "numchartevents_high": 500,    # >500 chart events → critical monitoring
    "numtransfers_high":   5,      # >5 transfers → highly unstable
    "numprocs_high":       20,     # >20 procedures → complex/deteriorating
    "numinput_high":       150,    # >150 inputs → aggressive fluid resuscitation
    "losdays_critical":    30,     # >30 days → prolonged / deteriorating
    "age_high_risk":       80,     # >80 years → high vulnerability
}

ALERT_LEVELS = {
    "CRITICAL": 3,   # immediate escalation
    "WARNING":  2,   # notify nurse
    "WATCH":    1,   # log and monitor
    "NORMAL":   0,
}


# Data structures
@dataclass
class PatientReading:
    hadm_id:        int
    timestamp:      str
    age:            float
    gender:         int           # 1=M, 0=F
    losdays:        float
    numchartevents: float
    numlabs:        float
    numprocs:       float
    numinput:       float
    numoutput:      float
    numtransfers:   float
    numrx:          float
    numnotes:       float
    numdiagnosis:   float
    admit_type:     str
    acuity_score:   float         # engineered feature from pipeline
    dx_sepsis:      int
    dx_cardiac:     int
    dx_respiratory: int
    dx_trauma:      int


@dataclass
class EdgeAlert:
    hadm_id:     int
    timestamp:   str
    level:       str              # CRITICAL / WARNING / WATCH / NORMAL
    level_int:   int
    triggers:    list             # which thresholds fired
    reading:     dict             # original reading
    forwarded:   bool = False     # did we reach the fog server?


# Core detector logic
class EdgeDetector:
    def __init__(self, fog_url: str = "http://localhost:8000", retry_queue_size: int = 100):
        self.fog_url     = fog_url.rstrip("/")
        self.retry_queue = deque(maxlen=retry_queue_size)  # offline buffer
        self.session     = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def analyse(self, reading: PatientReading) -> EdgeAlert:
        """Apply threshold rules and return an alert."""
        triggers = []

        # Check each threshold
        if reading.numlabs > THRESHOLDS["numlabs_high"]:
            triggers.append(f"numlabs={reading.numlabs:.0f} > {THRESHOLDS['numlabs_high']}")

        if reading.numchartevents > THRESHOLDS["numchartevents_high"]:
            triggers.append(f"numchartevents={reading.numchartevents:.0f} > {THRESHOLDS['numchartevents_high']}")

        if reading.numtransfers > THRESHOLDS["numtransfers_high"]:
            triggers.append(f"numtransfers={reading.numtransfers:.0f} > {THRESHOLDS['numtransfers_high']}")

        if reading.numprocs > THRESHOLDS["numprocs_high"]:
            triggers.append(f"numprocs={reading.numprocs:.0f} > {THRESHOLDS['numprocs_high']}")

        if reading.numinput > THRESHOLDS["numinput_high"]:
            triggers.append(f"numinput={reading.numinput:.0f} > {THRESHOLDS['numinput_high']}")

        if reading.losdays > THRESHOLDS["losdays_critical"]:
            triggers.append(f"losdays={reading.losdays:.1f} > {THRESHOLDS['losdays_critical']}")

        if reading.age > THRESHOLDS["age_high_risk"]:
            triggers.append(f"age={reading.age:.0f} > {THRESHOLDS['age_high_risk']}")

        # Diagnosis flags (binary — any positive is a trigger)
        if reading.dx_sepsis:
            triggers.append("dx_sepsis=1")
        if reading.dx_cardiac:
            triggers.append("dx_cardiac=1")
        if reading.dx_respiratory:
            triggers.append("dx_respiratory=1")

        # Determine alert level based on trigger count + severity
        critical_triggers = [t for t in triggers if any(
            k in t for k in ["numchartevents", "numtransfers", "dx_sepsis", "dx_cardiac", "dx_respiratory"]
        )]

        if len(critical_triggers) >= 2 or (reading.dx_sepsis and reading.numchartevents > 300):
            level = "CRITICAL"
        elif len(triggers) >= 3 or len(critical_triggers) >= 1:
            level = "WARNING"
        elif len(triggers) >= 1:
            level = "WATCH"
        else:
            level = "NORMAL"

        return EdgeAlert(
            hadm_id   = reading.hadm_id,
            timestamp = reading.timestamp,
            level     = level,
            level_int = ALERT_LEVELS[level],
            triggers  = triggers,
            reading   = asdict(reading),
        )

    def forward_to_fog(self, alert: EdgeAlert) -> bool:
        """Send alert + reading to fog server. Returns True if successful."""
        payload = asdict(alert)
        try:
            resp = self.session.post(
                f"{self.fog_url}/predict",
                json=payload,
                timeout=2.0,
            )
            if resp.status_code == 200:
                alert.forwarded = True
                return True
            else:
                log.warning(f"Fog returned {resp.status_code}: {resp.text[:100]}")
                return False
        except requests.exceptions.ConnectionError:
            log.warning(f"Fog unreachable — queuing locally (queue size: {len(self.retry_queue)+1})")
            self.retry_queue.append(alert)
            return False
        except requests.exceptions.Timeout:
            log.warning("Fog request timed out — queuing locally")
            self.retry_queue.append(alert)
            return False

    def flush_retry_queue(self):
        """Try to forward any queued alerts when fog comes back online."""
        flushed = 0
        while self.retry_queue:
            alert = self.retry_queue[0]
            if self.forward_to_fog(alert):
                self.retry_queue.popleft()
                flushed += 1
            else:
                break
        if flushed:
            log.info(f"Flushed {flushed} queued alerts to fog")

    def process(self, reading: PatientReading) -> EdgeAlert:
        """Full edge pipeline: analyse → alert → forward."""
        start   = time.perf_counter()
        alert   = self.analyse(reading)
        latency = (time.perf_counter() - start) * 1000

        log.info(
            f"[{alert.hadm_id}] {alert.level:8s} | "
            f"{len(alert.triggers)} triggers | "
            f"latency: {latency:.2f}ms | "
            f"triggers: {alert.triggers[:3]}"
        )

        self.flush_retry_queue()
        self.forward_to_fog(alert)

        return alert


# Demo simulation
def run_demo(fog_url: str):
    """Simulate a stream of patient readings from the processed dataset."""
    import pandas as pd
    from pathlib import Path

    ROOT     = Path(__file__).resolve().parents[2]
    PROC_DIR = ROOT / "data" / "processed"

    try:
        X = pd.read_parquet(PROC_DIR / "X_test.parquet")
        y = pd.read_parquet(PROC_DIR / "y_test.parquet").squeeze()
    except FileNotFoundError:
        log.error("Run data_pipeline.py first to generate processed splits.")
        return

    detector = EdgeDetector(fog_url=fog_url)
    log.info(f"Starting edge simulation — {len(X)} patients, fog at {fog_url}")
    log.info("-" * 60)

    stats = {"CRITICAL": 0, "WARNING": 0, "WATCH": 0, "NORMAL": 0}

    for i, (idx, row) in enumerate(X.head(20).iterrows()):
        reading = PatientReading(
            hadm_id        = int(idx),
            timestamp      = datetime.utcnow().isoformat(),
            age            = float(row.get("age", 60)),
            gender         = int(row.get("gender", 1)),
            losdays        = float(row.get("losdays", 5)),
            numchartevents = float(row.get("numchartevents", 100)),
            numlabs        = float(row.get("numlabs", 50)),
            numprocs       = float(row.get("numprocs", 5)),
            numinput       = float(row.get("numinput", 20)),
            numoutput      = float(row.get("numoutput", 15)),
            numtransfers   = float(row.get("numtransfers", 1)),
            numrx          = float(row.get("numrx", 10)),
            numnotes       = float(row.get("numnotes", 5)),
            numdiagnosis   = float(row.get("numdiagnosis", 8)),
            admit_type     = "EMERGENCY",
            acuity_score   = float(row.get("acuity_score", 100)),
            dx_sepsis      = int(row.get("dx_sepsis", 0)),
            dx_cardiac     = int(row.get("dx_cardiac", 0)),
            dx_respiratory = int(row.get("dx_respiratory", 0)),
            dx_trauma      = int(row.get("dx_trauma", 0)),
        )

        alert = detector.process(reading)
        stats[alert.level] += 1
        time.sleep(0.1)

    log.info("\n" + "=" * 60)
    log.info("Edge simulation complete:")
    for level, count in stats.items():
        log.info(f"  {level:8s}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fog-url", default="http://localhost:8000")
    args = parser.parse_args()
    run_demo(args.fog_url)