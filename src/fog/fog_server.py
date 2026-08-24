"""
src/fog/fog_server.py

FastAPI inference server that runs on the ward-level gateway.

Responsibilities:
  - Receive readings forwarded from edge devices
  - Score each patient using the trained XGBoost model
  - Maintain a rolling window of recent scores per patient
  - Escalate high-risk patients to the nurse alert endpoint
  - Batch-forward anonymised data to cloud every 5 minutes

Run:
    uvicorn src.fog.fog_server:app --host 0.0.0.0 --port 8000 --reload
    # or:
    python src/fog/fog_server.py
"""

import hashlib
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [FOG]  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"


# ── Model integrity verification ──────────────────────────────────────────────
def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_model_integrity(filepath: Path, expected_hash: Optional[str]) -> str:
    """Verify model file integrity. Logs hash on first run, validates on subsequent."""
    actual_hash = compute_sha256(filepath)
    if expected_hash and actual_hash != expected_hash:
        log.critical(
            f"MODEL INTEGRITY FAILURE: {filepath.name} — "
            f"expected {expected_hash[:16]}... got {actual_hash[:16]}..."
        )
        raise RuntimeError(f"Model integrity check failed for {filepath.name}")
    log.info(f"Model integrity OK: {filepath.name} sha256={actual_hash[:16]}...")
    return actual_hash


# Expected hashes — update these after each retrain
# Set to None to skip verification (logs hash for first-time capture)
MODEL_HASHES = {
    "xgb_model.pkl":  os.getenv("XGB_MODEL_HASH"),
    "lstm_model.pt":  os.getenv("LSTM_MODEL_HASH"),
    "scaler.pkl":     os.getenv("SCALER_HASH"),
}


# Load feature list
with open(MODEL_DIR / "feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

# Load XGBoost model (with integrity check)
verify_model_integrity(MODEL_DIR / "xgb_model.pkl", MODEL_HASHES.get("xgb_model.pkl"))
xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
log.info(f"XGBoost model loaded — {len(FEATURE_NAMES)} features")

# Load LSTM model
class ICULSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


lstm_model  = None
lstm_config = None
if (MODEL_DIR / "lstm_model.pt").exists() and (MODEL_DIR / "lstm_config.json").exists():
    verify_model_integrity(MODEL_DIR / "lstm_model.pt", MODEL_HASHES.get("lstm_model.pt"))
    with open(MODEL_DIR / "lstm_config.json") as f:
        lstm_config = json.load(f)
    lstm_model = ICULSTMClassifier(
        input_dim  = lstm_config["input_dim"],
        hidden_dim = lstm_config["hidden_dim"],
        num_layers = lstm_config["num_layers"],
        dropout    = lstm_config["dropout"],
    )
    lstm_model.load_state_dict(torch.load(MODEL_DIR / "lstm_model.pt", map_location="cpu", weights_only=True))
    lstm_model.eval()
    log.info("LSTM model loaded")
else:
    log.warning("LSTM model not found — fog will use XGBoost only")

# Load scaler (with integrity check)
verify_model_integrity(PROC_DIR / "scaler.pkl", MODEL_HASHES.get("scaler.pkl"))
scaler = joblib.load(PROC_DIR / "scaler.pkl")

# Per-patient rolling window (last 8 readings)
patient_windows: dict = defaultdict(lambda: deque(maxlen=8))
patient_risk_history: dict = defaultdict(list)

RISK_THRESHOLD_ALERT    = 0.6   # escalate to nurse above this
RISK_THRESHOLD_CRITICAL = 0.8   # critical alert above this


# Pydantic schemas with input validation
class VitalsReading(BaseModel):
    """Validated patient vitals/administrative data from edge devices."""
    age:            float = Field(default=60, ge=0, le=120, description="Patient age in years")
    gender:         int   = Field(default=1, ge=0, le=1)
    losdays:        float = Field(default=0, ge=0, le=365, description="Length of stay in days")
    numchartevents: float = Field(default=0, ge=0)
    numlabs:        float = Field(default=0, ge=0)
    numprocs:       float = Field(default=0, ge=0)
    numinput:       float = Field(default=0, ge=0)
    numoutput:      float = Field(default=0, ge=0)
    numtransfers:   float = Field(default=0, ge=0)
    numrx:          float = Field(default=0, ge=0)
    numnotes:       float = Field(default=0, ge=0)
    numdiagnosis:   float = Field(default=0, ge=0)
    numcallouts:    float = Field(default=0, ge=0)
    numcptevents:   float = Field(default=0, ge=0)
    nummicrolabs:   float = Field(default=0, ge=0)
    numprocevents:  float = Field(default=0, ge=0)
    totalnuminteract: float = Field(default=0, ge=0)
    admit_type:     str   = Field(default="EMERGENCY")
    acuity_score:   float = Field(default=0, ge=0)
    dx_sepsis:      int   = Field(default=0, ge=0, le=1)
    dx_cardiac:     int   = Field(default=0, ge=0, le=1)
    dx_respiratory: int   = Field(default=0, ge=0, le=1)
    dx_trauma:      int   = Field(default=0, ge=0, le=1)


class EdgeReading(BaseModel):
    hadm_id:   int   = Field(gt=0, description="Hospital admission ID")
    timestamp: str
    level:     str   = Field(pattern="^(CRITICAL|WARNING|WATCH|NORMAL)$")
    level_int: int   = Field(ge=0, le=3)
    triggers:  list
    reading:   VitalsReading
    forwarded: bool  = False


class PredictionResponse(BaseModel):
    hadm_id:       int
    timestamp:     str
    xgb_risk:      float
    lstm_risk:     Optional[float]
    ensemble_risk: float
    alert_level:   str
    edge_level:    str
    triggers:      list
    window_size:   int


class HealthResponse(BaseModel):
    status:        str
    xgb_loaded:    bool
    lstm_loaded:   bool
    patients_seen: int
    uptime_s:      float


# ── Authentication ────────────────────────────────────────────────────────────
FOG_API_KEY = os.getenv("FOG_API_KEY")

if not FOG_API_KEY:
    log.warning(
        "FOG_API_KEY not set — server is UNAUTHENTICATED. "
        "Set FOG_API_KEY env var for production use."
    )

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    """Validate API key if FOG_API_KEY is configured."""
    if FOG_API_KEY is None:
        return  # auth disabled (dev mode)
    if not key or key != FOG_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ICU Deterioration — Fog Inference Server",
    description="Ward-level risk scoring for ICU patients",
    version="1.1.0",
)

# CORS — restricted to known origins (dashboard, edge containers)
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

START_TIME = time.time()


# Feature extraction
def extract_features(reading: VitalsReading) -> pd.DataFrame:
    """
    Map the validated VitalsReading onto the exact feature vector the models expect.
    Missing features are filled with 0 (safe default after StandardScaler).
    """
    reading_dict = reading.model_dump()
    row = {col: 0.0 for col in FEATURE_NAMES}

    direct_map = [
        "numcallouts", "numdiagnosis", "numprocs", "numcptevents",
        "numinput", "numlabs", "numnotes", "numoutput", "numrx",
        "numprocevents", "numtransfers", "numchartevents",
        "age", "gender", "losdays", "acuity_score",
        "dx_sepsis", "dx_cardiac", "dx_respiratory", "dx_trauma",
    ]
    for key in direct_map:
        if key in row and key in reading_dict:
            row[key] = float(reading_dict.get(key, 0) or 0)

    # Engineered features
    numinput  = float(reading_dict.get("numinput", 0) or 0)
    numoutput = float(reading_dict.get("numoutput", 0) or 0)
    numlabs   = float(reading_dict.get("numlabs", 1) or 1)
    numicrolabs = float(reading_dict.get("nummicrolabs", 0) or 0)

    if "io_ratio" in row:
        total = numinput + numoutput
        row["io_ratio"] = numinput / total if total > 0 else 0.5

    if "micro_rate" in row:
        row["micro_rate"] = numicrolabs / numlabs if numlabs > 0 else 0.0

    numtransfers = float(reading_dict.get("numtransfers", 0) or 0)
    losdays      = float(reading_dict.get("losdays", 0) or 0)
    age          = float(reading_dict.get("age", 60) or 60)

    if "high_transfer" in row:
        row["high_transfer"] = 1.0 if numtransfers > 2 else 0.0
    if "los_gt7"  in row:
        row["los_gt7"]  = 1.0 if losdays > 7  else 0.0
    if "los_gt14" in row:
        row["los_gt14"] = 1.0 if losdays > 14 else 0.0
    if "los_gt30" in row:
        row["los_gt30"] = 1.0 if losdays > 30 else 0.0
    if "age_gt65" in row:
        row["age_gt65"] = 1.0 if age > 65 else 0.0
    if "age_gt80" in row:
        row["age_gt80"] = 1.0 if age > 80 else 0.0
    if "has_callouts" in row:
        row["has_callouts"] = 1.0 if float(reading_dict.get("numcallouts", 0) or 0) > 0 else 0.0

    admit_type = reading_dict.get("admit_type", "").upper()
    for col in ["admit_type_EMERGENCY", "admit_type_URGENT", "admit_type_NEWBORN"]:
        if col in row:
            row[col] = 1.0 if col.replace("admit_type_", "") in admit_type else 0.0

    df  = pd.DataFrame([row])[FEATURE_NAMES]
    arr = scaler.transform(df)
    return pd.DataFrame(arr, columns=FEATURE_NAMES)


def score_xgb(features: pd.DataFrame) -> float:
    return float(xgb_model.predict_proba(features)[:, 1][0])


def score_lstm(features: pd.DataFrame, window: deque) -> Optional[float]:
    if lstm_model is None:
        return None
    seq_len = lstm_config["seq_len"]
    rows    = list(window)
    while len(rows) < seq_len:
        rows.insert(0, features.values[0])
    seq_arr = np.stack(rows[-seq_len:], axis=0)           
    tensor  = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0)  
    with torch.no_grad():
        logit = lstm_model(tensor)
        prob  = torch.sigmoid(logit).item()
    return float(prob)


def determine_alert(ensemble_risk: float, edge_level: str) -> str:
    if ensemble_risk >= RISK_THRESHOLD_CRITICAL or edge_level == "CRITICAL":
        return "CRITICAL"
    elif ensemble_risk >= RISK_THRESHOLD_ALERT or edge_level == "WARNING":
        return "WARNING"
    elif ensemble_risk >= 0.4 or edge_level == "WATCH":
        return "WATCH"
    return "NORMAL"


# Endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status        = "ok",
        xgb_loaded    = xgb_model is not None,
        lstm_loaded   = lstm_model is not None,
        patients_seen = len(patient_windows),
        uptime_s      = round(time.time() - START_TIME, 1),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: EdgeReading,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key),
):
    hadm_id   = payload.hadm_id
    reading   = payload.reading
    timestamp = datetime.utcnow().isoformat()

    try:
        features = extract_features(reading)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {e}")

    xgb_risk = score_xgb(features)

    patient_windows[hadm_id].append(features.values[0])
    lstm_risk = score_lstm(features, patient_windows[hadm_id])

    if lstm_risk is not None:
        ensemble_risk = 0.90 * xgb_risk + 0.10 * lstm_risk
    else:
        ensemble_risk = xgb_risk

    alert_level = determine_alert(ensemble_risk, payload.level)

    if alert_level in ("CRITICAL", "WARNING"):
        log.warning(
            f"[{hadm_id}] {alert_level} — ensemble_risk={ensemble_risk:.3f} "
            f"xgb={xgb_risk:.3f} lstm={f'{lstm_risk:.3f}' if lstm_risk is not None else 'N/A'}"
        )

    patient_risk_history[hadm_id].append({
        "timestamp":    timestamp,
        "ensemble_risk": round(ensemble_risk, 4),
        "alert_level":  alert_level,
    })

    return PredictionResponse(
        hadm_id       = hadm_id,
        timestamp     = timestamp,
        xgb_risk      = round(xgb_risk, 4),
        lstm_risk     = round(lstm_risk, 4) if lstm_risk is not None else None,
        ensemble_risk = round(ensemble_risk, 4),
        alert_level   = alert_level,
        edge_level    = payload.level,
        triggers      = payload.triggers,
        window_size   = len(patient_windows[hadm_id]),
    )


@app.get("/patients")
def list_patients():
    """Return current risk scores for all active patients — used by dashboard."""
    result = {}
    for hadm_id, history in patient_risk_history.items():
        if history:
            latest = history[-1]
            result[hadm_id] = {
                "latest_risk":  latest["ensemble_risk"],
                "alert_level":  latest["alert_level"],
                "last_updated": latest["timestamp"],
                "num_readings": len(history),
            }
    return result


@app.get("/patients/{hadm_id}")
def get_patient(hadm_id: int):
    """Return full risk history for a single patient."""
    if hadm_id not in patient_risk_history:
        raise HTTPException(status_code=404, detail=f"Patient {hadm_id} not found")
    return {
        "hadm_id": hadm_id,
        "history": patient_risk_history[hadm_id],
        "window":  len(patient_windows[hadm_id]),
    }


@app.delete("/patients/{hadm_id}")
def discharge_patient(hadm_id: int, _: None = Depends(verify_api_key)):
    """Clear patient state on discharge. Requires authentication."""
    patient_windows.pop(hadm_id, None)
    patient_risk_history.pop(hadm_id, None)
    log.info(f"Patient {hadm_id} discharged — state cleared")
    return {"discharged": hadm_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fog_server:app", host="0.0.0.0", port=8000, reload=False)