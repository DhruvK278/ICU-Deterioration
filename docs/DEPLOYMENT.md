# Deployment & Operations Guide

## Overview

The ICU Deterioration Prediction system is designed to run in multiple environments:
- **Local Development**: Directly in Python virtual environments.
- **On-Premise Ward Deployment**: Multi-container Docker Compose stack on hospital hardware.
- **Cloud / Staging Deployment**: Containerized or PaaS hosting (e.g. Render, AWS ECS, GCP Cloud Run).

---

## 1. Local Development Setup

### Prerequisites
- Python 3.11 or 3.12
- `git`
- GCC/build tools (for compiling certain scientific dependencies if wheel is unavailable)

### Step-by-Step Installation
```bash
# 1. Clone repository
git clone https://github.com/DhruvK278/ICU-Deterioration.git
cd ICU-Deterioration

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Running Services Locally

#### Terminal 1: Fog Inference Gateway & Agent Pipeline
```bash
uvicorn src.fog.fog_server:app --host 0.0.0.0 --port 8000 --reload
```
- API is available at `http://localhost:8000`
- Swagger UI documentation at `http://localhost:8000/docs`

#### Terminal 2: Clinician Monitoring Dashboard
```bash
streamlit run dashboard/app.py
```
- Dashboard opens at `http://localhost:8501`

#### Terminal 3: Bedside Edge Simulation
```bash
# Sends simulated telemetry readings to the Fog server
python src/edge/edge_detector.py --fog-url http://localhost:8000
```

#### Terminal 4 (Optional): MLflow Experiment Tracking
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
- UI available at `http://localhost:5000`

---

## 2. Docker Compose Deployment (Recommended for Wards)

The repository provides a production-ready [`docker-compose.yml`](file:///home/dhruvk27/Documents/Projects/ICU-Deterioration/docker-compose.yml) that coordinates all system tiers.

### Services Defined
- **`fog`**: FastAPI server on port `8000` with volume mounts to `./models` and `./data/processed`. Includes automated healthchecks.
- **`dashboard`**: Streamlit UI on port `8501`, depending on `fog` becoming healthy.
- **`edge`**: Simulated bedside edge agent pushing readings to `fog`.
- **`mlflow`**: Centralized MLflow tracking server on port `5000` with persistent volume `mlflow_data`.

### Starting the Stack
```bash
# Build and start all services in the background
docker-compose up --build -d

# Check running container statuses
docker-compose ps

# Tail logs of the fog server
docker-compose logs -f fog

# Shut down the stack
docker-compose down
```

### Targeting Individual Services
```bash
# Run only the fog server
docker-compose up -d fog

# Run fog and dashboard without edge simulation
docker-compose up -d fog dashboard
```

---

## 3. Environment Variables Reference

| Variable | Default | Service | Purpose |
| :--- | :--- | :--- | :--- |
| `FOG_URL` | `http://localhost:8000` | Dashboard | Base URL used by the dashboard to communicate with Fog |
| `FOG_API_KEY` | *(None / Unauthenticated)* | Fog, Edge, Dashboard | Shared API secret for `X-API-Key` authentication |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Fog, Cloud Training | Tracking URI for metric and artifact logging |
| `CORS_ORIGINS` | `http://localhost:8501...` | Fog | Comma-separated list of allowed CORS origins |
| `FOG_SERVER_HOST` | `localhost` | Edge | Hostname or IP of the ward Fog server |
| `FOG_SERVER_PORT` | `8000` | Edge | Port of the ward Fog server |
| `XGB_MODEL_HASH` | *(None / Log only)* | Fog | Expected SHA-256 hash for `xgb_model.pkl` |
| `LSTM_MODEL_HASH` | *(None / Log only)* | Fog | Expected SHA-256 hash for `lstm_model.pt` |
| `SCALER_HASH` | *(None / Log only)* | Fog | Expected SHA-256 hash for `scaler.pkl` |

---

## 4. Cryptographic Model Integrity Verification

To prevent unauthorized tampering, weight poisoning, or corrupted model downloads in critical care deployments, `fog_server.py` implements SHA-256 checksum validation:

### Generating Hashes for New Retrained Models
```bash
sha256sum models/xgb_model.pkl
sha256sum models/lstm_model.pt
sha256sum data/processed/scaler.pkl
```

### Locking Model Checksums in Production
Set the computed hashes as environment variables prior to launching the Fog server:
```bash
export XGB_MODEL_HASH="1a25950344c114b2..."
export LSTM_MODEL_HASH="96583dca949076f7..."
export SCALER_HASH="9605d193a7aeecd8..."

uvicorn src.fog.fog_server:app --host 0.0.0.0 --port 8000
```

If any file hash fails to match the expected value, the server logs a critical alert and aborts startup with a `RuntimeError`. If the environment variables are unset, the server logs the detected hash for audit inspection.

---

## 5. Cloud PaaS Deployment (Render)

The project includes a [`render.yaml`](file:///home/dhruvk27/Documents/Projects/ICU-Deterioration/render.yaml) blueprint to deploy the Streamlit dashboard as a managed web service:

```yaml
services:
  - type: web
    name: icu-deterioration-dashboard
    runtime: python
    region: oregon
    plan: free
    buildCommand: pip install streamlit plotly pandas requests numpy
    startCommand: streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: FOG_URL
        sync: false  # Configure your live Fog Gateway URL in the Render UI
      - key: PYTHON_VERSION
        value: 3.11.0
```

---

## 6. System Verification Checklist

Before placing the system into operation, verify the following:

- [ ] All unit and validation tests pass: `pytest tests/ -v`
- [ ] End-to-end integration test passes: `python src/fog/test_fog_integration.py`
- [ ] Fog server `/health` endpoint reports `xgb_loaded: true` and `lstm_loaded: true`
- [ ] SQLite audit database is initialized at `data/audit_log.db`
- [ ] `FOG_API_KEY` is set to a secure string in production
- [ ] Clinician dashboard can poll `/alerts/pending` and execute actions
