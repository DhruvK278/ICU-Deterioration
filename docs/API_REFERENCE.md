# Fog Gateway API Reference

The Ward Fog Gateway (`src/fog/fog_server.py`) provides a high-throughput, low-latency REST API built with **FastAPI**. It handles real-time patient telemetry scoring, multi-agent pipeline orchestration, patient state querying, and clinician human-in-the-loop action workflows.

---

## Base URL & Authentication

- **Default Local Address**: `http://localhost:8000`
- **Docker Compose Network**: `http://fog:8000`
- **Interactive Documentation**: `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc`

### Authentication Header
When the `FOG_API_KEY` environment variable is defined on the server, all non-health endpoints require the `X-API-Key` HTTP header:

```http
X-API-Key: your_secret_api_key_here
```

If `FOG_API_KEY` is empty or not set, the server runs in unauthenticated mode for local development.

---

## Endpoint Summary

| Method | Endpoint | Description | Auth Required? |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Server health, model loaded states, and uptime | No |
| `POST` | `/predict` | Dual-model ML inference (XGBoost + LSTM) | Yes (if configured) |
| `POST` | `/predict/agent` | Full LangGraph multi-agent decision support execution | Yes (if configured) |
| `GET` | `/patients` | Ward overview of all active patient risk scores | Yes (if configured) |
| `GET` | `/patients/{hadm_id}` | Longitudinal risk history and window size for one patient | Yes (if configured) |
| `DELETE` | `/patients/{hadm_id}` | Discharge patient and purge rolling window state | Yes (if configured) |
| `GET` | `/alerts/pending` | Fetch unacknowledged alerts for clinician queue | Yes (if configured) |
| `POST` | `/alerts/{alert_id}/action` | Submit clinician action (acknowledge, dismiss, escalate) | Yes (if configured) |

---

## Detailed Endpoint Documentation

### 1. Health Check
`GET /health`

Returns operational status, model availability flags, active patient count, and server uptime.

#### Example Request
```bash
curl -X GET http://localhost:8000/health
```

#### Example Response (`200 OK`)
```json
{
  "status": "ok",
  "xgb_loaded": true,
  "lstm_loaded": true,
  "patients_seen": 14,
  "uptime_s": 1284.5
}
```

---

### 2. Standard ML Inference
`POST /predict`

Scores incoming bedside telemetry using the XGBoost + LSTM ensemble model. Updates the patient's rolling observation window and risk history.

#### Request Headers
```http
Content-Type: application/json
X-API-Key: <FOG_API_KEY>
```

#### Request Body (`EdgeReading`)
```json
{
  "hadm_id": 109283,
  "timestamp": "2026-10-22T08:30:00Z",
  "level": "WARNING",
  "level_int": 2,
  "triggers": ["dx_sepsis=1", "numlabs=210 > 200"],
  "reading": {
    "age": 68.0,
    "gender": 1,
    "losdays": 5.4,
    "numchartevents": 320,
    "numlabs": 210,
    "numprocs": 8,
    "numinput": 45,
    "numoutput": 38,
    "numtransfers": 3,
    "numrx": 22,
    "numnotes": 14,
    "numdiagnosis": 11,
    "numcallouts": 1,
    "numcptevents": 12,
    "nummicrolabs": 6,
    "numprocevents": 5,
    "totalnuminteract": 420,
    "admit_type": "EMERGENCY",
    "acuity_score": 120.0,
    "dx_sepsis": 1,
    "dx_cardiac": 0,
    "dx_respiratory": 1,
    "dx_trauma": 0
  },
  "forwarded": false
}
```

#### Example Response (`200 OK`)
```json
{
  "hadm_id": 109283,
  "timestamp": "2026-10-22T08:30:00.123456",
  "xgb_risk": 0.8241,
  "lstm_risk": 0.7915,
  "ensemble_risk": 0.8208,
  "alert_level": "CRITICAL",
  "edge_level": "WARNING",
  "triggers": ["dx_sepsis=1", "numlabs=210 > 200"],
  "window_size": 4
}
```

#### Status Codes
- `200 OK`: Inference successful.
- `401 Unauthorized`: Missing or invalid `X-API-Key`.
- `422 Unprocessable Entity`: Validation failure (missing required field, invalid data type, or out-of-range clinical value).

---

### 3. Multi-Agent Decision Pipeline
`POST /predict/agent`

Invokes the compiled LangGraph clinical decision workflow:
1. **Triage Agent** evaluates edge triggers and fast-track suitability.
2. **Risk Agent** calculates ensemble risk score.
3. **Explanation Agent** runs SHAP `TreeExplainer` and converts feature weights into clinical terminology.
4. **Router Agent** assigns priority and clinical recipient role via `config/routing_policy.json`.
5. **Audit Agent** persists state snapshot into SQLite.

#### Request Body
Same `EdgeReading` payload as `/predict`.

#### Example Response (`200 OK`)
```json
{
  "patient_id": 109283,
  "timestamp": "2026-10-22T08:30:00.123456",
  "run_id": "b1a23c4d-5678-90ab-cdef-1234567890ab",
  "edge_level": "WARNING",
  "edge_triggers": ["dx_sepsis=1", "numlabs=210 > 200"],
  "fast_track": false,
  "xgb_risk": 0.8241,
  "lstm_risk": 0.7915,
  "ensemble_risk": 0.8208,
  "feature_vector": [68.0, 1.0, 5.4, "..."],
  "shap_values": {
    "numlabs": 1.45,
    "numchartevents": 0.89,
    "acuity_score": 0.65
  },
  "top_contributors": ["numlabs", "numchartevents", "acuity_score"],
  "clinical_summary": "Elevated risk (82.1%) driven primarily by: laboratory test frequency (+1.45), charting frequency (+0.89), clinical acuity score (+0.65).",
  "explanation_method": "shap+template",
  "alert_priority": "high",
  "route_target": "attending_physician",
  "alert_id": "c2b34d5e-6789-01bc-def0-2345678901bc",
  "pending_approval": true,
  "clinician_action": null,
  "clinician_notes": null,
  "errors": [],
  "latency_ms": {
    "triage_agent": 0.85,
    "risk_agent": 12.4,
    "explanation_agent": 5.2,
    "router_agent": 0.04
  }
}
```

---

### 4. Ward Patient Registry
`GET /patients`

Retrieves latest risk metrics for all active hospital admissions monitored by the fog server.

#### Example Response (`200 OK`)
```json
{
  "109283": {
    "latest_risk": 0.8208,
    "alert_level": "CRITICAL",
    "last_updated": "2026-10-22T08:30:00.123456",
    "num_readings": 4
  },
  "104512": {
    "latest_risk": 0.0821,
    "alert_level": "NORMAL",
    "last_updated": "2026-10-22T08:29:15.000000",
    "num_readings": 12
  }
}
```

---

### 5. Patient Trajectory & History
`GET /patients/{hadm_id}`

Retrieves longitudinal risk progression and window size for a specific admission.

#### Example Request
```bash
curl -X GET http://localhost:8000/patients/109283
```

#### Example Response (`200 OK`)
```json
{
  "hadm_id": 109283,
  "window": 4,
  "history": [
    {
      "timestamp": "2026-10-22T08:00:00Z",
      "ensemble_risk": 0.4501,
      "alert_level": "WATCH"
    },
    {
      "timestamp": "2026-10-22T08:30:00Z",
      "ensemble_risk": 0.8208,
      "alert_level": "CRITICAL"
    }
  ]
}
```

#### Status Codes
- `200 OK`: History retrieved.
- `404 Not Found`: Patient `hadm_id` has no recorded history in the current server session.

---

### 6. Discharge Patient
`DELETE /patients/{hadm_id}`

Clears the patient from in-memory rolling window queues (`patient_windows`) and risk histories (`patient_risk_history`).

#### Example Response (`200 OK`)
```json
{
  "discharged": 109283
}
```

---

### 7. Human-in-the-Loop Pending Alerts
`GET /alerts/pending`

Returns all active clinical alerts where `pending_approval == 1`, ordered chronologically (newest first). Used by the dashboard to populate the alert review inbox.

#### Example Response (`200 OK`)
```json
{
  "alerts": [
    {
      "run_id": "b1a23c4d-5678-90ab-cdef-1234567890ab",
      "patient_id": 109283,
      "timestamp": "2026-10-22T08:30:00.123456",
      "edge_level": "WARNING",
      "fast_track": 0,
      "ensemble_risk": 0.8208,
      "alert_priority": "high",
      "route_target": "attending_physician",
      "clinical_summary": "Elevated risk (82.1%) driven primarily by: laboratory test frequency (+1.45)...",
      "pending_approval": 1,
      "clinician_action": null,
      "clinician_notes": null,
      "full_state": {
        "shap_values": { "numlabs": 1.45 },
        "top_contributors": ["numlabs", "numchartevents", "acuity_score"]
      }
    }
  ]
}
```

---

### 8. Clinician Alert Action
`POST /alerts/{alert_id}/action`

Records a verified clinical decision on an alert (`alert_id` corresponds to the `run_id`). Updates `pending_approval = 0` in SQLite and attaches the clinician's notes and selected action.

#### Request Body (`AlertAction`)
```json
{
  "action": "acknowledge",
  "notes": "Reviewed labs and vital signs. Sepsis bundle initiated and fluid bolus ordered."
}
```
*Permitted actions*: `"acknowledge"`, `"dismiss"`, `"escalate"`.

#### Example Response (`200 OK`)
```json
{
  "status": "success",
  "alert_id": "b1a23c4d-5678-90ab-cdef-1234567890ab",
  "action": "acknowledge"
}
```

#### Status Codes
- `200 OK`: Action committed to audit log.
- `404 Not Found`: Alert run ID does not exist in database.
- `422 Unprocessable Entity`: Invalid action string.
