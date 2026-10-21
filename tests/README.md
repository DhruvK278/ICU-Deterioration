# Testing & QA Suite

This directory contains automated end-to-end and integration tests for the ICU Deterioration Prediction system, covering both the FastAPI ward inference server (`src/fog/fog_server.py`) and the LangGraph multi-agent decision support pipeline (`src/agents/`).

The tests are designed to exercise genuine inference logic, validation schemas, and agent state transitions without brittle mocking.

---

## Test Suites Overview

### 1. API Contract & Validation Suite (`test_inference.py`)
Tests the FastAPI `/predict` endpoint against strict Pydantic payload definitions and model inference boundaries:

- **Valid Payload (`test_predict_valid_request`)**: Asserts 200 OK happy path, verifying response JSON schema, types, and that `xgb_risk` and `ensemble_risk` fall strictly within $[0.0, 1.0]$.
- **Missing Required Fields (`test_predict_missing_required_field`)**: Verifies that omitting required identifiers (`hadm_id`) raises a 422 Unprocessable Entity error at the Pydantic schema validation boundary.
- **Data Type Enforcement (`test_predict_wrong_data_type`)**: Tests type coercion safety by injecting string values into numeric clinical attributes (`acuity_score`), asserting rejection with 422.
- **Clinical Range Enforcement (`test_predict_out_of_range_value`)**: Tests boundary constraints (`age` > 120 or negative values), verifying rejection with 422 to prevent model poisoning.
- **Malformed Payload Handling (`test_predict_malformed_json`)**: Validates framework handling of invalid JSON strings.
- **Edge Case Fallback (`test_predict_unvalidated_field_behavior`)**: Verifies graceful handling of non-standard categorical entries (e.g. `admit_type`).

### 2. Multi-Agent Pipeline Suite (`test_agents.py`)
Tests the LangGraph `StateGraph` clinical decision orchestrator (`src/agents/orchestrator.py`):

- **End-to-End Orchestration**: Initializes an `AgentState` for an acute patient, invoking the compiled graph.
- **Triage & Fast-Track Routing**: Verifies condition evaluations (`should_fast_track`) and edge trigger processing.
- **Dual-Model Risk Scoring**: Checks that the Risk Agent properly executes XGBoost and LSTM ensemble models.
- **SHAP Explanation & Clinical Translation**: Validates TreeExplainer execution, top contributor extraction, and clinical feature description mapping via `config/feature_descriptions.json`.
- **Policy Routing**: Evaluates alert priority and routing target assignment via `config/routing_policy.json`.
- **Audit Persistence**: Confirms that state transitions and metadata are written to the local SQLite audit log (`data/audit_log.db`).

### 3. Ward Server Integration Test (`src/fog/test_fog_integration.py`)
A standalone end-to-end integration test that spins up the Fog server on a background thread (`uvicorn`) and executes realistic edge detector simulation requests:
- Health check validation (`/health`)
- Normal vs critical patient trajectory simulation
- Rolling window expansion (`patient_windows`)
- Active patient registry query (`/patients`)
- Patient discharge state cleanup (`DELETE /patients/{hadm_id}`)

---

## How to Run Tests

### Run All Pytest Suites
From the project root:
```bash
pytest tests/ -v
```

### Run Individual Suites
Run only API boundary tests:
```bash
pytest tests/test_inference.py -v
```

Run only the Multi-Agent pipeline test:
```bash
pytest tests/test_agents.py -v
```
Or execute directly as a script:
```bash
python tests/test_agents.py
```

### Run Fog Gateway Integration Test
```bash
python src/fog/test_fog_integration.py
```
