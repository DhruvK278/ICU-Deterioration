import copy
import json

def test_predict_valid_request(client, valid_edge_payload):
    """
    Validates a well-formed request containing realistic clinical data.
    Asserts a 200 OK response and the exact structural shape/types of the output.
    """
    response = client.post("/predict", json=valid_edge_payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify response schema
    assert "hadm_id" in data
    assert data["hadm_id"] == valid_edge_payload["hadm_id"]
    
    assert "xgb_risk" in data
    assert isinstance(data["xgb_risk"], float)
    assert 0.0 <= data["xgb_risk"] <= 1.0
    
    assert "ensemble_risk" in data
    assert isinstance(data["ensemble_risk"], float)
    assert 0.0 <= data["ensemble_risk"] <= 1.0
    
    assert "alert_level" in data
    assert data["alert_level"] in ["CRITICAL", "WARNING", "WATCH", "NORMAL"]


def test_predict_missing_required_field(client, valid_edge_payload):
    """
    Validates that omitting a strongly required field (hadm_id) from the payload
    results in a 422 Unprocessable Entity error at the Pydantic validation layer.
    """
    payload = copy.deepcopy(valid_edge_payload)
    del payload["hadm_id"]
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    
    data = response.json()
    errors = data.get("detail", [])
    assert len(errors) > 0
    assert errors[0]["loc"] == ["body", "hadm_id"]
    assert errors[0]["type"] == "missing"


def test_predict_wrong_data_type(client, valid_edge_payload):
    """
    Validates that providing a strictly incorrect data type (string instead of float)
    results in a 422 error, explicitly calling out the failed type coercion.
    """
    payload = copy.deepcopy(valid_edge_payload)
    payload["reading"]["acuity_score"] = "invalid_score_string"
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    
    data = response.json()
    errors = data.get("detail", [])
    assert len(errors) > 0
    assert errors[0]["loc"] == ["body", "reading", "acuity_score"]


def test_predict_out_of_range_value(client, valid_edge_payload):
    """
    Validates bounds enforcement from the model schema.
    The 'age' field in VitalsReading enforces le=120. Sending 150 should
    trigger a graceful 422 validation error, not a crash during ML inference.
    """
    payload = copy.deepcopy(valid_edge_payload)
    payload["reading"]["age"] = 150
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    
    data = response.json()
    errors = data.get("detail", [])
    assert len(errors) > 0
    assert errors[0]["loc"] == ["body", "reading", "age"]
    # Pydantic v2 uses 'less_than_equal', v1 uses 'value_error.number.not_le'
    assert "le" in errors[0]["type"] or "less_than" in errors[0]["type"]


def test_predict_malformed_json(client):
    """
    Validates that a completely malformed JSON request body is caught securely
    at the framework boundary before any custom code executes.
    """
    malformed_json = '{"hadm_id": 1234, "timestamp": "2023-01-01'
    
    response = client.post(
        "/predict",
        content=malformed_json,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    
    data = response.json()
    errors = data.get("detail", [])
    assert len(errors) > 0
    # Pydantic v2 throws json_invalid
    assert "json" in errors[0]["type"]


def test_predict_unvalidated_field_behavior(client, valid_edge_payload):
    """
    QA Insight: The 'admit_type' field is defined as a basic 'str' in the Pydantic 
    model with a default of "EMERGENCY". It does not enforce any specific categories
    (e.g., URGENT, NEWBORN) through Enums or regex patterns at the validation level.
    
    This test verifies that providing an unexpected arbitrary string does not break 
    the endpoint, as the feature extraction logic falls back safely to 0.0 for known 
    one-hot encoded columns.
    
    Recommendation: For tighter safety bounds, 'admit_type' should use a Pydantic Enum.
    """
    payload = copy.deepcopy(valid_edge_payload)
    payload["reading"]["admit_type"] = "NOT_A_REAL_ADMISSION_TYPE_123!@#"
    
    response = client.post("/predict", json=payload)
    
    # Gracefully handles it without crashing
    assert response.status_code == 200
    
    data = response.json()
    assert "ensemble_risk" in data
