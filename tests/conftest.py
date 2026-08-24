import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from src.fog.fog_server import app


@pytest.fixture
def client():
    """
    Returns a TestClient instance for the FastAPI application.
    This uses the real application, including the loaded models,
    providing genuine end-to-end inference testing.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_vitals_payload():
    """
    A dictionary representing a baseline valid VitalsReading.
    Contains typical, realistic clinical values.
    """
    return {
        "age": 65.5,
        "gender": 1,
        "losdays": 4.2,
        "numchartevents": 150,
        "numlabs": 24,
        "numprocs": 1,
        "numinput": 3,
        "numoutput": 5,
        "numtransfers": 1,
        "numrx": 12,
        "numnotes": 3,
        "numdiagnosis": 8,
        "numcallouts": 0,
        "numcptevents": 2,
        "nummicrolabs": 1,
        "numprocevents": 1,
        "totalnuminteract": 210,
        "admit_type": "EMERGENCY",
        "acuity_score": 15.0,
        "dx_sepsis": 0,
        "dx_cardiac": 1,
        "dx_respiratory": 0,
        "dx_trauma": 0
    }


@pytest.fixture
def valid_edge_payload(valid_vitals_payload):
    """
    A dictionary representing a baseline valid EdgeReading wrapping the vitals.
    """
    return {
        "hadm_id": 123456,
        "timestamp": datetime.utcnow().isoformat(),
        "level": "WATCH",
        "level_int": 2,
        "triggers": ["abnormal HR", "low SpO2"],
        "reading": valid_vitals_payload,
        "forwarded": False
    }
