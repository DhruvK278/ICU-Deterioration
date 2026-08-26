"""
src/agents/risk_agent.py

Wraps the existing XGBoost and LSTM models into a LangGraph agent.
Extracts features, runs inference, and updates the state.
"""

import time
import logging
from typing import Dict, Any
from .agent_state import AgentState

# We import the existing model logic directly from the fog server
# to avoid code duplication and ensure we use the exact same feature engineering.
from src.fog.fog_server import (
    VitalsReading, 
    extract_features, 
    score_xgb, 
    score_lstm, 
    patient_windows
)

log = logging.getLogger(__name__)

def risk_agent(state: AgentState) -> Dict[str, Any]:
    """
    Computes ML risk scores if the patient is not fast-tracked.
    """
    start_time = time.time()
    
    # If the patient was fast-tracked by triage, skip ML scoring
    if state.get("fast_track"):
        return {
            "latency_ms": {**state.get("latency_ms", {}), "risk_agent": (time.time() - start_time) * 1000}
        }
        
    hadm_id = state["patient_id"]
    raw_reading = state.get("raw_reading", {})
    
    try:
        # Validate and extract using existing logic
        reading = VitalsReading(**raw_reading)
        features_df = extract_features(reading)
        
        # 1. Score XGBoost
        xgb_risk = score_xgb(features_df)
        
        # 2. Score LSTM (Stateful rolling window)
        feature_vector = features_df.values[0].tolist()
        patient_windows[hadm_id].append(feature_vector)
        lstm_risk = score_lstm(features_df, patient_windows[hadm_id])
        
        # 3. Ensemble Risk
        if lstm_risk is not None:
            ensemble_risk = 0.90 * xgb_risk + 0.10 * lstm_risk
        else:
            ensemble_risk = xgb_risk
            
        latency = (time.time() - start_time) * 1000
        
        return {
            "xgb_risk": round(xgb_risk, 4),
            "lstm_risk": round(lstm_risk, 4) if lstm_risk else None,
            "ensemble_risk": round(ensemble_risk, 4),
            "feature_vector": feature_vector,
            "latency_ms": {**state.get("latency_ms", {}), "risk_agent": latency}
        }
        
    except Exception as e:
        log.error(f"[{hadm_id}] Risk Agent failed: {str(e)}")
        errors = state.get("errors", [])
        errors.append(f"RiskAgent Error: {str(e)}")
        
        return {
            "errors": errors,
            "latency_ms": {**state.get("latency_ms", {}), "risk_agent": (time.time() - start_time) * 1000}
        }
