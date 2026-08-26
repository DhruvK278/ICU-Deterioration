"""
src/agents/explanation_agent.py

Calculates exact SHAP values for the patient's feature vector against the XGBoost model.
Generates a human-readable clinical explanation template.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import shap

from .agent_state import AgentState
from src.fog.fog_server import xgb_model, FEATURE_NAMES

log = logging.getLogger(__name__)

# Load clinical descriptions mapping
ROOT = Path(__file__).resolve().parents[3]
try:
    with open(ROOT / "config" / "feature_descriptions.json") as f:
        FEATURE_DESCS = json.load(f)
except Exception as e:
    log.warning("Could not load feature_descriptions.json, using raw names.")
    FEATURE_DESCS = {}

# Initialize TreeExplainer (fast, exact for XGBoost)
explainer = shap.TreeExplainer(xgb_model)

def explanation_agent(state: AgentState) -> Dict[str, Any]:
    """
    Computes SHAP values and generates a template-based explanation.
    """
    start_time = time.time()
    
    # Skip if fast-tracked or if risk score is very low (save compute)
    if state.get("fast_track") or state.get("ensemble_risk", 0) < 0.3:
        return {
            "latency_ms": {**state.get("latency_ms", {}), "explanation_agent": (time.time() - start_time) * 1000}
        }
        
    feature_vector = state.get("feature_vector")
    if not feature_vector:
        return {}
        
    try:
        # 1. Compute SHAP values
        X_patient = np.array(feature_vector).reshape(1, -1)
        shap_vals = explainer.shap_values(X_patient)[0]
        
        # 2. Map SHAP values to feature names
        shap_dict = {name: float(val) for name, val in zip(FEATURE_NAMES, shap_vals)}
        
        # 3. Extract top 3 contributing positive features (driving risk UP)
        top_features = sorted(
            [(k, v) for k, v in shap_dict.items() if v > 0], 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        top_contributors = [f[0] for f in top_features]
        
        # 4. Generate Clinical Summary (Template-based for now)
        if top_features:
            reasons = []
            for feat, val in top_features:
                desc = FEATURE_DESCS.get(feat, feat.replace("_", " "))
                reasons.append(f"{desc} (+{val:.2f})")
                
            risk_pct = state.get("ensemble_risk", 0) * 100
            summary = f"Elevated risk ({risk_pct:.1f}%) driven primarily by: " + ", ".join(reasons) + "."
        else:
            summary = "Risk elevated, but specific driving factors are widely distributed."
            
        latency = (time.time() - start_time) * 1000
        
        return {
            "shap_values": shap_dict,
            "top_contributors": top_contributors,
            "clinical_summary": summary,
            "explanation_method": "shap+template",
            "latency_ms": {**state.get("latency_ms", {}), "explanation_agent": latency}
        }
        
    except Exception as e:
        log.error(f"[{state['patient_id']}] Explanation Agent failed: {str(e)}")
        errors = state.get("errors", [])
        errors.append(f"ExplanationAgent Error: {str(e)}")
        
        return {
            "errors": errors,
            "latency_ms": {**state.get("latency_ms", {}), "explanation_agent": (time.time() - start_time) * 1000}
        }
