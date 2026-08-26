"""
tests/test_agents.py

Tests the Phase 2 Analytical Agents in isolation to ensure they
can parse state, run models, compute SHAP, and return expected updates.
"""

import sys
import uuid
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.agents.agent_state import create_initial_state
from src.agents.triage_agent import triage_agent
from src.agents.risk_agent import risk_agent
from src.agents.explanation_agent import explanation_agent

def run_test():
    print("=== Testing Agents Pipeline ===")
    
    # 1. Mock a severe reading
    raw_reading = {
        "age": 83, "gender": 0, "losdays": 38.0,
        "numchartevents": 650, "numlabs": 260, "numprocs": 28,
        "numinput": 190, "numoutput": 95, "numtransfers": 8,
        "numrx": 38, "numnotes": 42, "numdiagnosis": 20,
        "numcallouts": 4, "numcptevents": 45, "nummicrolabs": 22,
        "numprocevents": 32, "totalnuminteract": 850,
        "admit_type": "EMERGENCY", "acuity_score": 900.0,
        "dx_sepsis": 1, "dx_cardiac": 1, "dx_respiratory": 0, "dx_trauma": 0,
    }
    
    # Init state
    state = create_initial_state(
        patient_id=9001,
        timestamp=datetime.utcnow().isoformat(),
        run_id=str(uuid.uuid4()),
        edge_level="WARNING",  # Start with warning to test ML scoring
        edge_triggers=["dx_sepsis=1", "numlabs=260 > 200"],
        raw_reading=raw_reading
    )
    
    print("\n--- 1. Triage Agent ---")
    triage_update = triage_agent(state)
    state.update(triage_update)
    print(f"Fast Track: {state.get('fast_track')}")
    
    print("\n--- 2. Risk Agent ---")
    risk_update = risk_agent(state)
    state.update(risk_update)
    print(f"XGB Risk: {state.get('xgb_risk')}")
    print(f"LSTM Risk: {state.get('lstm_risk')}")
    print(f"Ensemble Risk: {state.get('ensemble_risk')}")
    if state.get("errors"):
        print(f"Errors: {state['errors']}")
        
    print("\n--- 3. Explanation Agent ---")
    exp_update = explanation_agent(state)
    state.update(exp_update)
    print(f"Top Contributors: {state.get('top_contributors')}")
    print(f"Clinical Summary: {state.get('clinical_summary')}")
    
    print("\n=== Test Complete ===")
    print("Latencies:", json.dumps(state.get("latency_ms"), indent=2))

if __name__ == "__main__":
    run_test()
