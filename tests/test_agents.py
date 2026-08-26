"""
tests/test_agents.py

Tests the full LangGraph Orchestrator pipeline to verify that
agents pass state correctly through conditional edges and into the DB.
"""

import sys
import uuid
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.agents.agent_state import create_initial_state
from src.agents.orchestrator import clinical_orchestrator

def run_test():
    print("=== Testing LangGraph Pipeline ===")
    
    # Mock a reading (High Risk case)
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
        edge_level="WARNING",
        edge_triggers=["dx_sepsis=1", "numlabs=260 > 200"],
        raw_reading=raw_reading
    )
    
    print(f"Invoking graph for Patient {state['patient_id']}...")
    
    # Run the graph
    final_state = clinical_orchestrator.invoke(state)
    
    print("\n--- Final Output State ---")
    print(f"Fast Track:       {final_state.get('fast_track')}")
    print(f"Ensemble Risk:    {final_state.get('ensemble_risk')}")
    print(f"Alert Priority:   {final_state.get('alert_priority')}")
    print(f"Route Target:     {final_state.get('route_target')}")
    print(f"Clinical Summary: {final_state.get('clinical_summary')}")
    
    if final_state.get("errors"):
        print(f"\nERRORS ENCOUNTERED:\n{final_state['errors']}")
        
    print("\nLatencies:", json.dumps(final_state.get("latency_ms"), indent=2))
    print("=== Test Complete ===")

if __name__ == "__main__":
    run_test()
