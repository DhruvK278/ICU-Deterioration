"""
src/agents/triage_agent.py

The first node in the LangGraph pipeline. 
Checks the edge-generated alerts and decides whether to fast-track
the patient (bypassing the heavy ML models) or proceed normally.
"""

import time
import logging
from typing import Dict, Any
from .agent_state import AgentState

log = logging.getLogger(__name__)

def triage_agent(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the edge triggers and sets the fast_track flag.
    Returns a dict with the fields to update in the state.
    """
    start_time = time.time()
    
    edge_level = state.get("edge_level", "NORMAL")
    triggers = state.get("edge_triggers", [])
    
    fast_track = False
    
    # Fast-track logic: If it's a critical alert and has multiple severe triggers, bypass ML
    if edge_level == "CRITICAL" and len(triggers) >= 2:
        log.warning(f"[{state['patient_id']}] Fast-tracking CRITICAL patient. Triggers: {triggers}")
        fast_track = True
        
    latency = (time.time() - start_time) * 1000
    
    return {
        "fast_track": fast_track,
        "latency_ms": {**state.get("latency_ms", {}), "triage_agent": latency}
    }
