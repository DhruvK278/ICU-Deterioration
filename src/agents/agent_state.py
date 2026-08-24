"""
src/agents/agent_state.py

Defines the TypedDict schema for the shared state passed between
all agents in the LangGraph orchestrator.
"""

from typing import TypedDict, Optional, Literal, List, Dict, Any


class AgentState(TypedDict):
    # Identity & Context
    patient_id: int
    timestamp: str
    run_id: str                          # UUID for deduplication

    # 1. Triage Agent Outputs
    edge_level: Literal["CRITICAL", "WARNING", "WATCH", "NORMAL"]
    edge_triggers: List[str]
    fast_track: bool                     # If True, bypass ML scoring

    # 2. Risk Stratification Agent Outputs
    xgb_risk: Optional[float]
    lstm_risk: Optional[float]
    ensemble_risk: Optional[float]
    feature_vector: Optional[List[float]]

    # 3. Explanation Agent Outputs
    shap_values: Optional[Dict[str, float]]   # feature_name -> shap_value
    top_contributors: Optional[List[str]]     # top 3 features driving risk
    clinical_summary: Optional[str]           # plain-language explanation text
    explanation_method: Optional[str]         # e.g., "shap+template", "shap+llm"

    # 4. Alert Router Agent Outputs
    alert_priority: Optional[Literal["critical", "high", "medium", "low", "log_only"]]
    route_target: Optional[Literal["attending_physician", "nurse", "charge_nurse", "log"]]
    alert_id: Optional[str]                  # UUID for tracking

    # 5. Human-in-the-loop (Updated later by dashboard via API)
    pending_approval: bool
    clinician_action: Optional[Literal["acknowledge", "dismiss", "escalate"]]
    clinician_notes: Optional[str]

    # Metadata & Telemetry
    errors: List[str]                         # accumulated errors across agents
    latency_ms: Dict[str, float]              # agent_name -> processing_time_ms
    
    # The raw incoming reading dict (used to extract features if needed)
    raw_reading: Dict[str, Any]


def create_initial_state(
    patient_id: int, 
    timestamp: str, 
    run_id: str, 
    edge_level: str, 
    edge_triggers: List[str], 
    raw_reading: Dict[str, Any]
) -> AgentState:
    """Helper to initialize the state with default values."""
    
    # Enforce literal typing at runtime for initial state
    if edge_level not in ["CRITICAL", "WARNING", "WATCH", "NORMAL"]:
        edge_level = "NORMAL"
        
    return {
        "patient_id": patient_id,
        "timestamp": timestamp,
        "run_id": run_id,
        "edge_level": edge_level,
        "edge_triggers": edge_triggers,
        "fast_track": False,
        "xgb_risk": None,
        "lstm_risk": None,
        "ensemble_risk": None,
        "feature_vector": None,
        "shap_values": None,
        "top_contributors": None,
        "clinical_summary": None,
        "explanation_method": None,
        "alert_priority": None,
        "route_target": None,
        "alert_id": None,
        "pending_approval": False,
        "clinician_action": None,
        "clinician_notes": None,
        "errors": [],
        "latency_ms": {},
        "raw_reading": raw_reading,
    }
