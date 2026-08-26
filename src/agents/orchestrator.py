"""
src/agents/orchestrator.py

Compiles the LangGraph StateGraph, wiring the agents together with conditional edges.
This acts as the main entry point for the fog server to trigger the multi-agent pipeline.
"""

import logging
from langgraph.graph import StateGraph, END

from .agent_state import AgentState
from .triage_agent import triage_agent
from .risk_agent import risk_agent
from .explanation_agent import explanation_agent
from .router_agent import router_agent
from .audit_agent import log_alert_state

log = logging.getLogger(__name__)

def should_fast_track(state: AgentState):
    """
    Conditional edge router: bypasses ML scoring for critical fast-track patients.
    """
    if state.get("fast_track"):
        log.info(f"[{state.get('patient_id')}] Fast-tracking -> skipping ML scoring")
        return "router"
    return "risk"

def build_graph():
    """
    Constructs and compiles the LangGraph pipeline.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Add agent nodes
    workflow.add_node("triage", triage_agent)
    workflow.add_node("risk", risk_agent)
    workflow.add_node("explanation", explanation_agent)
    workflow.add_node("router", router_agent)
    workflow.add_node("audit", log_alert_state)
    
    # 2. Define flow (edges)
    workflow.set_entry_point("triage")
    
    # Conditional logic after triage
    workflow.add_conditional_edges(
        "triage",
        should_fast_track,
        {
            "router": "router",  # Fast-track path
            "risk": "risk"       # Normal ML path
        }
    )
    
    # Linear path for standard flow
    workflow.add_edge("risk", "explanation")
    workflow.add_edge("explanation", "router")
    
    # Convergence path
    workflow.add_edge("router", "audit")
    workflow.add_edge("audit", END)
    
    # Compile graph
    return workflow.compile()

# The pre-compiled graph to be imported by the API server
clinical_orchestrator = build_graph()
