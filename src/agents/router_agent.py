"""
src/agents/router_agent.py

Evaluates the computed risk score against configurable hospital policies
to determine the alert priority and routing target (e.g., attending vs nurse).
"""

import time
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, Any

from .agent_state import AgentState

log = logging.getLogger(__name__)

# Load routing policies
ROOT = Path(__file__).resolve().parents[2]
try:
    with open(ROOT / "config" / "routing_policy.json") as f:
        ROUTING_CONFIG = json.load(f)
    POLICIES = ROUTING_CONFIG.get("policies", [])
except Exception as e:
    log.warning("Could not load routing_policy.json, using defaults.")
    POLICIES = [{"default": True, "priority": "log_only", "target": "log"}]

def router_agent(state: AgentState) -> Dict[str, Any]:
    """
    Determines alert routing based on risk and policies.
    """
    start_time = time.time()
    
    ensemble_risk = state.get("ensemble_risk") or 0.0
    edge_level = state.get("edge_level", "NORMAL")
    
    priority = "log_only"
    target = "log"
    
    for policy in POLICIES:
        if policy.get("default"):
            priority = policy.get("priority", priority)
            target = policy.get("target", target)
            break
            
        min_risk = policy.get("min_risk", 0.0)
        edge_levels = policy.get("edge_levels", [])
        
        # Check if conditions match
        risk_match = ensemble_risk >= min_risk
        level_match = edge_level in edge_levels if edge_levels else True
        
        if risk_match and level_match:
            priority = policy.get("priority")
            target = policy.get("target")
            break
            
    latency = (time.time() - start_time) * 1000
    
    return {
        "alert_priority": priority,
        "route_target": target,
        "alert_id": str(uuid.uuid4()),
        "latency_ms": {**state.get("latency_ms", {}), "router_agent": latency}
    }
