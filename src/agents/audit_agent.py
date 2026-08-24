"""
src/agents/audit_agent.py

Logs all LangGraph state decisions to a local SQLite database
for regulatory compliance and monitoring.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .agent_state import AgentState

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "audit_log.db"

def init_db():
    """Create the audit log tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Main alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            run_id TEXT PRIMARY KEY,
            patient_id INTEGER,
            timestamp TEXT,
            edge_level TEXT,
            fast_track BOOLEAN,
            ensemble_risk REAL,
            alert_priority TEXT,
            route_target TEXT,
            clinical_summary TEXT,
            pending_approval BOOLEAN,
            clinician_action TEXT,
            clinician_notes TEXT,
            full_state_json TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    log.info(f"Audit DB initialized at {DB_PATH}")

def log_alert_state(state: AgentState) -> AgentState:
    """
    Saves the final agent state to the SQLite database.
    This acts as the final node in the LangGraph.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Prepare serializable state
    state_to_save = dict(state)
    
    # Extract values for columns safely
    run_id = state.get("run_id")
    patient_id = state.get("patient_id")
    timestamp = state.get("timestamp")
    edge_level = state.get("edge_level")
    fast_track = state.get("fast_track", False)
    ensemble_risk = state.get("ensemble_risk")
    alert_priority = state.get("alert_priority")
    route_target = state.get("route_target")
    clinical_summary = state.get("clinical_summary")
    pending_approval = state.get("pending_approval", True)
    clinician_action = state.get("clinician_action")
    clinician_notes = state.get("clinician_notes")
    
    cursor.execute('''
        INSERT OR REPLACE INTO alerts (
            run_id, patient_id, timestamp, edge_level, fast_track, 
            ensemble_risk, alert_priority, route_target, clinical_summary,
            pending_approval, clinician_action, clinician_notes, full_state_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        run_id, patient_id, timestamp, edge_level, fast_track, 
        ensemble_risk, alert_priority, route_target, clinical_summary,
        pending_approval, clinician_action, clinician_notes, json.dumps(state_to_save)
    ))
    
    conn.commit()
    conn.close()
    
    log.info(f"[Audit] Logged run {run_id} for patient {patient_id}")
    
    return state

# Initialize DB on module load
init_db()
