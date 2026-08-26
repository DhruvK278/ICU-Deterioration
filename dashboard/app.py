"""
dashboard/app.py

Real-time ICU risk monitoring UI built with Streamlit.

Shows:
  - Ward overview: all active patients with colour-coded risk
  - Live risk score per patient (auto-refreshes every 10s)
  - Alert history and trend chart
  - Manual patient lookup
  - Multi-Agent Human-in-the-Loop Alert Queue (NEW)

Run locally:
    streamlit run dashboard/app.py

"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Config
FOG_URL       = os.getenv("FOG_URL", "http://localhost:8000")
FOG_API_KEY   = os.getenv("FOG_API_KEY", "")
REFRESH_SECS  = 10
MAX_HISTORY   = 20

# Auth headers for fog server
FOG_HEADERS = {}
if FOG_API_KEY:
    FOG_HEADERS["X-API-Key"] = FOG_API_KEY

# Risk level colours
LEVEL_COLORS = {
    "CRITICAL": "#E24B4A",
    "WARNING":  "#EF9F27",
    "WATCH":    "#378ADD",
    "NORMAL":   "#1D9E75",
}
LEVEL_BG = {
    "CRITICAL": "#FCEBEB",
    "WARNING":  "#FAEEDA",
    "WATCH":    "#E6F1FB",
    "NORMAL":   "#E1F5EE",
}

st.set_page_config(
    page_title="ICU Deterioration Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
.risk-card {
    padding: 1rem 1.25rem;
    border-radius: 12px;
    border: 0.5px solid #e0e0e0;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 12px;
    color: #888;
    margin-bottom: 2px;
}
.metric-value {
    font-size: 22px;
    font-weight: 500;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
}
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# API helpers
@st.cache_data(ttl=REFRESH_SECS)
def fetch_patients() -> dict:
    try:
        r = requests.get(f"{FOG_URL}/patients", headers=FOG_HEADERS, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=REFRESH_SECS)
def fetch_patient_history(hadm_id: int) -> dict:
    try:
        r = requests.get(f"{FOG_URL}/patients/{hadm_id}", headers=FOG_HEADERS, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=5)
def fetch_health() -> dict:
    try:
        r = requests.get(f"{FOG_URL}/health", headers=FOG_HEADERS, timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=5)
def fetch_pending_alerts() -> list:
    try:
        r = requests.get(f"{FOG_URL}/alerts/pending", headers=FOG_HEADERS, timeout=3)
        if r.status_code == 200:
            return r.json().get("alerts", [])
    except Exception:
        pass
    return []


def resolve_alert(alert_id: str, action: str):
    """Sends action to the human-in-the-loop callback endpoint."""
    payload = {"action": action, "notes": "Resolved via Dashboard"}
    try:
        requests.post(f"{FOG_URL}/alerts/{alert_id}/action", json=payload, headers=FOG_HEADERS, timeout=3)
    except Exception as e:
        st.error(f"Failed to update alert: {e}")


def send_test_patient(hadm_id: int, risk_level: str, use_agent: bool = False):
    """Send a synthetic patient reading to the fog server for demo purposes."""
    profiles = {
        "high": {
            "age": 83, "gender": 0, "losdays": 38.0,
            "numchartevents": 650, "numlabs": 260, "numprocs": 28,
            "numinput": 190, "numoutput": 95, "numtransfers": 8,
            "numrx": 38, "numnotes": 42, "numdiagnosis": 20,
            "numcallouts": 4, "numcptevents": 45, "nummicrolabs": 22,
            "numprocevents": 32, "totalnuminteract": 850,
            "admit_type": "EMERGENCY", "acuity_score": 900.0,
            "dx_sepsis": 1, "dx_cardiac": 1, "dx_respiratory": 0, "dx_trauma": 0,
        },
        "medium": {
            "age": 65, "gender": 1, "losdays": 8.0,
            "numchartevents": 200, "numlabs": 80, "numprocs": 10,
            "numinput": 40, "numoutput": 35, "numtransfers": 2,
            "numrx": 15, "numnotes": 12, "numdiagnosis": 8,
            "numcallouts": 1, "numcptevents": 15, "nummicrolabs": 5,
            "numprocevents": 8, "totalnuminteract": 200,
            "admit_type": "EMERGENCY", "acuity_score": 250.0,
            "dx_sepsis": 0, "dx_cardiac": 1, "dx_respiratory": 0, "dx_trauma": 0,
        },
        "low": {
            "age": 45, "gender": 1, "losdays": 2.0,
            "numchartevents": 60, "numlabs": 25, "numprocs": 3,
            "numinput": 10, "numoutput": 8, "numtransfers": 0,
            "numrx": 5, "numnotes": 4, "numdiagnosis": 3,
            "numcallouts": 0, "numcptevents": 5, "nummicrolabs": 1,
            "numprocevents": 2, "totalnuminteract": 45,
            "admit_type": "ELECTIVE", "acuity_score": 55.0,
            "dx_sepsis": 0, "dx_cardiac": 0, "dx_respiratory": 0, "dx_trauma": 0,
        },
    }
    payload = {
        "hadm_id":   hadm_id,
        "timestamp": datetime.utcnow().isoformat(),
        "level":     "CRITICAL" if risk_level == "high" else "WARNING" if risk_level == "medium" else "NORMAL",
        "level_int": 3 if risk_level == "high" else 2 if risk_level == "medium" else 0,
        "triggers":  ["dx_sepsis=1", "numlabs=260"] if risk_level == "high" else [],
        "reading":   profiles[risk_level],
        "forwarded": False,
    }
    
    endpoint = f"{FOG_URL}/predict/agent" if use_agent else f"{FOG_URL}/predict"
    
    try:
        r = requests.post(endpoint, json=payload, headers=FOG_HEADERS, timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# Sidebar
with st.sidebar:
    st.markdown("### ICU Monitor")
    st.markdown(f"**Fog server:** `{FOG_URL}`")

    health = fetch_health()
    if health.get("status") == "ok":
        st.success("Fog server online")
        st.markdown(f"XGBoost: {'✓' if health.get('xgb_loaded') else '✗'}")
        st.markdown(f"LSTM: {'✓' if health.get('lstm_loaded') else '✗'}")
        st.markdown(f"Uptime: {int(health.get('uptime_s', 0))}s")
        st.markdown(f"Patients seen: {health.get('patients_seen', 0)}")
    else:
        st.error("Fog server offline")
        st.markdown("Start with:")
        st.code("uvicorn src.fog.fog_server:app\n--port 8000")

    st.divider()
    st.markdown("### Add demo patient")
    demo_id   = st.number_input("Patient ID", value=9001, step=1)
    demo_risk = st.selectbox("Risk profile", ["high", "medium", "low"])
    use_agent = st.checkbox("Route through LangGraph Agent", value=True)
    
    if st.button("Send to fog ↗"):
        result = send_test_patient(int(demo_id), demo_risk, use_agent)
        if result:
            if use_agent:
                st.success(f"Agent Processed! Routed to: {result.get('route_target', 'N/A')}")
            else:
                st.success(f"Scored: {result.get('ensemble_risk', 0):.3f} → {result.get('alert_level', 'N/A')}")
            st.cache_data.clear()
            
        else:
            st.error("Failed — is fog running?")

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        st.markdown(f"Refreshing every {REFRESH_SECS}s")


# Main content
st.markdown("## Intelligent Clinical Decision Support System")
st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")

patients = fetch_patients()
pending_alerts = fetch_pending_alerts()

if not patients and not pending_alerts:
    st.info("No active patients. Use the sidebar to add demo patients, or start the edge simulator.")
    st.code("python src/edge/edge_detector.py --fog-url http://localhost:8000")
else:
    # Metrics
    total    = len(patients)
    critical = sum(1 for p in patients.values() if p["alert_level"] == "CRITICAL")
    warning  = sum(1 for p in patients.values() if p["alert_level"] == "WARNING")
    watch    = sum(1 for p in patients.values() if p["alert_level"] == "WATCH")
    normal   = sum(1 for p in patients.values() if p["alert_level"] == "NORMAL")
    pending_count = len(pending_alerts)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total patients", total)
    c2.metric("Critical", critical)
    c3.metric("Warning",  warning)
    c4.metric("Watch",    watch)
    c5.metric("Normal",   normal)
    c6.metric("Pending Alerts", pending_count, delta_color="inverse")

    st.divider()

    tab1, tab2 = st.tabs(["WARD OVERVIEW", f"ALERT QUEUE ({pending_count})"])

    with tab2:
        st.markdown("### Human-in-the-Loop Alert Queue")
        
        if not pending_alerts:
            st.success("🎉 No pending alerts! All patients are stable or alerts have been acknowledged.")
        else:
            for alert in pending_alerts:
                state = alert.get("full_state", {})
                risk = state.get("ensemble_risk") or 0.0
                edge_level = state.get("edge_level", "NORMAL")
                summary = state.get("clinical_summary", "No SHAP explanation provided.")
                priority = state.get("alert_priority", "low").upper()
                target = state.get("route_target", "Unknown").replace("_", " ").title()
                patient_id = state.get("patient_id", "Unknown")
                fast_track = state.get("fast_track", False)
                
                # Determine colors based on priority
                color = LEVEL_COLORS.get("CRITICAL" if priority == "CRITICAL" else "WARNING" if priority == "HIGH" else "NORMAL")
                bg = LEVEL_BG.get("CRITICAL" if priority == "CRITICAL" else "WARNING" if priority == "HIGH" else "NORMAL")
                
                with st.container():
                    st.markdown(f"""
                    <div style="background:{bg}; border-left:6px solid {color}; padding:16px 20px; border-radius:0 8px 8px 0; margin-bottom:12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <h4 style="margin-top:0; color:{color};">🚨 Patient {patient_id} — Priority: {priority}</h4>
                        <p style="margin-bottom:8px; font-size:16px;"><strong>Target Clinician:</strong> {target}</p>
                        <p style="margin-bottom:8px; font-size:16px;"><strong>Ensemble Risk Score:</strong> {risk:.3f} 
                            {"<span style='color:red;font-weight:bold;'>(FAST-TRACKED)</span>" if fast_track else ""}
                        </p>
                        <p style="margin-bottom:8px; font-size:16px;"><strong>AI Explanation (SHAP):</strong> {summary}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action Buttons
                    col1, col2, col3, _ = st.columns([1,1,1,6])
                    if col1.button("✅ Acknowledge", key=f"ack_{alert['run_id']}"):
                        resolve_alert(alert['run_id'], "acknowledge")
                        st.cache_data.clear()
                        st.rerun()
                    if col2.button("❌ Dismiss", key=f"dis_{alert['run_id']}"):
                        resolve_alert(alert['run_id'], "dismiss")
                        st.cache_data.clear()
                        st.rerun()
                    if col3.button("⚠️ Escalate", key=f"esc_{alert['run_id']}"):
                        resolve_alert(alert['run_id'], "escalate")
                        st.cache_data.clear()
                        st.rerun()
                    
                    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)


    with tab1:
        st.markdown("### Ward overview")
        
        if patients:
            sorted_patients = sorted(
                patients.items(),
                key=lambda x: x[1]["latest_risk"],
                reverse=True
            )[:MAX_HISTORY]

            rows = []
            for hadm_id, info in sorted_patients:
                rows.append({
                    "Patient ID":    hadm_id,
                    "Risk score":    round(float(info["latest_risk"]), 3),
                    "Alert level":   info["alert_level"],
                    "Readings":      info["num_readings"],
                    "Last updated":  info["last_updated"][:19].replace("T", " "),
                })
            df = pd.DataFrame(rows)

            def highlight_level(val):
                color = LEVEL_COLORS.get(val, "#888")
                bg    = LEVEL_BG.get(val, "#f5f5f5")
                return f"background-color: {bg}; color: {color}; font-weight: 500; border-radius: 4px; padding: 2px 8px;"

            styled = df.style.applymap(highlight_level, subset=["Alert level"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.divider()

            # Patient detail
            st.markdown("### Patient detail")

            patient_ids = [str(k) for k in patients.keys()]
            selected    = st.selectbox("Select patient", patient_ids)

            if selected:
                detail = fetch_patient_history(int(selected))
                info   = patients.get(int(selected)) or patients.get(selected, {})
                level  = info.get("alert_level", "NORMAL")
                risk   = float(info.get("latest_risk", 0))

                color = LEVEL_COLORS.get(level, "#888")
                bg    = LEVEL_BG.get(level, "#f5f5f5")
                st.markdown(
                    f'<div style="background:{bg};border-left:4px solid {color};'
                    f'padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:1rem">'
                    f'<span style="color:{color};font-weight:500;font-size:16px">'
                    f'{level}</span> — Risk score: <strong>{risk:.3f}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                history = detail.get("history", [])
                if len(history) >= 2:
                    times  = [h["timestamp"][:19].replace("T", " ") for h in history]
                    scores = [h["ensemble_risk"] for h in history]
                    levels = [h["alert_level"] for h in history]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=times, y=scores,
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(
                            size=8,
                            color=[LEVEL_COLORS.get(l, "#888") for l in levels],
                            line=dict(width=1, color="white"),
                        ),
                        name="Risk score",
                        hovertemplate="<b>%{x}</b><br>Risk: %{y:.3f}<extra></extra>",
                    ))
                    fig.add_hline(y=0.8, line_dash="dash", line_color="#E24B4A",
                                  annotation_text="Critical threshold (0.8)")
                    fig.add_hline(y=0.6, line_dash="dot",  line_color="#EF9F27",
                                  annotation_text="Warning threshold (0.6)")
                    fig.update_layout(
                        height=280,
                        margin=dict(l=0, r=0, t=20, b=0),
                        xaxis_title="Time",
                        yaxis_title="Ensemble risk",
                        yaxis=dict(range=[0, 1]),
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Only 1 reading so far — trend chart appears after 2+ readings.")

                if history:
                    hist_df = pd.DataFrame(history)
                    hist_df["timestamp"] = hist_df["timestamp"].str[:19].str.replace("T", " ")
                    hist_df["ensemble_risk"] = hist_df["ensemble_risk"].round(3)
                    hist_df.columns = ["Timestamp", "Risk score", "Alert level"]
                    st.dataframe(hist_df.iloc[::-1], use_container_width=True, hide_index=True)


# Auto-refresh
if auto_refresh and health.get("status") == "ok":
    time.sleep(REFRESH_SECS)
    st.cache_data.clear()
    st.rerun()