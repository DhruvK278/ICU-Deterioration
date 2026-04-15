"""
dashboard/app.py

Real-time ICU risk monitoring UI built with Streamlit.

Shows:
  - Ward overview: all active patients with colour-coded risk
  - Live risk score per patient (auto-refreshes every 10s)
  - Alert history and trend chart
  - Manual patient lookup

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
REFRESH_SECS  = 10
MAX_HISTORY   = 20 

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
        r = requests.get(f"{FOG_URL}/patients", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=REFRESH_SECS)
def fetch_patient_history(hadm_id: int) -> dict:
    try:
        r = requests.get(f"{FOG_URL}/patients/{hadm_id}", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=5)
def fetch_health() -> dict:
    try:
        r = requests.get(f"{FOG_URL}/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def send_test_patient(hadm_id: int, risk_level: str):
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
        "triggers":  ["dx_sepsis=1"] if risk_level == "high" else [],
        "reading":   profiles[risk_level],
        "forwarded": False,
    }
    try:
        r = requests.post(f"{FOG_URL}/predict", json=payload, timeout=3)
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
    if st.button("Send to fog ↗"):
        result = send_test_patient(int(demo_id), demo_risk)
        if result:
            st.success(f"Scored: {result['ensemble_risk']:.3f} → {result['alert_level']}")
            st.cache_data.clear()
        else:
            st.error("Failed — is fog running?")

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        st.markdown(f"Refreshing every {REFRESH_SECS}s")


# Main content
st.markdown("## ICU deterioration monitor")
st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")

patients = fetch_patients()

if not patients:
    st.info("No active patients. Use the sidebar to add demo patients, or start the edge simulator.")
    st.code("python src/edge/edge_detector.py --fog-url http://localhost:8000")
else:
    total    = len(patients)
    critical = sum(1 for p in patients.values() if p["alert_level"] == "CRITICAL")
    warning  = sum(1 for p in patients.values() if p["alert_level"] == "WARNING")
    watch    = sum(1 for p in patients.values() if p["alert_level"] == "WATCH")
    normal   = sum(1 for p in patients.values() if p["alert_level"] == "NORMAL")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total patients", total)
    c2.metric("Critical", critical,  delta=None)
    c3.metric("Warning",  warning,   delta=None)
    c4.metric("Watch",    watch,      delta=None)
    c5.metric("Normal",   normal,     delta=None)

    st.divider()

    st.markdown("### Ward overview")

    sorted_patients = sorted(
        patients.items(),
        key=lambda x: x[1]["latest_risk"],
        reverse=True
    )[:MAX_HISTORY]

    # Build dataframe for table
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

    # Colour-code the alert level column
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