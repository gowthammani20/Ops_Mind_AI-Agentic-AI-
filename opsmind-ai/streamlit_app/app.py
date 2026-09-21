"""
OpsMind AI — Streamlit Dashboard
Tabs: Login | Dashboard | Anomaly Detection | Predictions | Agent | Chatbot | Admin
"""

import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="OpsMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────

for key, default in {
    "token": None, "username": None, "role": None,
    "chat_history": [], "last_metrics": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def api(method: str, endpoint: str, **kwargs) -> dict:
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        resp = getattr(requests, method)(
            f"{API_BASE}{endpoint}", headers=headers,
            timeout=30, **kwargs
        )
        return resp.json() if resp.ok else {"error": resp.json().get("detail", resp.text)}
    except requests.ConnectionError:
        return {"error": "❌ Cannot connect to API. Make sure the FastAPI server is running."}
    except Exception as e:
        return {"error": str(e)}

def login_ui():
    st.markdown("## 🔐 Login to OpsMind AI")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        | Username | Password | Role |
        |----------|----------|------|
        | gowtham | gowtham2026 | Super Admin |
        | admin | admin123 | Admin |
        | analyst | analyst123 | Analyst |
        | viewer | viewer123 | Viewer |
        """)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("🔑 Login", use_container_width=True, type="primary"):
            data = api("post", "/auth/token",
                       data={"username": username, "password": password, "grant_type": "password"},
                       headers={"Content-Type": "application/x-www-form-urlencoded"})
            if "error" in data:
                st.error(data["error"])
            else:
                st.session_state.token    = data["access_token"]
                st.session_state.username = username
                st.session_state.role     = data.get("role", "viewer")
                st.success(f"✅ Welcome, {username}! Role: {data.get('role')}")
                time.sleep(0.5)
                st.rerun()

def gauge_chart(value: float, title: str, max_val: float = 100, color: str = "#00b4d8"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, max_val * 0.6],   "color": "#1e3a5f"},
                {"range": [max_val * 0.6, max_val * 0.8], "color": "#2d6a9f"},
                {"range": [max_val * 0.8, max_val], "color": "#c1121f"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "value": max_val * 0.85},
        },
        number={"suffix": "%" if max_val == 100 else "ms", "font": {"size": 20}},
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig

def severity_badge(severity: str) -> str:
    colors = {"critical": "🔴", "high": "🟠", "medium": "🟡", "normal": "🟢"}
    return colors.get(severity, "⚪") + " " + severity.upper()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("# 🤖 OpsMind AI")
        st.markdown("*Enterprise Agentic Platform*")
        st.divider()
        if st.session_state.token:
            st.success(f"👤 **{st.session_state.username}**")
            st.info(f"🎭 Role: `{st.session_state.role}`")
            if st.button("🚪 Logout"):
                for k in ["token", "username", "role", "chat_history"]:
                    st.session_state[k] = None if k != "chat_history" else []
                st.rerun()
        st.divider()
        st.markdown("**🔧 Tech Stack**")
        st.markdown("""
- 🧠 LangGraph + LangChain
- 🦙 Ollama (llama3.2)
- ⚡ FastAPI
- 📊 Scikit-learn + XGBoost
- 🔮 TensorFlow (LSTM)
- 🗄️ ChromaDB (RAG)
- 🔐 OAuth2 + JWT + RBAC
- 🎨 Streamlit
        """)
        st.divider()
        st.caption(f"UTC: {datetime.utcnow().strftime('%H:%M:%S')}")


# ─────────────────────────────────────────
# TAB: DASHBOARD
# ─────────────────────────────────────────

def tab_dashboard():
    st.header("📊 Live Operations Dashboard")
    inject = st.toggle("💥 Inject Anomaly (Demo Mode)", value=False)

    if st.button("🔄 Refresh Metrics", type="primary"):
        metrics = api("get", f"/metrics/live?inject_anomaly={str(inject).lower()}")
        if "error" not in metrics:
            st.session_state.last_metrics = metrics

    metrics = st.session_state.last_metrics
    if not metrics:
        metrics = api("get", f"/metrics/live?inject_anomaly={str(inject).lower()}")
        st.session_state.last_metrics = metrics

    if "error" in metrics:
        st.error(metrics["error"])
        return

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🖥 CPU Usage",      f"{metrics['cpu_usage']:.1f}%",
                delta=f"{metrics['cpu_usage'] - 45:.1f}%",
                delta_color="inverse")
    col2.metric("💾 Memory Usage",   f"{metrics['memory_usage']:.1f}%",
                delta=f"{metrics['memory_usage'] - 60:.1f}%",
                delta_color="inverse")
    col3.metric("🌐 Network Latency",f"{metrics['network_latency']:.0f}ms",
                delta=f"{metrics['network_latency'] - 20:.0f}ms",
                delta_color="inverse")
    col4.metric("⚠️ Error Rate",     f"{metrics['error_rate']:.2f}%",
                delta=f"{metrics['error_rate'] - 2:.2f}%",
                delta_color="inverse")

    st.divider()

    # Gauge row
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(gauge_chart(metrics["cpu_usage"],    "CPU %"),      use_container_width=True)
    g2.plotly_chart(gauge_chart(metrics["memory_usage"], "Memory %"),   use_container_width=True)
    g3.plotly_chart(gauge_chart(min(metrics["network_latency"], 2000),
                                "Latency ms", max_val=2000, color="#f77f00"),
                    use_container_width=True)

    # Historical chart
    st.subheader("📈 Historical Trend (Last 100 Points)")
    hist = api("get", "/metrics/history?n_points=100")
    if "error" not in hist and isinstance(hist, list):
        df = pd.DataFrame(hist)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fig = px.line(df, x="timestamp",
                          y=["cpu_usage", "memory_usage", "error_rate"],
                          title="System Metrics Over Time",
                          color_discrete_map={
                              "cpu_usage": "#00b4d8",
                              "memory_usage": "#90e0ef",
                              "error_rate": "#c1121f",
                          })
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white")
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# TAB: ANOMALY DETECTION
# ─────────────────────────────────────────

def tab_anomaly():
    st.header("🚨 Anomaly Detection")
    st.markdown("Powered by **Isolation Forest** (ML) + **LSTM Autoencoder** (Deep Learning)")

    with st.expander("📥 Enter Custom Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            cpu   = st.slider("CPU Usage %",      0, 100, 45)
            mem   = st.slider("Memory Usage %",   0, 100, 60)
            disk  = st.slider("Disk I/O %",       0, 100, 30)
            lat   = st.slider("Network Latency (ms)", 0, 2000, 20)
        with col2:
            err   = st.slider("Error Rate %",     0.0, 100.0, 2.0)
            req   = st.number_input("Request Count", 0, 100000, 500)
            resp  = st.slider("Response Time (ms)", 0, 10000, 150)

    if st.button("🔍 Detect Anomaly", type="primary"):
        payload = {
            "cpu_usage": cpu, "memory_usage": mem, "disk_io": disk,
            "network_latency": lat, "error_rate": err,
            "request_count": req, "response_time": resp,
        }
        result = api("post", "/anomaly/detect", json=payload)

        if "error" in result:
            st.error(result["error"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Anomaly Detected", "⚠️ YES" if result["is_anomaly"] else "✅ NO")
            col2.metric("Anomaly Score",    f"{result['anomaly_score']:.4f}")
            col3.metric("Severity",         severity_badge(result.get("severity", "normal")))

            if result["is_anomaly"]:
                st.error("🔴 **ANOMALY DETECTED** — Immediate investigation recommended!")
                if result.get("severity") == "critical":
                    st.warning("🚨 CRITICAL severity — trigger incident response protocol NOW.")
            else:
                st.success("✅ System operating within normal parameters.")

            # Score visualization
            score = result["anomaly_score"]
            fig = go.Figure(go.Bar(
                x=["Normal", "Anomaly Score"],
                y=[1.0 - score, score],
                marker_color=["#2ecc71", "#e74c3c"],
            ))
            fig.update_layout(title="Anomaly Score Breakdown",
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white", height=250)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# TAB: PREDICTIVE ANALYTICS
# ─────────────────────────────────────────

def tab_predictions():
    st.header("🔮 Predictive Failure Analytics")
    st.markdown("Powered by **XGBoost** classifier trained on historical ops data")

    live_metrics = api("get", "/metrics/live")
    if "error" not in live_metrics:
        st.info("📡 Using current live metrics. Edit below to simulate scenarios.")
        cpu   = live_metrics.get("cpu_usage", 45)
        mem   = live_metrics.get("memory_usage", 60)
        err   = live_metrics.get("error_rate", 2.0)
        lat   = live_metrics.get("network_latency", 20.0)
        resp  = live_metrics.get("response_time", 150.0)
    else:
        cpu, mem, err, lat, resp = 45, 60, 2.0, 20.0, 150.0

    with st.expander("⚙️ Tune Scenario"):
        c1, c2 = st.columns(2)
        with c1:
            cpu  = st.slider("CPU %",      0, 100, int(cpu),   key="pred_cpu")
            mem  = st.slider("Memory %",   0, 100, int(mem),   key="pred_mem")
            err  = st.slider("Error Rate", 0.0, 100.0, float(err), key="pred_err")
        with c2:
            lat  = st.slider("Latency ms", 0, 2000, int(lat),  key="pred_lat")
            resp = st.slider("Response ms", 0, 10000, int(resp), key="pred_resp")

    if st.button("🔮 Predict Failure Probability", type="primary"):
        payload = {
            "cpu_usage": cpu, "memory_usage": mem, "disk_io": 30,
            "network_latency": lat, "error_rate": err,
            "request_count": 500, "response_time": resp,
        }
        result = api("post", "/predict/failure", json=payload)

        if "error" in result:
            st.error(result["error"])
        else:
            prob = result["failure_probability"]
            risk = result["risk_level"]

            col1, col2 = st.columns(2)
            col1.metric("Failure Probability", f"{prob * 100:.1f}%")
            col2.metric("Risk Level", severity_badge(risk))

            # Probability gauge
            color = "#e74c3c" if prob > 0.75 else ("#f39c12" if prob > 0.5 else "#2ecc71")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={"text": "Failure Probability (%)"},
                delta={"reference": 25},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": color},
                       "steps": [
                           {"range": [0, 25],  "color": "#1e3a5f"},
                           {"range": [25, 50], "color": "#2d6a4f"},
                           {"range": [50, 75], "color": "#7d4f00"},
                           {"range": [75, 100],"color": "#4a0404"},
                       ]},
                number={"suffix": "%"},
            ))
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Recommended Actions")
            st.info(result.get("recommendation", "No specific recommendation."))

    # Feature Importance
    st.subheader("📊 Feature Importance (XGBoost)")
    fi = api("get", "/predict/feature-importance")
    if "feature_importance" in fi:
        fi_df = pd.DataFrame(list(fi["feature_importance"].items()),
                             columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="blues",
                     title="XGBoost Feature Importance")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# TAB: AGENT
# ─────────────────────────────────────────

def tab_agent():
    st.header("🧠 LangGraph Multi-Agent System")
    st.markdown("""
    **Agent Pipeline:**
    `Supervisor → Metrics Agent → Anomaly Agent → Predict Agent → RAG Agent → Report Agent`
    """)

    predefined = [
        "Analyze current system health and detect any anomalies",
        "Investigate critical system performance degradation",
        "Generate a full incident report for the ops team",
        "What is the failure risk and recommended remediation?",
    ]
    task = st.selectbox("📝 Choose a task or type your own:", predefined + ["Custom..."])
    if task == "Custom...":
        task = st.text_area("Enter your custom task:")

    col1, col2 = st.columns(2)
    with col1:
        inject = st.checkbox("💥 Inject anomaly for demo", value=False)
    with col2:
        st.info("⏱️ Agent typically takes 30–120s depending on Ollama model speed")

    if st.button("🚀 Run Agent Pipeline", type="primary"):
        with st.spinner("🤖 Multi-agent pipeline running..."):
            result = api("post", "/agent/run",
                         json={"task": task, "inject_anomaly": inject})

        if "error" in result:
            st.error(result["error"])
        else:
            # Agent status
            col1, col2, col3 = st.columns(3)
            col1.metric("Status", "✅ Completed")
            col2.metric("Iterations", result.get("iterations", "?"))
            col3.metric("Role", st.session_state.role or "analyst")

            st.subheader("📋 Agent Report")
            st.markdown(result.get("report", "No report generated."))

            with st.expander("🔍 Raw Results"):
                st.json({
                    "anomaly": result.get("anomaly"),
                    "prediction": result.get("prediction"),
                    "metrics": result.get("metrics"),
                })


# ─────────────────────────────────────────
# TAB: CHATBOT
# ─────────────────────────────────────────

def tab_chatbot():
    st.header("💬 OpsMind AI Chatbot")
    st.markdown("*Conversational assistant powered by Ollama (llama3.2)*")

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about system health, incidents, recommendations..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                result = api("post", "/agent/chat", json={
                    "message": prompt,
                    "conversation_history": st.session_state.chat_history[:-1],
                })
            if "error" in result:
                response = f"❌ Error: {result['error']}"
            else:
                response = result.get("response", "I couldn't generate a response.")
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────
# TAB: ADMIN
# ─────────────────────────────────────────

def tab_admin():
    st.header("⚙️ Admin Panel")
    if st.session_state.role not in ("admin", "super_admin"):
        st.error("🔒 Access denied. Admin or Super Admin role required.")
        return

    tab1, tab2, tab3 = st.tabs(["👥 Users", "📜 Audit Logs", "🔑 RBAC Reference"])

    with tab1:
        users = api("get", "/auth/users")
        if isinstance(users, list):
            st.dataframe(pd.DataFrame(users), use_container_width=True)
        else:
            st.error(users.get("error", "Could not fetch users"))

        st.subheader("➕ Create New User")
        with st.form("create_user"):
            nu = st.text_input("Username")
            ne = st.text_input("Email")
            nf = st.text_input("Full Name")
            np_ = st.text_input("Password", type="password")
            nr = st.selectbox("Role", ["viewer", "analyst", "admin"])
            if st.form_submit_button("Create User"):
                result = api("post", "/auth/register", json={
                    "username": nu, "email": ne, "full_name": nf,
                    "password": np_, "role": nr,
                })
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(result.get("message"))

    with tab2:
        logs = api("get", "/system/audit-logs?limit=100")
        if "logs" in logs:
            st.dataframe(pd.DataFrame(logs["logs"]), use_container_width=True)
            st.caption(f"Total requests logged: {logs.get('total', 0)}")
        else:
            st.error(logs.get("error", "Could not fetch logs"))

    with tab3:
        st.markdown("""
| Role | Permissions |
|------|-------------|
| **super_admin** | All permissions (`*`) |
| **admin** | Read/write metrics, anomalies, predictions; manage users & agents; audit logs |
| **analyst** | Read metrics/anomalies/predictions; write predictions; audit logs |
| **viewer** | Read-only: metrics, anomalies, predictions |
| **agent** | M2M: read/write metrics & anomalies; read predictions |
        """)


# ─────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────

def main():
    render_sidebar()

    if not st.session_state.token:
        login_ui()
        return

    st.markdown(f"## 🤖 OpsMind AI — Enterprise Agentic Operations Platform")
    st.markdown(f"*Logged in as **{st.session_state.username}** | Role: `{st.session_state.role}`*")

    tabs = st.tabs([
        "📊 Dashboard",
        "🚨 Anomaly Detection",
        "🔮 Predictions",
        "🧠 Agent",
        "💬 Chatbot",
        "⚙️ Admin",
    ])

    with tabs[0]: tab_dashboard()
    with tabs[1]: tab_anomaly()
    with tabs[2]: tab_predictions()
    with tabs[3]: tab_agent()
    with tabs[4]: tab_chatbot()
    with tabs[5]: tab_admin()


if __name__ == "__main__":
    main()
