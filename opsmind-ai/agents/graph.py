"""
OpsMind AI — LangGraph Multi-Agent Orchestration
Agents:
  1. Supervisor       — Routes tasks to specialist agents
  2. MetricsAgent     — Pulls and summarizes system metrics
  3. AnomalyAgent     — Runs anomaly detection, raises alerts
  4. PredictAgent     — Predicts failures, recommends actions
  5. RAGAgent         — Retrieves knowledge from vector store (ChromaDB)
  6. ReportAgent      — Synthesizes final actionable report
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────
# 1. SHARED STATE DEFINITION
# ─────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task: str                          # Original user request
    current_agent: str                 # Which agent is active
    metrics: Optional[Dict[str, Any]]  # Live metrics payload
    anomaly_result: Optional[Dict]     # From AnomalyAgent
    prediction_result: Optional[Dict]  # From PredictAgent
    rag_context: Optional[str]         # Retrieved knowledge chunks
    report: Optional[str]              # Final synthesized report
    iteration_count: int               # Guard against infinite loops
    error: Optional[str]               # Error propagation
    user_role: str                     # RBAC role of requesting user


# ─────────────────────────────────────────
# 2. LLM FACTORY (Ollama)
# ─────────────────────────────────────────

def get_llm(
    model: str = "llama3.2",
    temperature: float = 0.1,
    base_url: str = "http://localhost:11434",
) -> ChatOllama:
    """Returns a ChatOllama instance. Swap model for richer capabilities."""
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
    )


# ─────────────────────────────────────────
# 3. MOCK METRICS (replace with real REST API / Prometheus in prod)
# ─────────────────────────────────────────

import numpy as np

def fetch_live_metrics(inject_anomaly: bool = False) -> Dict[str, Any]:
    """Simulates fetching real-time metrics from a monitoring REST API."""
    if inject_anomaly:
        return {
            "cpu_usage": 94.2,
            "memory_usage": 97.1,
            "disk_io": 88.5,
            "network_latency": 1240.0,
            "error_rate": 38.7,
            "request_count": 12450.0,
            "response_time": 3800.0,
            "service": "api-gateway",
            "timestamp": datetime.utcnow().isoformat(),
        }
    return {
        "cpu_usage": float(np.random.normal(45, 12)),
        "memory_usage": float(np.random.normal(60, 10)),
        "disk_io": float(np.random.normal(30, 8)),
        "network_latency": float(np.random.normal(22, 6)),
        "error_rate": float(np.random.beta(0.5, 10) * 5),
        "request_count": float(np.random.poisson(500)),
        "response_time": float(np.random.normal(155, 30)),
        "service": "api-gateway",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────
# 4. AGENT NODE IMPLEMENTATIONS
# ─────────────────────────────────────────

def supervisor_node(state: AgentState) -> AgentState:
    """Routes the task to the most appropriate specialist agent."""
    llm = get_llm(temperature=0.0)

    system = SystemMessage(content="""You are the Supervisor Agent of OpsMind AI.
Your job: analyze the user request and decide which specialist agent to call next.

Available agents:
- metrics_agent   : Fetch and summarize current system metrics
- anomaly_agent   : Detect anomalies in the metrics
- predict_agent   : Predict failures and generate recommendations
- rag_agent       : Retrieve relevant knowledge/runbooks from vector store
- report_agent    : Synthesize all findings into a final report
- END             : All tasks complete

RULES:
1. Always start with metrics_agent for operational queries.
2. After metrics_agent, call anomaly_agent.
3. After anomaly_agent, call predict_agent.
4. Call rag_agent when context/runbook knowledge is needed.
5. Call report_agent last to synthesize everything.
6. Output ONLY a JSON object: {"next_agent": "<agent_name>"}
""")

    context = f"""
Task: {state['task']}
Iteration: {state['iteration_count']}
Metrics fetched: {state.get('metrics') is not None}
Anomaly checked: {state.get('anomaly_result') is not None}
Prediction done: {state.get('prediction_result') is not None}
RAG context: {state.get('rag_context') is not None}
Report ready: {state.get('report') is not None}
"""
    response = llm.invoke([system, HumanMessage(content=context)])
    text = response.content.strip()

    # Parse JSON safely
    match = re.search(r'\{[^}]+\}', text)
    next_agent = "report_agent"  # fallback
    if match:
        try:
            parsed = json.loads(match.group())
            next_agent = parsed.get("next_agent", "report_agent")
        except json.JSONDecodeError:
            pass

    # Safety: prevent infinite loops
    if state["iteration_count"] >= 8:
        next_agent = "report_agent"

    return {
        **state,
        "current_agent": next_agent,
        "iteration_count": state["iteration_count"] + 1,
        "messages": state["messages"] + [
            AIMessage(content=f"[Supervisor → {next_agent}]", name="supervisor")
        ],
    }


def metrics_agent_node(state: AgentState) -> AgentState:
    """Fetches live metrics and provides a natural-language summary."""
    llm = get_llm(temperature=0.1)

    # Detect if anomaly injection is requested
    inject = "critical" in state["task"].lower() or "anomaly" in state["task"].lower()
    metrics = fetch_live_metrics(inject_anomaly=inject)

    system = SystemMessage(content="""You are the Metrics Agent.
Summarize the system metrics concisely. Highlight anything that looks unusual.
Use bullet points. Be factual and brief.""")

    summary_prompt = HumanMessage(content=f"Metrics snapshot:\n{json.dumps(metrics, indent=2)}")
    response = llm.invoke([system, summary_prompt])

    return {
        **state,
        "metrics": metrics,
        "messages": state["messages"] + [
            AIMessage(content=response.content, name="metrics_agent")
        ],
    }


def anomaly_agent_node(state: AgentState) -> AgentState:
    """Runs Isolation Forest anomaly detection and explains the result."""
    llm = get_llm(temperature=0.0)

    metrics = state.get("metrics", fetch_live_metrics())

    # Try to use trained model; fall back to rule-based if not available
    try:
        from ml.models import AnomalyDetector
        ad = AnomalyDetector.load()
        anomaly_result = ad.predict(metrics)
    except Exception:
        # Rule-based fallback
        anomaly_score = 0.0
        if metrics.get("cpu_usage", 0) > 85:     anomaly_score += 0.3
        if metrics.get("memory_usage", 0) > 90:  anomaly_score += 0.3
        if metrics.get("error_rate", 0) > 15:    anomaly_score += 0.25
        if metrics.get("network_latency", 0) > 500: anomaly_score += 0.15
        anomaly_result = {
            "is_anomaly": anomaly_score > 0.4,
            "anomaly_score": round(min(anomaly_score, 1.0), 4),
            "severity": "high" if anomaly_score > 0.6 else ("medium" if anomaly_score > 0.3 else "normal"),
            "detected_at": datetime.utcnow().isoformat(),
        }

    system = SystemMessage(content="""You are the Anomaly Detection Agent.
Explain the anomaly detection results clearly. 
If anomaly detected: explain what's wrong, severity, and immediate actions.
If normal: confirm system health.""")

    prompt = HumanMessage(content=f"""
Metrics: {json.dumps(metrics, indent=2)}
Anomaly Result: {json.dumps(anomaly_result, indent=2)}
""")
    response = llm.invoke([system, prompt])

    return {
        **state,
        "anomaly_result": anomaly_result,
        "messages": state["messages"] + [
            AIMessage(content=response.content, name="anomaly_agent")
        ],
    }


def predict_agent_node(state: AgentState) -> AgentState:
    """Predicts failure probability and provides remediation recommendations."""
    llm = get_llm(temperature=0.1)

    metrics = state.get("metrics", fetch_live_metrics())

    try:
        from ml.models import FailurePredictor
        fp = FailurePredictor.load()
        prediction = fp.predict_proba(metrics)
    except Exception:
        # Rule-based fallback
        score = 0.0
        if metrics.get("cpu_usage", 0) > 85:     score += 0.35
        if metrics.get("memory_usage", 0) > 90:  score += 0.30
        if metrics.get("error_rate", 0) > 15:    score += 0.25
        if metrics.get("network_latency", 0) > 500: score += 0.10
        prob = min(score, 0.99)
        prediction = {
            "failure_probability": round(prob, 4),
            "risk_level": "critical" if prob > 0.75 else ("high" if prob > 0.5 else "low"),
            "recommendation": "Monitor CPU, memory, and error rates closely.",
            "predicted_at": datetime.utcnow().isoformat(),
        }

    system = SystemMessage(content="""You are the Predictive Analytics Agent.
Based on failure prediction results:
1. State the risk level clearly
2. Explain WHY (which metrics are concerning)
3. Give a prioritized action plan (numbered list)
4. Estimate time-to-failure if critical""")

    prompt = HumanMessage(content=f"""
Current Metrics: {json.dumps(metrics, indent=2)}
Prediction: {json.dumps(prediction, indent=2)}
""")
    response = llm.invoke([system, prompt])

    return {
        **state,
        "prediction_result": prediction,
        "messages": state["messages"] + [
            AIMessage(content=response.content, name="predict_agent")
        ],
    }


def rag_agent_node(state: AgentState) -> AgentState:
    """Retrieves relevant runbooks/knowledge from ChromaDB vector store."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path="./data/chroma_db")
        collection = client.get_or_create_collection("opsmind_runbooks")

        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = encoder.encode([state["task"]]).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )
        docs = results.get("documents", [[]])[0]
        rag_context = "\n\n".join(docs) if docs else "No relevant runbooks found."
    except Exception as e:
        rag_context = (
            "Runbook: HIGH CPU — Scale horizontally. Kill non-essential processes. "
            "Check for memory leaks. Consider caching strategy.\n"
            "Runbook: HIGH MEMORY — Trigger GC, analyze heap dumps. "
            "Check for connection pool exhaustion.\n"
            "Runbook: HIGH ERROR RATE — Inspect upstream dependencies. "
            "Enable circuit breaker. Roll back recent deployment if correlated."
        )

    llm = get_llm(temperature=0.1)
    system = SystemMessage(content="You are the Knowledge Retrieval Agent. Summarize relevant runbooks and knowledge for this situation.")
    prompt = HumanMessage(content=f"Task: {state['task']}\nRetrieved Context:\n{rag_context}")
    response = llm.invoke([system, prompt])

    return {
        **state,
        "rag_context": rag_context,
        "messages": state["messages"] + [
            AIMessage(content=response.content, name="rag_agent")
        ],
    }


def report_agent_node(state: AgentState) -> AgentState:
    """Synthesizes all agent findings into a final structured report."""
    llm = get_llm(temperature=0.2)

    system = SystemMessage(content="""You are the Report Agent.
Synthesize all agent findings into a professional incident/analysis report.

Structure:
## 🔍 Executive Summary
## 📊 Metrics Overview
## 🚨 Anomaly Analysis
## 🔮 Failure Prediction
## 📋 Recommended Actions (prioritized)
## 🔗 References & Runbooks

Be concise, technical, and actionable. Use markdown formatting.""")

    context = f"""
TASK: {state['task']}
METRICS: {json.dumps(state.get('metrics', {}), indent=2)}
ANOMALY: {json.dumps(state.get('anomaly_result', {}), indent=2)}
PREDICTION: {json.dumps(state.get('prediction_result', {}), indent=2)}
RUNBOOKS: {state.get('rag_context', 'N/A')}
USER ROLE: {state.get('user_role', 'viewer')}
"""
    response = llm.invoke([system, HumanMessage(content=context)])
    report = response.content

    return {
        **state,
        "report": report,
        "messages": state["messages"] + [
            AIMessage(content=report, name="report_agent")
        ],
    }


# ─────────────────────────────────────────
# 5. ROUTING LOGIC
# ─────────────────────────────────────────

def route_from_supervisor(state: AgentState) -> Literal[
    "metrics_agent", "anomaly_agent", "predict_agent",
    "rag_agent", "report_agent", "__end__"
]:
    agent = state.get("current_agent", "report_agent")
    if state.get("report") is not None:
        return "__end__"
    mapping = {
        "metrics_agent": "metrics_agent",
        "anomaly_agent": "anomaly_agent",
        "predict_agent": "predict_agent",
        "rag_agent":     "rag_agent",
        "report_agent":  "report_agent",
        "END":           "__end__",
    }
    return mapping.get(agent, "report_agent")


# ─────────────────────────────────────────
# 6. GRAPH BUILDER
# ─────────────────────────────────────────

def build_opsmind_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor",    supervisor_node)
    graph.add_node("metrics_agent", metrics_agent_node)
    graph.add_node("anomaly_agent", anomaly_agent_node)
    graph.add_node("predict_agent", predict_agent_node)
    graph.add_node("rag_agent",     rag_agent_node)
    graph.add_node("report_agent",  report_agent_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor routes conditionally
    graph.add_conditional_edges("supervisor", route_from_supervisor, {
        "metrics_agent": "metrics_agent",
        "anomaly_agent": "anomaly_agent",
        "predict_agent": "predict_agent",
        "rag_agent":     "rag_agent",
        "report_agent":  "report_agent",
        "__end__":       END,
    })

    # All specialist agents return to supervisor for next routing decision
    for node in ["metrics_agent", "anomaly_agent", "predict_agent", "rag_agent"]:
        graph.add_edge(node, "supervisor")

    # Report agent always ends
    graph.add_edge("report_agent", END)

    return graph.compile()


# ─────────────────────────────────────────
# 7. PUBLIC RUNNER
# ─────────────────────────────────────────

def run_agent(task: str, user_role: str = "analyst") -> Dict[str, Any]:
    """Execute the full agent graph for a given task."""
    graph = build_opsmind_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "current_agent": "metrics_agent",
        "metrics": None,
        "anomaly_result": None,
        "prediction_result": None,
        "rag_context": None,
        "report": None,
        "iteration_count": 0,
        "error": None,
        "user_role": user_role,
    }

    final_state = graph.invoke(initial_state)
    return {
        "report": final_state.get("report", "No report generated."),
        "anomaly": final_state.get("anomaly_result"),
        "prediction": final_state.get("prediction_result"),
        "metrics": final_state.get("metrics"),
        "iterations": final_state.get("iteration_count"),
    }


if __name__ == "__main__":
    result = run_agent(
        task="Analyze current system health and detect any anomalies",
        user_role="admin"
    )
    print(result["report"])
