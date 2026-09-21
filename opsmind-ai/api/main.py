"""
OpsMind AI — FastAPI REST API Backend
Endpoints: Auth, Metrics, Anomaly, Prediction, Agent, Health
All protected by OAuth2 + JWT + RBAC
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

# Local imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.auth_core import (
    Role, UserCreate, UserResponse, UserInDB, USERS_DB,
    authenticate_user, issue_tokens, get_current_user,
    require_permission, require_role, get_password_hash,
)
from ml.models import AnomalyDetector, FailurePredictor, generate_synthetic_metrics
from agents.graph import run_agent, fetch_live_metrics

# ─────────────────────────────────────────
# 1. APP SETUP
# ─────────────────────────────────────────

app = FastAPI(
    title="OpsMind AI — Enterprise Agentic Platform",
    description="""
## OpsMind AI REST API

Agentic AI platform with:
- **LangGraph multi-agent orchestration**
- **ML/DL anomaly detection** (Isolation Forest + LSTM Autoencoder)
- **XGBoost predictive failure analytics**
- **OAuth2 + JWT + RBAC authorization**
- **RAG-powered knowledge retrieval**

Built with LangChain · LangGraph · Ollama · FastAPI · Scikit-learn · XGBoost · TensorFlow
    """,
    version="1.0.0",
    contact={"name": "Gowtham", "email": "gowtham@opsmind.ai"},
    license_info={"name": "MIT"},
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# In-memory request log (replace with DB in production)
REQUEST_LOG: list = []

# ─────────────────────────────────────────
# 2. REQUEST LOGGING MIDDLEWARE
# ─────────────────────────────────────────

@app.middleware("http")
async def audit_log_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    REQUEST_LOG.append({
        "path": request.url.path,
        "method": request.method,
        "status": response.status_code,
        "duration_ms": duration,
        "timestamp": datetime.utcnow().isoformat(),
    })
    if len(REQUEST_LOG) > 1000:
        REQUEST_LOG.pop(0)
    return response


# ─────────────────────────────────────────
# 3. PYDANTIC REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────

class MetricsSnapshot(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_latency: float
    error_rate: float
    request_count: float
    response_time: float
    service: str = "api-gateway"

class AgentRequest(BaseModel):
    task: str
    inject_anomaly: bool = False

class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []


# ─────────────────────────────────────────
# 4. AUTH ENDPOINTS
# ─────────────────────────────────────────

@app.post("/auth/token", tags=["Authentication"],
          summary="Login and get JWT access + refresh tokens")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return issue_tokens(user)


@app.post("/auth/register", tags=["Authentication"],
          summary="Register a new user (admin only)")
async def register(
    user_data: UserCreate,
    current_user: UserInDB = Depends(require_role([Role.SUPER_ADMIN, Role.ADMIN])),
):
    if user_data.username in USERS_DB:
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = UserInDB(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
        hashed_password=get_password_hash(user_data.password),
    )
    USERS_DB[user_data.username] = new_user
    return {"message": f"User '{user_data.username}' created with role '{user_data.role}'"}


@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"],
         summary="Get current user profile")
async def get_me(current_user: UserInDB = Depends(get_current_user)):
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
    )


@app.get("/auth/users", tags=["Authentication"],
         summary="List all users (admin only)")
async def list_users(
    current_user: UserInDB = Depends(require_role([Role.SUPER_ADMIN, Role.ADMIN]))
):
    return [
        {"username": u.username, "role": u.role, "email": u.email}
        for u in USERS_DB.values()
    ]


# ─────────────────────────────────────────
# 5. METRICS ENDPOINTS
# ─────────────────────────────────────────

@app.get("/metrics/live", tags=["Metrics"],
         summary="Get live system metrics snapshot")
async def get_live_metrics(
    inject_anomaly: bool = False,
    current_user: UserInDB = Depends(require_permission("read:metrics")),
):
    return fetch_live_metrics(inject_anomaly=inject_anomaly)


@app.get("/metrics/history", tags=["Metrics"],
         summary="Get historical metrics (simulated)")
async def get_metrics_history(
    n_points: int = 100,
    current_user: UserInDB = Depends(require_permission("read:metrics")),
):
    df = generate_synthetic_metrics(n_samples=n_points)
    return df.drop(columns=["label"]).tail(n_points).to_dict(orient="records")


@app.post("/metrics/ingest", tags=["Metrics"],
          summary="Ingest a metrics snapshot (write)")
async def ingest_metrics(
    metrics: MetricsSnapshot,
    current_user: UserInDB = Depends(require_permission("write:metrics")),
):
    return {
        "status": "ingested",
        "service": metrics.service,
        "timestamp": datetime.utcnow().isoformat(),
        "received": metrics.model_dump(),
    }


# ─────────────────────────────────────────
# 6. ANOMALY DETECTION ENDPOINTS
# ─────────────────────────────────────────

@app.post("/anomaly/detect", tags=["Anomaly Detection"],
          summary="Run anomaly detection on metrics")
async def detect_anomaly(
    metrics: MetricsSnapshot,
    current_user: UserInDB = Depends(require_permission("read:anomalies")),
):
    try:
        ad = AnomalyDetector.load()
        result = ad.predict(metrics.model_dump())
    except Exception:
        # Fallback rule-based
        m = metrics.model_dump()
        score = 0.0
        if m["cpu_usage"] > 85:          score += 0.30
        if m["memory_usage"] > 90:       score += 0.30
        if m["error_rate"] > 15:         score += 0.25
        if m["network_latency"] > 500:   score += 0.15
        score = min(score, 1.0)
        result = {
            "is_anomaly": score > 0.4,
            "anomaly_score": round(score, 4),
            "severity": "high" if score > 0.6 else ("medium" if score > 0.3 else "normal"),
            "detected_at": datetime.utcnow().isoformat(),
            "note": "rule-based fallback (model not trained yet)",
        }
    return result


@app.get("/anomaly/status", tags=["Anomaly Detection"],
         summary="Check current system anomaly status")
async def get_anomaly_status(
    current_user: UserInDB = Depends(require_permission("read:anomalies")),
):
    metrics = fetch_live_metrics()
    try:
        ad = AnomalyDetector.load()
        result = ad.predict(metrics)
    except Exception:
        result = {"is_anomaly": False, "anomaly_score": 0.1, "severity": "normal"}
    return {"metrics": metrics, "anomaly": result}


# ─────────────────────────────────────────
# 7. PREDICTIVE ANALYTICS ENDPOINTS
# ─────────────────────────────────────────

@app.post("/predict/failure", tags=["Predictive Analytics"],
          summary="Predict system failure probability")
async def predict_failure(
    metrics: MetricsSnapshot,
    current_user: UserInDB = Depends(require_permission("read:predictions")),
):
    try:
        fp = FailurePredictor.load()
        result = fp.predict_proba(metrics.model_dump())
    except Exception:
        m = metrics.model_dump()
        prob = 0.0
        if m["cpu_usage"] > 85:        prob += 0.35
        if m["memory_usage"] > 90:     prob += 0.30
        if m["error_rate"] > 15:       prob += 0.25
        if m["network_latency"] > 500: prob += 0.10
        prob = min(prob, 0.99)
        result = {
            "failure_probability": round(prob, 4),
            "risk_level": "critical" if prob > 0.75 else ("high" if prob > 0.5 else "low"),
            "recommendation": "Monitor closely. Train models for better predictions.",
            "predicted_at": datetime.utcnow().isoformat(),
        }
    return result


@app.get("/predict/feature-importance", tags=["Predictive Analytics"],
         summary="Get XGBoost feature importance scores")
async def get_feature_importance(
    current_user: UserInDB = Depends(require_permission("read:predictions")),
):
    try:
        fp = FailurePredictor.load()
        return {"feature_importance": fp.get_feature_importance()}
    except Exception:
        return {
            "feature_importance": {
                "error_rate": 0.28,
                "cpu_usage": 0.22,
                "memory_usage": 0.20,
                "network_latency": 0.15,
                "response_time": 0.08,
                "disk_io": 0.05,
                "request_count": 0.02,
            },
            "note": "Estimated values (model not trained yet)"
        }


# ─────────────────────────────────────────
# 8. AGENTIC AI ENDPOINTS
# ─────────────────────────────────────────

@app.post("/agent/run", tags=["Agentic AI"],
          summary="Run the full LangGraph multi-agent pipeline")
async def run_agent_endpoint(
    request: AgentRequest,
    current_user: UserInDB = Depends(require_permission("read:metrics")),
):
    """Triggers the full LangGraph agent graph: Supervisor → Metrics → Anomaly → Predict → RAG → Report."""
    try:
        result = run_agent(task=request.task, user_role=current_user.role.value)
        return {
            "status": "completed",
            "task": request.task,
            "user": current_user.username,
            "role": current_user.role.value,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.post("/agent/chat", tags=["Agentic AI"],
          summary="Chat with the OpsMind AI assistant")
async def chat(
    req: ChatRequest,
    current_user: UserInDB = Depends(get_current_user),
):
    """Conversational chatbot backed by Ollama LLM with operational context."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage as LCAIMessage

        llm = ChatOllama(model="llama3.2", temperature=0.3)

        system = SystemMessage(content=f"""You are OpsMind AI — an enterprise IT operations assistant.
You help with system monitoring, anomaly detection, incident response, and capacity planning.
Current user: {current_user.full_name} | Role: {current_user.role.value}
Current time: {datetime.utcnow().isoformat()}
Be concise, technical, and helpful. Use markdown formatting.""")

        messages = [system]
        for msg in req.conversation_history[-10:]:  # keep last 10 turns
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(LCAIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=req.message))

        response = llm.invoke(messages)
        return {
            "response": response.content,
            "model": "llama3.2 via Ollama",
            "user": current_user.username,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ─────────────────────────────────────────
# 9. SYSTEM / HEALTH ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health", tags=["System"],
         summary="Health check (public)")
async def health():
    return {
        "status": "healthy",
        "app": "OpsMind AI",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/system/info", tags=["System"],
         summary="Detailed system info (authenticated)")
async def system_info(current_user: UserInDB = Depends(get_current_user)):
    return {
        "app": "OpsMind AI",
        "version": "1.0.0",
        "agents": ["supervisor", "metrics_agent", "anomaly_agent", "predict_agent", "rag_agent", "report_agent"],
        "ml_models": ["isolation_forest", "xgboost_classifier", "lstm_autoencoder"],
        "auth": "OAuth2 + JWT + RBAC",
        "llm_backend": "Ollama (llama3.2)",
        "vector_store": "ChromaDB",
        "requesting_user": current_user.username,
        "role": current_user.role.value,
    }


@app.get("/system/audit-logs", tags=["System"],
         summary="View recent API audit logs (admin only)")
async def audit_logs(
    limit: int = 50,
    current_user: UserInDB = Depends(require_permission("read:audit_logs")),
):
    return {"logs": REQUEST_LOG[-limit:], "total": len(REQUEST_LOG)}


# ─────────────────────────────────────────
# 10. ENTRYPOINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
