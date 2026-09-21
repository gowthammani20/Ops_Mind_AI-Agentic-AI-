# 🤖 OpsMind AI — Enterprise Agentic Operations Platform

> **Portfolio Project** — Built to demonstrate enterprise-grade Agentic AI engineering skills

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.45-green)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What This Project Demonstrates

| Skill Domain | Technologies Used |
|---|---|
| **Agentic AI** | LangGraph multi-agent orchestration, LangChain, Ollama |
| **Machine Learning** | Scikit-learn (Isolation Forest), XGBoost, anomaly detection |
| **Deep Learning** | TensorFlow, LSTM Autoencoder for time-series anomalies |
| **Enterprise Security** | OAuth2, JWT tokens, RBAC (5 roles), audit logging |
| **REST API** | FastAPI, Pydantic, middleware, OpenAPI/Swagger |
| **RAG** | ChromaDB vector store, Sentence Transformers, semantic search |
| **Cloud Native** | Docker, Docker Compose, Kubernetes (HPA, namespaces) |
| **UI/Dashboard** | Streamlit, Plotly interactive charts, real-time gauges |
| **MLOps** | Model training pipeline, artifact persistence, fallback logic |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (Port 8501)              │
│  Dashboard │ Anomaly │ Predictions │ Agent │ Chat │ Admin│
└────────────────────────┬────────────────────────────────┘
                         │ HTTP REST
┌────────────────────────▼────────────────────────────────┐
│              FastAPI REST API (Port 8000)                │
│   OAuth2 + JWT + RBAC   │   Audit Logging Middleware     │
│   /auth  /metrics  /anomaly  /predict  /agent  /health   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│           LangGraph Multi-Agent System                   │
│                                                          │
│  ┌──────────┐                                            │
│  │Supervisor│ ──routes──► Metrics → Anomaly → Predict   │
│  └──────────┘                              ↓             │
│                             RAG Agent → Report Agent     │
└────────────────────────┬────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
    ┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
    │Ollama LLM   │ │ChromaDB│ │ML Models   │
    │(llama3.2)   │ │(RAG)   │ │IsoForest   │
    │Port 11434   │ │        │ │XGBoost     │
    └─────────────┘ └────────┘ │LSTM Autoenc│
                               └────────────┘
```

---

## 🚀 Quick Start (Google Colab)

1. Open `colab/OpsMind_AI.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run cells 1–7 in order
4. Get your public URL from Cell 7 output

**Login:** `gowtham` / `gowtham2026` (Super Admin)

---

## 📁 Project Structure

```
opsmind/
├── auth/
│   └── auth_core.py        # OAuth2 + JWT + RBAC
├── agents/
│   └── graph.py            # LangGraph multi-agent system
├── api/
│   └── main.py             # FastAPI REST API (30+ endpoints)
├── ml/
│   └── models.py           # Isolation Forest + XGBoost + LSTM
├── streamlit_app/
│   └── app.py              # Streamlit dashboard
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.streamlit
│   └── k8s-deploy.yaml     # Kubernetes manifests + HPA
├── colab/
│   └── OpsMind_AI.ipynb    # Ready-to-run Colab notebook
├── docker-compose.yml
├── requirements.txt
├── DEPLOY.md               # Deployment options guide
└── README.md
```

---

## 🔐 RBAC Roles

| Role | Permissions |
|------|-------------|
| **super_admin** | Everything (`*`) |
| **admin** | Read/write all; manage users & agents; audit logs |
| **analyst** | Read all metrics; write predictions; audit logs |
| **viewer** | Read-only access to metrics, anomalies, predictions |
| **agent** | Machine-to-machine service account |

---

## 🤖 Agent Pipeline

```
User Task
    │
    ▼
Supervisor Agent (LLM routing)
    │
    ├── Metrics Agent      → Fetches & summarizes live metrics
    ├── Anomaly Agent      → Runs Isolation Forest detection
    ├── Predict Agent      → XGBoost failure probability
    ├── RAG Agent          → Retrieves relevant runbooks
    └── Report Agent       → Synthesizes final incident report
```

---

## 📊 ML Models

| Model | Type | Purpose |
|-------|------|---------|
| Isolation Forest | Unsupervised ML | Real-time anomaly detection |
| XGBoost Classifier | Supervised ML | Failure probability prediction |
| LSTM Autoencoder | Deep Learning | Sequential anomaly detection |

---

## 🛠️ Local Development

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/opsmind-ai.git
cd opsmind-ai

# 2. Install
pip install -r requirements.txt

# 3. Start Ollama (must have Ollama installed)
ollama serve &
ollama pull llama3.2

# 4. Train models
python -c "from ml.models import train_all_models; train_all_models()"

# 5. Start API
uvicorn api.main:app --port 8000 --reload

# 6. Start UI (new terminal)
streamlit run streamlit_app/app.py
```

---

## 📡 Key API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/token` | POST | None | Login → JWT tokens |
| `/auth/me` | GET | Any | Current user profile |
| `/metrics/live` | GET | Viewer+ | Real-time metrics |
| `/anomaly/detect` | POST | Viewer+ | Anomaly detection |
| `/predict/failure` | POST | Viewer+ | Failure prediction |
| `/agent/run` | POST | Analyst+ | Full agent pipeline |
| `/agent/chat` | POST | Any | Chatbot |
| `/system/audit-logs` | GET | Admin+ | Audit trail |

Interactive docs: `http://localhost:8000/docs`

---

## 👨‍💻 Built by Gowtham
*B.Tech AI & Data Science | Agentic AI Engineer (Intern @ PLM Indishtech)*
