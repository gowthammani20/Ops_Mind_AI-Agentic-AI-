<h1 align="center">🤖 OpsMind AI</h1>
<p align="center">
  <strong>Enterprise Agentic Operations Intelligence Platform</strong><br/>
  LangGraph · LangChain · Ollama · FastAPI · Scikit-learn · XGBoost · ChromaDB · Streamlit
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-0.2.45-4B8BBE?style=flat"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-2.1-EB6100?style=flat"/>
  <img src="https://img.shields.io/badge/Ollama-llama3.2-black?style=flat"/>
  <img src="https://img.shields.io/badge/Auth-OAuth2%20%2B%20JWT%20%2B%20RBAC-green?style=flat"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat"/>
</p>

---

## 📌 Overview

**OpsMind AI** is a production-grade, end-to-end Agentic AI system built to autonomously monitor IT operations, detect anomalies, predict system failures, retrieve knowledge from a vector database, and synthesize everything into a structured incident report — all orchestrated by a **LangGraph multi-agent state machine** running on a **local Ollama LLM**.

This project is designed to demonstrate real-world skills across **Agentic AI engineering**, **ML/DL model development**, **enterprise REST API design**, **security architecture**, and **cloud-native deployment** — the exact combination that differentiates AI engineers in MNC hiring.

> Built by **Gowtham** — B.Tech AI & Data Science (2023–2027), SSM Institute of Engineering and Technology | AI Engineering Intern @ PLM Indishtech Pvt. Ltd., Bangalore

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit UI  (Port 8501)                    │
│   Dashboard │ Anomaly │ Predictions │ Agent │ Chat │ Admin      │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP / REST
┌────────────────────────▼────────────────────────────────────────┐
│               FastAPI REST API  (Port 8000)                     │
│      OAuth2 Bearer + JWT + RBAC   │   Audit Logging Middleware  │
│  /auth  /metrics  /anomaly  /predict  /agent  /system  /health  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│             LangGraph Multi-Agent Orchestrator                  │
│                                                                 │
│   ┌───────────┐                                                 │
│   │ Supervisor│──routes──► Metrics ──► Anomaly ──► Predict     │
│   └───────────┘                                   ▼            │
│                                          RAG ──► Report        │
└───────────────────┬──────────────────────┬──────────────────────┘
                    │                      │
       ┌────────────▼──────┐   ┌───────────▼────────────────┐
       │  Ollama LLM       │   │  ML / DL Models            │
       │  llama3.2         │   │  • Isolation Forest        │
       │  Port 11434       │   │  • XGBoost Classifier      │
       └───────────────────┘   │  • LSTM Autoencoder (opt.) │
                               └────────────────────────────┘
                    │
       ┌────────────▼──────────────┐
       │  ChromaDB Vector Store    │
       │  Sentence Transformers    │
       │  (RAG Knowledge Base)     │
       └───────────────────────────┘
```

---

## ✨ Features

### 🧠 Agentic AI (LangGraph)
- **6-agent state machine** with LLM-driven routing: Supervisor → Metrics → Anomaly → Predict → RAG → Report
- Shared typed state (`AgentState`) passed across all agents
- Loop-guard (max 8 iterations), rule-based fallback when ML models are not yet trained
- Each agent uses a dedicated system prompt and temperature

### 🚨 Anomaly Detection (ML + DL)
- **Isolation Forest** — unsupervised, no labels needed, detects point anomalies in real time
- **LSTM Autoencoder** — deep learning sequential anomaly detection via reconstruction error (optional, requires TensorFlow)
- Anomaly severity scoring: `normal → medium → high → critical`

### 🔮 Predictive Analytics (XGBoost)
- **XGBoost** binary classifier predicts failure probability (0–1) for the next 30 minutes
- Feature engineering: rolling trend features (cpu_trend, error_trend) derived from time-series
- Full classification report + AUC score logged at training time
- Feature importance API endpoint for model explainability

### 🗄️ RAG Knowledge Base (ChromaDB)
- Operational runbooks embedded with **Sentence Transformers** (`all-MiniLM-L6-v2`)
- Semantic similarity search at query time — retrieves top-3 relevant runbooks per task
- Graceful fallback to hardcoded runbooks if ChromaDB is unavailable

### 🔐 Enterprise Security
- **OAuth2 Password Flow** with Bearer token issuance
- **JWT** access tokens (30-min expiry) + refresh tokens (7-day expiry), signed with HS256
- **RBAC** — 5 roles with granular permission matrix:

  | Role | Permissions |
  |------|-------------|
  | `super_admin` | All (`*`) |
  | `admin` | Read/write metrics, anomalies, predictions; manage users; audit logs |
  | `analyst` | Read all + write predictions + audit logs |
  | `viewer` | Read-only: metrics, anomalies, predictions |
  | `agent` | M2M service account: read/write metrics & anomalies |

- HTTP audit logging middleware — every request logged with path, method, status, latency
- Dependency injection guards on every protected endpoint

### ⚡ REST API (FastAPI)
- 17 endpoints across 5 groups: Auth, Metrics, Anomaly, Predictions, Agent, System
- Auto-generated **Swagger UI** at `/docs` and **ReDoc** at `/redoc`
- GZip middleware, CORS middleware
- Pydantic v2 request/response validation

### 🎨 Streamlit Dashboard
- **Login page** with role display
- **Live Dashboard** — KPI metrics, Plotly gauge charts, historical trend line chart
- **Anomaly tab** — slider controls, score bar chart, severity badge
- **Predictions tab** — failure probability gauge, XGBoost feature importance bar chart
- **Agent tab** — task selector, full report rendered in Markdown, raw JSON expander
- **Chatbot tab** — multi-turn conversation with Ollama via `/agent/chat`, conversation history
- **Admin tab** — user management, audit log table, RBAC reference (admin/super_admin only)

---

## 📁 Project Structure

```
opsmind-ai/
│
├── auth/
│   ├── __init__.py
│   └── auth_core.py          # OAuth2, JWT, RBAC, user store, FastAPI deps
│
├── ml/
│   ├── __init__.py
│   ├── models.py             # IsolationForest + XGBoost + synthetic data generator
│   └── artifacts/            # Saved model .pkl files (created after training)
│
├── agents/
│   ├── __init__.py
│   └── graph.py              # LangGraph state machine — all 6 agents + router
│
├── api/
│   ├── __init__.py
│   └── main.py               # FastAPI app — 17 REST endpoints
│
├── streamlit_app/
│   ├── __init__.py
│   └── app.py                # Streamlit dashboard — 6 tabs
│
├── data/
│   └── chroma_db/            # ChromaDB persisted vector store (created at runtime)
│
├── OpsMind_AI_Colab.ipynb    # Google Colab notebook — run everything in 9 cells
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start — Google Colab (Recommended)

> No local setup needed. Runs entirely in Colab free tier (T4 GPU).

**Step 1:** Open `OpsMind_AI_Colab.ipynb` in [Google Colab](https://colab.research.google.com)

**Step 2:** Runtime → Change runtime type → **T4 GPU**

**Step 3:** Run cells **1 through 7** in order

**Step 4:** Cell 7 prints a **public ngrok URL** — open it in your browser

| Cell | What it does |
|------|--------------|
| 1 | Installs all pip dependencies |
| 2 | Creates folder structure, sets `sys.path` (fixes `ModuleNotFoundError`) |
| 3 | Writes all source files to disk |
| 4 | Installs Ollama, starts server, pulls `llama3.2` (~2 GB) |
| 5 | Trains ML models + seeds ChromaDB with 6 operational runbooks |
| 6 | Starts FastAPI on port 8000, polls until healthy |
| 7 | Starts Streamlit on port 8501, creates ngrok public tunnel |
| 8 | Smoke test — verifies login, metrics, anomaly, RBAC |
| 9 | (Optional) Runs the full LangGraph agent pipeline |

---

## 💻 Local Setup

**Prerequisites:** Python 3.11+, [Ollama installed](https://ollama.com)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/opsmind-ai.git
cd opsmind-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama and pull the model
ollama serve &
ollama pull llama3.2

# 4. Train ML models
python -c "from ml.models import train_all_models; train_all_models()"

# 5. Start FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start Streamlit (new terminal)
streamlit run streamlit_app/app.py
```

Open:
- **Streamlit UI** → http://localhost:8501
- **API Swagger** → http://localhost:8000/docs

---

## 🔑 Login Credentials

| Username | Password | Role |
|----------|----------|------|
| `gowtham` | `gowtham2026` | Super Admin |
| `admin` | `admin123` | Admin |
| `analyst` | `analyst123` | Analyst |
| `viewer` | `viewer123` | Viewer |

---

## 📡 API Reference

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| `POST` | `/auth/token` | None | Login → JWT access + refresh tokens |
| `GET` | `/auth/me` | Any | Current user profile |
| `POST` | `/auth/register` | Admin+ | Create a new user |
| `GET` | `/auth/users` | Admin+ | List all users |
| `GET` | `/metrics/live` | Viewer+ | Real-time metrics snapshot |
| `GET` | `/metrics/history` | Viewer+ | Historical metrics (simulated) |
| `POST` | `/metrics/ingest` | Agent+ | Ingest a metrics payload |
| `POST` | `/anomaly/detect` | Viewer+ | Run Isolation Forest on metrics |
| `GET` | `/anomaly/status` | Viewer+ | Current system anomaly status |
| `POST` | `/predict/failure` | Viewer+ | XGBoost failure probability |
| `GET` | `/predict/feature-importance` | Viewer+ | XGBoost feature importance |
| `POST` | `/agent/run` | Analyst+ | Run full LangGraph pipeline |
| `POST` | `/agent/chat` | Any | Multi-turn chatbot (Ollama) |
| `GET` | `/health` | None | Health check |
| `GET` | `/system/info` | Any | Platform info |
| `GET` | `/system/audit-logs` | Admin+ | Recent request audit log |

Interactive docs: `http://localhost:8000/docs`

---

## 🤖 Agent Pipeline — How It Works

```
User Task  ──►  Supervisor Agent (LLM decides next step)
                      │
          ┌───────────┼───────────┬────────────┬────────────┐
          ▼           ▼           ▼            ▼            ▼
    Metrics Agent  Anomaly    Predict       RAG Agent   Report Agent
    (fetch &       Agent      Agent         (ChromaDB   (synthesize
    summarize)     (Isolation  (XGBoost     semantic    full Markdown
                   Forest)     classifier)  search)     report)
          │           │           │            │
          └───────────┴───────────┴────────────┘
                      │
              Back to Supervisor
              (until report ready)
```

The Supervisor uses the LLM to decide which agent runs next based on what's already been completed in the shared `AgentState`. Each specialist agent updates the state and returns to the Supervisor. The pipeline ends when the Report Agent writes the final incident report.

---

## 🧪 Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Agent Framework | LangGraph + LangChain | 0.2.45 / 0.3.7 |
| LLM Backend | Ollama (llama3.2) | latest |
| ML — Anomaly | Scikit-learn Isolation Forest | 1.5.2 |
| ML — Prediction | XGBoost Classifier | 2.1.2 |
| DL — Anomaly | LSTM Autoencoder (TensorFlow) | optional |
| Vector Store | ChromaDB + Sentence Transformers | 0.5.20 |
| REST API | FastAPI + Uvicorn | 0.115.4 |
| Auth | python-jose (JWT) + passlib (bcrypt) | 3.3.0 |
| UI | Streamlit + Plotly | 1.40.0 |
| Data | Pandas + NumPy | 2.2.3 / 1.26.4 |
| Colab Tunnel | pyngrok | 7.2.0 |

---

## 🎯 Skills Demonstrated

This project was built specifically to showcase the following in MNC interviews:

| Domain | What's implemented |
|--------|-------------------|
| **Agentic AI** | LangGraph state machine, multi-agent routing, shared typed state, loop protection |
| **ML Engineering** | Unsupervised anomaly detection, supervised failure prediction, model persistence, fallback logic |
| **Deep Learning** | LSTM Autoencoder for time-series anomaly detection via reconstruction error |
| **RAG** | ChromaDB vector store, semantic embedding, top-k retrieval for knowledge augmentation |
| **API Design** | RESTful design, Pydantic validation, middleware, OpenAPI/Swagger docs |
| **Security** | OAuth2 password flow, JWT issuance & validation, 5-tier RBAC, dependency injection |
| **Data Engineering** | Synthetic time-series generation, feature engineering, trend features |
| **UI / DataViz** | Streamlit multi-tab app, Plotly gauges, live charts, interactive controls |
| **MLOps** | Model training pipeline, artifact persistence, model loading with fallback |
| **Cloud / DevOps** | Google Colab deployment, ngrok tunneling, `sys.path` management |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">
  Built with 🧠 by <strong>Gowtham</strong> · B.Tech AI & Data Science · SSM Institute of Engineering and Technology<br/>
  AI Engineering Intern @ PLM Indishtech Pvt. Ltd., Bangalore
</p>
