# 🚀 OpsMind AI — Deployment Guide

## Option 1: Google Colab (Recommended for Portfolio Demo)

**Free, no credit card, GPU available**

1. Open `colab/OpsMind_AI.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run cells in order (takes ~10 minutes total)
4. Cell 7 gives you a **public ngrok URL** to share with interviewers

---

## Option 2: Railway.app (Free Tier — Best for Always-On)

**Gives you a real domain like `opsmind.railway.app`**

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Init project
railway init

# 4. Deploy API
railway up --service api

# 5. Set environment variables in Railway dashboard:
#    SECRET_KEY=<your-secret>
#    OLLAMA_BASE_URL=https://your-ollama-instance
#    APP_ENV=production

# 6. Get your URL
railway domain
```

**Note:** Ollama won't run on Railway free tier (needs GPU/lots of RAM).
Use `groq.com` API as a free alternative LLM:
```python
# Replace in agents/graph.py:
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192", api_key="your-groq-api-key")
# Groq is FREE and very fast — perfect for deployment
```

---

## Option 3: Render.com (Free Web Service)

```yaml
# render.yaml
services:
  - type: web
    name: opsmind-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: SECRET_KEY
        generateValue: true
      - key: APP_ENV
        value: production

  - type: web
    name: opsmind-ui
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run streamlit_app/app.py --server.port $PORT --server.address 0.0.0.0
```

Deploy with: `render deploy`

---

## Option 4: Docker Locally

```bash
# Copy env file
cp .env.example .env

# Build and start all services
docker-compose up --build

# Access:
#   Streamlit UI : http://localhost:8501
#   FastAPI Docs : http://localhost:8000/docs
#   Ollama       : http://localhost:11434
```

---

## Option 5: Kubernetes (Minikube — Local)

```bash
# Start Minikube
minikube start --memory=4096 --cpus=4

# Build images
docker build -f docker/Dockerfile.api -t opsmind-api:latest .
docker build -f docker/Dockerfile.streamlit -t opsmind-streamlit:latest .

# Load images into Minikube
minikube image load opsmind-api:latest
minikube image load opsmind-streamlit:latest

# Deploy
kubectl apply -f docker/k8s-deploy.yaml

# Get URL
minikube service opsmind-ui-service -n opsmind --url
```

---

## Free LLM Alternatives to Ollama (for cloud deploy)

| Provider | Free Tier | Models |
|----------|-----------|--------|
| **Groq** | 14,400 req/day | Llama 3, Mixtral |
| **Google AI Studio** | 1,500 req/day | Gemini 1.5 Flash |
| **Together.ai** | $5 free credits | Llama 3, Mistral |
| **Hugging Face** | Inference API | Many open models |

```python
# Switch to Groq (fastest, recommended for deploy):
pip install langchain-groq
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192", groq_api_key="gsk_...")
```

---

## Portfolio Presentation Tips

### What to highlight in interviews:
1. **Multi-agent orchestration** — LangGraph state machine with 6 agents
2. **Production security** — OAuth2 + JWT + RBAC with role-based permissions
3. **ML pipeline** — Isolation Forest + XGBoost + LSTM Autoencoder
4. **RAG knowledge base** — ChromaDB + semantic search for runbook retrieval
5. **Cloud-native** — Docker + Kubernetes-ready with HPA
6. **REST API design** — FastAPI with OpenAPI docs, middleware, audit logging

### Architecture diagram to draw on whiteboard:
```
User → Streamlit UI → FastAPI (OAuth2/JWT/RBAC)
                         ↓
              LangGraph Supervisor Agent
                ↙    ↓    ↓    ↓    ↘
           Metrics  Anomaly Predict RAG Report
           Agent    Agent   Agent   Agent Agent
                         ↓
              Ollama LLM (llama3.2)
              ChromaDB (vector store)
              ML Models (sklearn/xgb/tf)
```
