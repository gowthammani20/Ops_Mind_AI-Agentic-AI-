"""
╔══════════════════════════════════════════════════════════════╗
║     OpsMind AI — Google Colab Setup & Runner                ║
║     Run this entire file cell by cell in Google Colab       ║
╚══════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
  1. Open Google Colab: https://colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU (free)
  3. Copy each cell below into a Colab cell and run in order
  4. The Streamlit UI will be accessible via ngrok public URL
"""

# ════════════════════════════════════════
# CELL 1 — Clone & Install
# ════════════════════════════════════════
CELL_1 = """
# Clone the project (after you push to GitHub)
# !git clone https://github.com/YOUR_USERNAME/opsmind-ai.git
# %cd opsmind-ai

# OR: create files inline (if not using GitHub)
import os
os.makedirs("opsmind", exist_ok=True)
%cd opsmind

# Install all dependencies
!pip install -q \\
    langchain==0.3.7 \\
    langchain-community==0.3.7 \\
    langchain-core==0.3.15 \\
    langgraph==0.2.45 \\
    langchain-ollama==0.2.0 \\
    fastapi==0.115.4 \\
    uvicorn[standard]==0.32.0 \\
    python-jose[cryptography]==3.3.0 \\
    passlib[bcrypt]==1.7.4 \\
    scikit-learn==1.5.2 \\
    xgboost==2.1.2 \\
    chromadb==0.5.20 \\
    sentence-transformers==3.3.0 \\
    streamlit==1.40.0 \\
    plotly==5.24.1 \\
    pydantic==2.9.2 \\
    pyngrok==7.2.0 \\
    requests==2.32.3 \\
    faiss-cpu==1.9.0 \\
    httpx==0.27.2

print("✅ Installation complete!")
"""

# ════════════════════════════════════════
# CELL 2 — Install & Start Ollama in Colab
# ════════════════════════════════════════
CELL_2 = """
# Install Ollama binary
!curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server in background
import subprocess
import time
import threading

def run_ollama():
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

t = threading.Thread(target=run_ollama, daemon=True)
t.start()
time.sleep(5)
print("✅ Ollama server started")

# Pull the model (choose based on your RAM)
# llama3.2 = 2GB     (fastest, good for Colab free tier)
# llama3.1 = 4.7GB   (better quality)
# mistral  = 4.1GB   (good alternative)
!ollama pull llama3.2

print("✅ Model pulled and ready!")
"""

# ════════════════════════════════════════
# CELL 3 — Train ML Models
# ════════════════════════════════════════
CELL_3 = """
import sys
sys.path.insert(0, '/content/opsmind')  # adjust path if needed

from ml.models import train_all_models

print("🏋️ Training ML models...")
ad, fp, lstm = train_all_models(save=True)
print("✅ All models trained!")
"""

# ════════════════════════════════════════
# CELL 4 — Initialize ChromaDB with Runbooks
# ════════════════════════════════════════
CELL_4 = """
import chromadb
from sentence_transformers import SentenceTransformer
import os

os.makedirs("data/chroma_db", exist_ok=True)
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection("opsmind_runbooks")

runbooks = [
    "HIGH CPU USAGE RUNBOOK: When CPU exceeds 85%, immediately check running processes with 'top'. "
    "Identify high-CPU processes and consider killing non-essential ones. "
    "Scale horizontally by adding more compute nodes. Enable CPU throttling for batch jobs. "
    "Check for infinite loops or recursive calls in application code.",

    "HIGH MEMORY USAGE RUNBOOK: When memory exceeds 90%, trigger garbage collection. "
    "Analyze heap dumps to identify memory leaks. Check connection pool exhaustion. "
    "Consider increasing JVM heap size or containerr memory limits. "
    "Restart services with graceful drain if memory cannot be reclaimed.",

    "HIGH ERROR RATE RUNBOOK: When error rate exceeds 15%, immediately check upstream dependencies. "
    "Enable circuit breaker pattern to prevent cascading failures. "
    "Roll back recent deployment if error rate correlates with deploy time. "
    "Check database connection pools and external API rate limits.",

    "HIGH NETWORK LATENCY RUNBOOK: Latency above 500ms requires immediate routing check. "
    "Verify DNS resolution and CDN configuration. Check for network congestion or packet loss. "
    "Consider enabling compression for API responses. Review load balancer health checks.",

    "INCIDENT RESPONSE PROTOCOL: P1 incidents require immediate escalation within 5 minutes. "
    "Create incident channel in Slack. Assign incident commander. "
    "Conduct 15-minute sync calls every 30 minutes. Document all actions taken. "
    "Post-incident review required within 48 hours of resolution.",

    "KUBERNETES SCALING RUNBOOK: When HPA triggers, verify node capacity first. "
    "Check PodDisruptionBudget constraints. Monitor rollout with kubectl rollout status. "
    "Ensure resource requests/limits are properly set to avoid OOMKilled pods.",
]

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(runbooks).tolist()

collection.add(
    documents=runbooks,
    ids=[f"runbook_{i}" for i in range(len(runbooks))],
    embeddings=embeddings,
)
print(f"✅ ChromaDB initialized with {len(runbooks)} runbooks!")
"""

# ════════════════════════════════════════
# CELL 5 — Start FastAPI in Background
# ════════════════════════════════════════
CELL_5 = """
import subprocess
import time
import threading

def run_api():
    subprocess.Popen(
        ["python", "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
        stdout=open("api.log", "w"),
        stderr=open("api_err.log", "w"),
    )

t = threading.Thread(target=run_api, daemon=True)
t.start()
time.sleep(8)

# Test the API
import requests
try:
    r = requests.get("http://localhost:8000/health")
    print("✅ FastAPI running:", r.json())
except Exception as e:
    print("❌ API not ready yet:", e)
    print("Check logs: !cat api_err.log")
"""

# ════════════════════════════════════════
# CELL 6 — Start Streamlit + ngrok Tunnel
# ════════════════════════════════════════
CELL_6 = """
import subprocess
import threading
import time
from pyngrok import ngrok

# Start Streamlit
def run_streamlit():
    subprocess.Popen(
        ["streamlit", "run", "streamlit_app/app.py",
         "--server.port=8501",
         "--server.headless=true",
         "--browser.gatherUsageStats=false"],
        stdout=open("streamlit.log", "w"),
        stderr=open("streamlit_err.log", "w"),
    )

t = threading.Thread(target=run_streamlit, daemon=True)
t.start()
time.sleep(10)

# Create ngrok tunnel (free, no account needed for basic)
# For persistent URL: sign up at ngrok.com and set authtoken
# !ngrok authtoken YOUR_TOKEN

public_url = ngrok.connect(8501)
api_url    = ngrok.connect(8000)

print("=" * 55)
print("🌐 STREAMLIT UI:", public_url)
print("📡 FASTAPI DOCS:", api_url, "/docs")
print("=" * 55)
print()
print("🔑 LOGIN CREDENTIALS:")
print("  gowtham   / gowtham2026  (Super Admin)")
print("  admin     / admin123     (Admin)")
print("  analyst   / analyst123   (Analyst)")
print("  viewer    / viewer123    (Viewer)")
"""

# ════════════════════════════════════════
# CELL 7 — Quick Test (no UI)
# ════════════════════════════════════════
CELL_7 = """
# Quick test without running full UI
import requests

BASE = "http://localhost:8000"

# 1. Login
resp = requests.post(f"{BASE}/auth/token",
    data={"username": "gowtham", "password": "gowtham2026",
          "grant_type": "password"})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("✅ Login:", resp.json()["role"])

# 2. Fetch live metrics
metrics = requests.get(f"{BASE}/metrics/live", headers=headers).json()
print("📊 Metrics:", {k: round(v, 2) for k, v in metrics.items() if isinstance(v, (int, float))})

# 3. Anomaly detection
anomaly_payload = {**metrics, "service": "test"}
anomaly_payload.pop("timestamp", None)
result = requests.post(f"{BASE}/anomaly/detect", json=anomaly_payload, headers=headers).json()
print("🚨 Anomaly:", result)

# 4. Failure prediction
pred = requests.post(f"{BASE}/predict/failure", json=anomaly_payload, headers=headers).json()
print("🔮 Prediction:", pred)

print("\\n✅ All API endpoints working!")
"""

# ════════════════════════════════════════
# CELL 8 — Run Agent (Standalone, no Ollama needed for test)
# ════════════════════════════════════════
CELL_8 = """
# Test agent pipeline directly
from agents.graph import run_agent

print("🧠 Running LangGraph multi-agent pipeline...")
print("(This requires Ollama to be running)")
print()

result = run_agent(
    task="Analyze current system health and provide a detailed incident report",
    user_role="analyst"
)

print("=" * 60)
print(result.get("report", "No report generated"))
print("=" * 60)
print("Iterations:", result.get("iterations"))
"""

# Print all cells
if __name__ == "__main__":
    print(__doc__)
    cells = {
        "CELL 1 — Install Dependencies": CELL_1,
        "CELL 2 — Install & Start Ollama": CELL_2,
        "CELL 3 — Train ML Models": CELL_3,
        "CELL 4 — Initialize ChromaDB": CELL_4,
        "CELL 5 — Start FastAPI": CELL_5,
        "CELL 6 — Start Streamlit + ngrok": CELL_6,
        "CELL 7 — Quick API Test": CELL_7,
        "CELL 8 — Run Agent": CELL_8,
    }
    for name, code in cells.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(code)
