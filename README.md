
# 🧠 AutoResearch MLOps Pipeline

An agentic AutoML + MLOps system that combines:

* Hyperparameter optimization (Optuna)
* Multi-model training (Neural Network + XGBoost)
* Experiment tracking (MLflow)
* LLM-guided configuration refinement using **Ollama (Gemma 2B)**
* Memory-driven learning loop

This project demonstrates a self-improving ML system that learns from past experiments and continuously optimizes itself.

---

# 🚀 Features

* 🔁 AutoML loop using Optuna
* 🧠 LLM-guided hyperparameter tuning (Gemma 2B via Ollama)
* 🗂 Persistent memory of past experiments
* 📊 MLflow experiment tracking
* ⚖️ Multi-model comparison (NN vs XGBoost)
* ⏹ Early stopping logic
* 🔄 Continuous improvement system

---

# 🏗 Project Structure

autoresearch_mlops/

* core.py          → Main orchestration (Optuna loop)
* main.py          → Entry point
* models.py        → NN + XGBoost training
* eval.py          → Evaluation (RMSE)
* tracker.py       → MLflow logging
* memory.py        → Stores past runs
* llm.py           → LLM (Ollama integration)
* program.md       → Control logic for LLM
* mlflow.db        → MLflow database (auto-created)

---

## ⚙️ How It Works

User Data <br>
↓ <br>
Optuna Trial Loop <br>
↓ <br>
Config Generation (lr, epochs, model type) <br>
↓ <br>
LLM Suggestion (Gemma 2B via Ollama every few trials) <br>
↓ <br>
Model Training (NN or XGB) <br>
↓ <br>
Evaluation (RMSE) <br>
↓ <br>
MLflow Logging <br>
↓ <br>
Memory Update <br>
↓ <br>
Early Stopping Check <br>
↓ <br>
Repeat 
---

# 🧠 LLM Integration (Ollama + Gemma 2B)

This project uses a **local LLM** to guide hyperparameter tuning.

### Why Gemma 2B?

* Lightweight (runs locally)
* Fast inference
* No API cost
* Good enough for structured suggestions

---

## 🔧 Install Ollama

Download and install:

[https://ollama.com](https://ollama.com)

---

## 📥 Pull Gemma model

Run:

ollama pull gemma:2b

---

## ▶️ Run model

ollama run gemma:2b

---

## ⚙️ Start Ollama server

ollama serve

API will run at:

[http://localhost:11434](http://localhost:11434)

---

## 🧠 How LLM is used

* Reads past runs from memory
* Reads `program.md` (rules + strategy)
* Suggests better hyperparameters
* Injected into Optuna loop every few trials

---

# 📦 Installation

## 1. Clone Repository

git clone [https://github.com/your-username/autoresearch-mlops.git](https://github.com/your-username/autoresearch-mlops.git)
cd autoresearch-mlops

---

## 2. Create Virtual Environment

Using uv:

uv venv
uv pip install -r requirements.txt

OR using pip:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

---

## 3. Install Dependencies

pip install torch optuna mlflow xgboost pandas scikit-learn

---

# ▶️ Running the Project

## Step 1 — Start MLflow Server

(Use absolute path)

mlflow server --backend-store-uri sqlite:///C:/Autoreasearch/autoresearch_mlops/mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

---

## Step 2 — Start Ollama

ollama serve

---

## Step 3 — Run Training

uv run main.py

---

## Step 4 — Open MLflow UI

[http://127.0.0.1:5000](http://127.0.0.1:5000)

Select experiment:

autoresearch_mlops

---

# 📊 MLflow Output

* Experiment: autoresearch_mlops
* Runs = Optuna trials
* Metrics: RMSE
* Parameters:

  * model type
  * learning rate
  * epochs
* Model artifacts

---

# ⚠️ Common Issues & Fixes

## ❌ Experiment not showing

Fix:
Use same tracking URI everywhere

mlflow.set_tracking_uri("sqlite:///C:/Autoresearch/autoresearch_mlops/mlflow.db")

---

## ❌ Only "Default" experiment visible

Cause:
Training and server using different DB

Fix:
Use absolute path

---

## ❌ Ollama not responding

Check:

* ollama serve is running
* Port 11434 is accessible

---

## ❌ LLM not influencing results

Check:

* llm.py is being called
* program.md has meaningful rules
* memory is being updated

---

## ❌ Model logging warning (predict missing)

Cause:
PyTorch model logged with sklearn API

Fix (optional):

Use mlflow.pytorch for NN models

---

# 💡 Concept

Self-improving ML systems using:

* Feedback loops
* Memory
* LLM reasoning

This moves beyond static pipelines into:

* Adaptive systems
* Agentic AI
* Continuous learning pipelines

---

# 📜 License

MIT License

---


Just tell 👍
